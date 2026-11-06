from __future__ import annotations

import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset

from .labels import IGNORE_INDEX
from .sar_features import dual_pol_torch
from .optical_indices import OPTICAL_INDEX_NAMES, bare_soil_indices
from .text import encode_text


class LandUsePatchDataset(Dataset):
    def __init__(
        self,
        image_path: str | Path,
        label_path: str | Path,
        manifest_path: str | Path,
        vocab_path: str | Path,
        normalization_path: str | Path,
        patch_size: int,
        max_text_tokens: int,
        osm_poi_path: str | Path | None = None,
        osm_aoi_path: str | Path | None = None,
        use_poi: bool = True,
        use_aoi: bool = True,
        highres_path: str | Path | None = None,
        nighttime_path: str | Path | None = None,
        building_path: str | Path | None = None,
        road_features_path: str | Path | None = None,
        building_feature_mode: str = "full",
        split: str | None = None,
        augment: bool = False,
    ) -> None:
        self.image_path = str(image_path)
        self.label_path = str(label_path)
        frame = pd.read_csv(manifest_path)
        if split is not None:
            frame = frame[frame["split"] == split]
        self.frame = frame.reset_index(drop=True)
        self.vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        stats = json.loads(Path(normalization_path).read_text(encoding="utf-8"))["bands"]
        self.clip_min = np.array([x["clip_min"] for x in stats], dtype=np.float32)[:, None, None]
        self.clip_max = np.array([x["clip_max"] for x in stats], dtype=np.float32)[:, None, None]
        self.mean = np.array([x["mean"] for x in stats], dtype=np.float32)[:, None, None]
        self.std = np.array([x["std"] for x in stats], dtype=np.float32)[:, None, None]
        self.patch_size = patch_size
        self.max_text_tokens = max_text_tokens
        self.osm_poi_path = str(osm_poi_path) if osm_poi_path else None
        self.osm_aoi_path = str(osm_aoi_path) if osm_aoi_path else None
        self.use_poi = bool(use_poi)
        self.use_aoi = bool(use_aoi)
        self.highres_path = str(highres_path) if highres_path else None
        self.nighttime_path = str(nighttime_path) if nighttime_path else None
        self.building_path = str(building_path) if building_path else None
        self.road_features_path = str(road_features_path) if road_features_path else None
        if building_feature_mode not in {"full", "function_only", "morphology_only", "none"}:
            raise ValueError(
                "building_feature_mode must be one of full, function_only, "
                "morphology_only or none"
            )
        self.building_feature_mode = building_feature_mode
        self.augment = augment
        self._sources: dict[str, rasterio.io.DatasetReader] = {}
        with rasterio.open(self.image_path) as src:
            descriptions = list(src.descriptions)
        self.s2_indices = [i for i, name in enumerate(descriptions) if name and name.startswith("S2_")]
        self.s1_indices = [i for i, name in enumerate(descriptions) if name and name.startswith("S1_")]
        if not self.s2_indices or not self.s1_indices:
            raise ValueError(f"Could not identify S2/S1 bands from descriptions={descriptions}")
        if len(stats) != len(descriptions):
            raise ValueError(
                f"Normalization has {len(stats)} bands but image has {len(descriptions)}. "
                "Re-run prepare_data after changing the image."
            )
        self.s1_output_channels = len(self.s1_indices) + 4
        self.s2_output_channels = len(self.s2_indices) + len(OPTICAL_INDEX_NAMES)
        self.road_channels = 13 if self.road_features_path else 0
        self.building_channels = 0
        if self.building_path:
            with rasterio.open(self.building_path) as building_src:
                self.building_channels = building_src.count
        s2_names = [descriptions[i] for i in self.s2_indices]
        required = (
            "S2_B2_Blue",
            "S2_B3_Green",
            "S2_B4_Red",
            "S2_B8_NIR",
            "S2_B11_SWIR1",
            "S2_B12_SWIR2",
        )
        missing = [name for name in required if name not in s2_names]
        if missing:
            raise ValueError(f"Missing S2 bands required for indices: {missing}")
        self.s2_local_index = {name: s2_names.index(name) for name in required}
        self.urban_channels = (
            (3 if self.highres_path else 0)
            + (3 if self.nighttime_path else 0)
            + self.building_channels
        )

    def _source(self, path: str) -> rasterio.io.DatasetReader:
        source = self._sources.get(path)
        if source is None or source.closed:
            source = rasterio.open(path)
            self._sources[path] = source
        return source

    def _read_auxiliary(self, path: str, window: Window, count: int) -> np.ndarray:
        data = self._source(path).read(
            indexes=list(range(1, count + 1)),
            window=window,
            boundless=True,
            fill_value=0,
            out_dtype="float32",
        )
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_sources"] = {}
        return state

    def __del__(self) -> None:
        for source in getattr(self, "_sources", {}).values():
            source.close()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[index]
        window = Window(int(row.col_off), int(row.row_off), self.patch_size, self.patch_size)
        image = self._source(self.image_path).read(
            window=window,
            boundless=True,
            fill_value=np.nan,
            out_dtype="float32",
        )
        label = self._source(self.label_path).read(
            1,
            window=window,
            boundless=True,
            fill_value=IGNORE_INDEX,
            out_dtype="uint8",
        )
        raw_s1 = image[self.s1_indices].copy()
        raw_s1 = np.where(np.isfinite(raw_s1), raw_s1, self.mean[self.s1_indices])
        raw_s2 = image[self.s2_indices].copy()
        raw_s2 = np.where(np.isfinite(raw_s2), raw_s2, self.mean[self.s2_indices])
        optical_indices = bare_soil_indices(
            raw_s2[self.s2_local_index["S2_B2_Blue"]],
            raw_s2[self.s2_local_index["S2_B3_Green"]],
            raw_s2[self.s2_local_index["S2_B4_Red"]],
            raw_s2[self.s2_local_index["S2_B8_NIR"]],
            raw_s2[self.s2_local_index["S2_B11_SWIR1"]],
            raw_s2[self.s2_local_index["S2_B12_SWIR2"]],
        )
        image = np.clip(image, self.clip_min, self.clip_max)
        image = np.where(np.isfinite(image), image, self.mean)
        image = (image - self.mean) / self.std

        flip_h = self.augment and np.random.random() < 0.5
        flip_v = self.augment and np.random.random() < 0.5
        if self.augment:
            if flip_h:
                image = image[:, :, ::-1]
                raw_s1 = raw_s1[:, :, ::-1]
                optical_indices = optical_indices[:, :, ::-1]
                label = label[:, ::-1]
            if flip_v:
                image = image[:, ::-1, :]
                raw_s1 = raw_s1[:, ::-1, :]
                optical_indices = optical_indices[:, ::-1, :]
                label = label[::-1, :]
        image = np.ascontiguousarray(image)
        raw_s1 = np.ascontiguousarray(raw_s1)
        optical_indices = np.ascontiguousarray(optical_indices)
        label = np.ascontiguousarray(label)
        token_ids, text_mask = encode_text(str(row.text), self.vocab, self.max_text_tokens)
        s1 = torch.from_numpy(image[self.s1_indices])
        raw_s1_tensor = torch.from_numpy(np.ascontiguousarray(raw_s1))
        s1 = torch.cat([s1, dual_pol_torch(raw_s1_tensor[0], raw_s1_tensor[1])], dim=0)
        sample = {
            "s2": torch.cat(
                [
                    torch.from_numpy(image[self.s2_indices]),
                    torch.from_numpy(optical_indices),
                ],
                dim=0,
            ),
            "s1": s1,
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "text_mask": torch.tensor(text_mask, dtype=torch.bool),
            "label": torch.from_numpy(label.astype(np.int64)),
            "patch_id": str(row.patch_id),
        }
        urban_parts: list[np.ndarray] = []
        if self.highres_path:
            highres = self._read_auxiliary(self.highres_path, window, 3)
            highres = np.clip(highres / 127.5 - 1.0, -1.0, 1.0)
            urban_parts.append(highres)
        if self.nighttime_path:
            nighttime = self._read_auxiliary(self.nighttime_path, window, 3)
            # VIIRS radiance is heavy-tailed; band 2 is the source log1p layer.
            nighttime[0] = np.clip(np.log1p(np.maximum(nighttime[0], 0.0)) / 5.0, 0.0, 1.25)
            nighttime[1] = np.clip(nighttime[1] / 5.0, 0.0, 1.25)
            nighttime[2] = np.clip(nighttime[2] / 35.0, 0.0, 1.25)
            urban_parts.append(nighttime)
        if self.building_path:
            buildings = self._read_auxiliary(
                self.building_path, window, self.building_channels
            )
            # Bands 1--9 are transferred building-function masks.  The new
            # height product then stores coverage, mean/max height, an
            # FAR-like proxy and height range.  Fixed physical scales keep
            # inference deterministic and make the proxy explicit rather
            # than treating it as an observed planning FAR.
            buildings[: min(9, self.building_channels)] = np.clip(
                buildings[: min(9, self.building_channels)], 0.0, 1.0
            )
            if self.building_channels >= 10:
                buildings[9] = np.clip(buildings[9], 0.0, 1.0)
            if self.building_channels >= 11:
                buildings[10] = np.clip(buildings[10] / 100.0, 0.0, 3.0)
            if self.building_channels >= 12:
                buildings[11] = np.clip(buildings[11] / 100.0, 0.0, 3.0)
            if self.building_channels >= 13:
                buildings[12] = np.clip(buildings[12] / 10.0, 0.0, 5.0)
            if self.building_channels >= 14:
                buildings[13] = np.clip(buildings[13] / 100.0, 0.0, 3.0)
            # Keep the full 14-channel tensor geometry for controlled lineage
            # ablations.  Only the selected information group is zeroed, so the
            # common 95->96->48 fusion architecture is unchanged across runs.
            if self.building_feature_mode == "function_only":
                buildings[9:] = 0.0
            elif self.building_feature_mode == "morphology_only":
                buildings[:9] = 0.0
            elif self.building_feature_mode == "none":
                buildings[:] = 0.0
            urban_parts.append(buildings)
        if urban_parts:
            urban = np.concatenate(urban_parts, axis=0)
            if flip_h:
                urban = urban[:, :, ::-1]
            if flip_v:
                urban = urban[:, ::-1, :]
            sample["urban"] = torch.from_numpy(np.ascontiguousarray(urban))
        if self.road_features_path:
            roads = self._read_auxiliary(self.road_features_path, window, 13)
            roads[:11] = np.clip(roads[:11], 0.0, 1.0)
            roads[11:] = np.clip(roads[11:] / 255.0, 0.0, 1.0)
            if flip_h:
                roads = roads[:, :, ::-1]
            if flip_v:
                roads = roads[:, ::-1, :]
            sample["roads"] = torch.from_numpy(np.ascontiguousarray(roads))
        if self.osm_poi_path and self.osm_aoi_path:
            poi = self._source(self.osm_poi_path).read(
                window=window,
                boundless=True,
                fill_value=0,
                out_dtype="float32",
            )
            aoi = self._source(self.osm_aoi_path).read(
                1,
                window=window,
                boundless=True,
                fill_value=0,
                out_dtype="uint8",
            )
            poi = np.clip(np.log1p(poi) / np.log(10.0), 0.0, 2.0)
            aoi_onehot = np.eye(7, dtype=np.float32)[np.clip(aoi, 0, 6)][..., 1:].transpose(2, 0, 1)
            # Keep the 20-channel tensor geometry fixed while permitting
            # genuinely separate POI and AOI ablations.  Earlier modality
            # selection loaded both rasters whenever either was requested,
            # which made -POI and -AOI experiments indistinguishable.
            if not self.use_poi:
                poi[...] = 0.0
            if not self.use_aoi:
                aoi_onehot[...] = 0.0
            osm = np.ascontiguousarray(np.concatenate([poi, aoi_onehot], axis=0))
            if flip_h:
                osm = osm[:, :, ::-1]
            if flip_v:
                osm = osm[:, ::-1, :]
            osm = np.ascontiguousarray(osm)
            sample["osm"] = torch.from_numpy(osm)
        return sample
