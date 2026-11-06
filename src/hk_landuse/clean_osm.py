from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box

from .config import load_config


SEMANTIC_KEYS = (
    "amenity",
    "landuse",
    "building",
    "shop",
    "office",
    "leisure",
    "tourism",
    "railway",
    "public_transport",
    "highway",
    "industrial",
    "man_made",
    "aeroway",
    "sport",
    "healthcare",
)


def _safe_tags(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _token(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9_:=+.-]+", "_", text)
    return text.strip("_")


def _english_name(tags: dict[str, Any], fallback: Any) -> str:
    for key in ("name:en", "brand:en", "operator:en"):
        value = tags.get(key)
        if isinstance(value, str) and value.strip():
            return re.sub(r"\s+", " ", value).strip()
    ascii_parts = re.findall(r"[A-Za-z0-9][A-Za-z0-9 &'().,/-]*", str(fallback or ""))
    return re.sub(r"\s+", " ", " ".join(ascii_parts)).strip()


def _poi_semantic_text(row: pd.Series, tags: dict[str, Any]) -> str:
    parts = ["poi", f"category={_token(row.get('top_category', 'unknown'))}"]
    primary_key = _token(row.get("primary_key", ""))
    primary_value = _token(row.get("primary_value", ""))
    if primary_key and primary_value:
        parts.append(f"{primary_key}={primary_value}")
    for key in SEMANTIC_KEYS:
        value = _token(tags.get(key, ""))
        token = f"{key}={value}" if value else ""
        if token and token not in parts:
            parts.append(token)
    return " ".join(parts)


def _aoi_semantic_text(row: pd.Series, tags: dict[str, Any]) -> str:
    parts = ["aoi", f"osm_class={int(row['class_id'])}"]
    source_tag = _token(row.get("source_tag", ""))
    if source_tag:
        parts.append(source_tag)
    for key in SEMANTIC_KEYS:
        value = _token(tags.get(key, ""))
        token = f"{key}={value}" if value else ""
        if token and token not in parts:
            parts.append(token)
    return " ".join(parts)


def clean_poi(poi: gpd.GeoDataFrame, image_bounds, image_crs) -> tuple[gpd.GeoDataFrame, dict]:
    original = len(poi)
    poi = poi.to_crs(image_crs).copy()
    poi = poi[poi.geometry.notna() & ~poi.geometry.is_empty].copy()
    poi = poi[poi.geometry.geom_type == "Point"].copy()
    poi = poi[poi.geometry.within(image_bounds) | poi.geometry.touches(image_bounds)].copy()

    tags = poi["all_tags_json"].map(_safe_tags)
    poi["name_en"] = [
        _english_name(tag_dict, fallback)
        for tag_dict, fallback in zip(tags, poi.get("name", ""))
    ]
    poi["semantic_text"] = [
        _poi_semantic_text(row, tag_dict)
        for (_, row), tag_dict in zip(poi.iterrows(), tags)
    ]
    poi["x_round"] = poi.geometry.x.round(2)
    poi["y_round"] = poi.geometry.y.round(2)
    before_dedupe = len(poi)
    poi = poi.drop_duplicates(
        subset=["x_round", "y_round", "semantic_text", "target_class"], keep="first"
    ).drop(columns=["x_round", "y_round"])
    poi = poi.sort_values(["top_category", "subcategory", "osm_type", "osm_id"]).reset_index(drop=True)
    report = {
        "original_rows": original,
        "rows_after_geometry_and_extent_filter": before_dedupe,
        "semantic_duplicates_removed": before_dedupe - len(poi),
        "clean_rows": len(poi),
        "empty_english_name": int(poi["name_en"].eq("").sum()),
        "category_counts": poi["top_category"].value_counts().to_dict(),
    }
    return poi, report


def clean_aoi(
    aoi: gpd.GeoDataFrame,
    image_bounds,
    image_crs,
    min_area_m2: float,
) -> tuple[gpd.GeoDataFrame, dict]:
    original = len(aoi)
    aoi = aoi.to_crs(image_crs).copy()
    aoi["geometry"] = aoi.geometry.make_valid()
    aoi = aoi.explode(index_parts=False).reset_index(drop=True)
    aoi = aoi[aoi.geometry.notna() & ~aoi.geometry.is_empty].copy()
    aoi = aoi[aoi.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    aoi["geometry"] = aoi.geometry.intersection(image_bounds)
    aoi = aoi[aoi.geometry.notna() & ~aoi.geometry.is_empty].copy()
    aoi["area_m2_clean"] = aoi.geometry.area
    before_tiny = len(aoi)
    aoi = aoi[aoi["area_m2_clean"] >= min_area_m2].copy()

    tags = aoi["all_tags_json"].map(_safe_tags)
    aoi["name_en"] = [
        _english_name(tag_dict, fallback)
        for tag_dict, fallback in zip(tags, aoi.get("name", ""))
    ]
    aoi["semantic_text"] = [
        _aoi_semantic_text(row, tag_dict)
        for (_, row), tag_dict in zip(aoi.iterrows(), tags)
    ]
    aoi["geom_hash"] = aoi.geometry.normalize().to_wkb(hex=True)
    before_dedupe = len(aoi)
    aoi = aoi.drop_duplicates(
        subset=["geom_hash", "class_id", "source_tag"], keep="first"
    ).drop(columns="geom_hash")
    aoi = aoi.sort_values(
        ["priority", "area_m2_clean", "osm_type", "osm_id"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    report = {
        "original_rows": original,
        "rows_after_geometry_and_extent_filter": before_tiny,
        "below_min_area_removed": before_tiny - before_dedupe,
        "semantic_duplicates_removed": before_dedupe - len(aoi),
        "clean_rows": len(aoi),
        "class_counts": {str(k): int(v) for k, v in aoi["class_id"].value_counts().sort_index().items()},
    }
    return aoi, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean OSM POI/AOI without modifying source data.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["paths"]["cleaned_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(cfg["paths"]["image"]) as src:
        image_crs = src.crs
        image_bounds = box(*src.bounds)

    poi = gpd.read_file(cfg["paths"]["poi"])
    aoi = gpd.read_file(cfg["paths"]["aoi"])
    poi_clean, poi_report = clean_poi(poi, image_bounds, image_crs)
    aoi_clean, aoi_report = clean_aoi(
        aoi,
        image_bounds,
        image_crs,
        float(cfg["data"]["min_aoi_area_m2"]),
    )

    poi_path = out_dir / "poi_clean.gpkg"
    aoi_path = out_dir / "aoi_clean.gpkg"
    for path in (poi_path, aoi_path):
        if path.exists():
            path.unlink()
    poi_clean.to_file(poi_path, layer="poi", driver="GPKG")
    aoi_clean.to_file(aoi_path, layer="aoi", driver="GPKG")
    if bool(cfg["data"].get("write_clean_csv", True)):
        poi_clean.drop(columns="geometry").to_csv(out_dir / "poi_clean.csv", index=False, encoding="utf-8-sig")
        aoi_clean.drop(columns="geometry").to_csv(out_dir / "aoi_clean.csv", index=False, encoding="utf-8-sig")

    report = {
        "policy": {
            "source_files_modified": False,
            "proper_names_used_as_model_text": False,
            "model_text": "OSM semantic category and key=value tags only",
            "minimum_aoi_area_m2": float(cfg["data"]["min_aoi_area_m2"]),
            "spatial_extent": list(image_bounds.bounds),
            "crs": str(image_crs),
        },
        "poi": poi_report,
        "aoi": aoi_report,
    }
    (out_dir / "cleaning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
