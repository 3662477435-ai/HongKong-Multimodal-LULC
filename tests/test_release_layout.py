"""Contracts for the cleaned code/data layout; no private imagery is needed."""
from __future__ import annotations

import ast
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_runner", ROOT / "run_release.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ReleaseLayoutTests(unittest.TestCase):
    def test_main_model_registry_is_complete(self):
        main = runner.models()
        registry = json.loads((ROOT / "metadata/model_registry.json").read_text())
        self.assertEqual(len(main), 9)
        self.assertEqual(len(registry), 83)
        lookup = {record["id"]: record for record in registry}
        for record in main:
            self.assertEqual(record["checkpoint"], lookup[record["id"]]["checkpoint"])
            self.assertEqual(record["checkpoint_sha256"], lookup[record["id"]]["checkpoint_sha256"])

    def test_withheld_inputs_remain_required_and_documented(self):
        with (ROOT / "metadata/data_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            records = {row["key"]: row for row in csv.DictReader(stream)}
        for key in ("highres_aligned", "building_aligned"):
            self.assertEqual(records[key]["access_route"], "local_only_permission_pending")
            self.assertEqual(len(records[key]["sha256"]), 64)
            self.assertIn(Path(records[key]["release_path"]).name,
                          (ROOT / "docs/DATA_ACCESS.md").read_text())

    def test_data_directory_resolves_inputs_not_code_metadata(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "DATA_ROOT", Path(tmp)):
            config = runner.load_config(ROOT / "config_release.yaml")
            self.assertEqual(Path(config["paths"]["image"]), Path(tmp) / "data/inputs/sentinel1_sentinel2_10m.tif")
            self.assertEqual(runner.data_root(), Path(tmp))
            self.assertEqual(len(runner.models()), 9)
            self.assertIn("highres", config["model"]["modalities"])

    def test_metrics_only_need_archived_confusions(self):
        import numpy as np
        matrix = np.eye(8, dtype=int) * 5
        saved = {"confusion_matrix": matrix.tolist(), "overall_accuracy": 1.0,
                 "macro_iou_present": 1.0, "macro_f1_present": 1.0}
        record = {"name": "fixture", "metrics": "results/fixture.json"}
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "DATA_ROOT", Path(tmp)), patch.object(runner, "models", return_value=[record]):
            (Path(tmp) / "results").mkdir()
            (Path(tmp) / "results/fixture.json").write_text(json.dumps({"test": {"official8": saved}}))
            self.assertEqual(runner.metrics(), 0)
            output = json.loads((Path(tmp) / "reproduced_metrics.json").read_text())
            self.assertEqual(output[0]["maximum_difference"], 0.0)
            self.assertEqual(output[0]["valid_pixels"], 40)

    def test_no_internal_import_points_to_a_removed_module(self):
        package = ROOT / "src/hk_landuse"
        files = list(package.glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.level and node.module:
                        module = node.module.split(".")[0]
                    elif node.module and node.module.startswith("hk_landuse."):
                        module = node.module.split(".")[1]
                    else:
                        continue
                    self.assertTrue((package / (module + ".py")).is_file(),
                                    f"{path.name}: missing internal module {module}")

    def test_runner_references_existing_modules_and_scripts(self):
        for path in (ROOT / "scripts").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                value = node.value
                if value.startswith("hk_landuse.") and " " not in value:
                    self.assertTrue((ROOT / "src" / (value.replace(".", "/") + ".py")).is_file(), f"{path.name}: {value}")
                if value.startswith("scripts/") and value.endswith(".py"):
                    self.assertTrue((ROOT / value).is_file(), f"{path.name}: {value}")


if __name__ == "__main__":
    unittest.main()
