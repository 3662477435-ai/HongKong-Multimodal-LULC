from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run(command: list[str], repo: Path) -> None:
    print("RUN", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo / "src")
    subprocess.run(command, cwd=repo, env=environment, check=True)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    root = repo / "outputs" / "full" / "eight_class_20260725"
    results = root / "final_results"
    maps = root / "prediction_maps"
    figures = root / "figures"
    for directory in (results, maps, figures):
        directory.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, "-m", "hk_landuse.summarize_eight_class_results",
        "--experiment-root", str(root),
        "--output-dir", str(results),
        "--manifest", str(repo / "artifacts" / "full" / "prepared" / "manifest_hier_v2.csv"),
        "--bootstrap", "5000",
    ], repo)
    run([
        sys.executable, "-m", "hk_landuse.predict_experiment_models_8class",
        "--config", "config_release.yaml",
        "--experiment-root", str(root),
        "--output-dir", str(maps),
        "--seed", "42",
        "--context-margin", "16",
        "--batch-size", "2",
    ], repo)
    run([
        sys.executable, "-m", "hk_landuse.export_eight_class_figures",
        "--results-dir", str(results),
        "--map-dir", str(maps),
        "--output-dir", str(figures),
    ], repo)
    run([
        sys.executable, "-m", "hk_landuse.write_eight_class_report",
        "--results-dir", str(results),
        "--output", str(root / "RESULTS_REPORT_8CLASS.md"),
    ], repo)


if __name__ == "__main__":
    main()
