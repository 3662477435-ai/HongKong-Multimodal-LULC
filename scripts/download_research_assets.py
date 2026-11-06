"""Download, hash-check, join and safely extract the v1.0.0 research assets.

Standard-library only. No credentials, no deletion of downloaded assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

BASE_URL = "https://github.com/3662477435-ai/HongKong-Multimodal-LULC/releases/download/v1.0.0"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path: Path, expected: dict) -> None:
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        raise ValueError(f"Missing or wrong-size file: {path}")
    if digest(path) != expected["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {path}")


def download(name: str, directory: Path, base: str) -> Path:
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("Invalid asset name")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    if target.exists():
        return target
    partial = directory / (name + ".downloading")
    # Retain partial transfers for diagnosis; never overwrite existing data.
    if partial.exists():
        raise FileExistsError(f"Incomplete transfer exists: {partial}. Move it aside before retrying.")
    request = urllib.request.Request(base.rstrip("/") + "/" + name, headers={"User-Agent": "HK-LULC-research-assets/1.0"})
    print("Downloading " + name, flush=True)
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("xb") as stream:
        shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
    partial.rename(target)
    return target


def join_archive(record: dict, directory: Path) -> Path:
    target = directory / record["file"]
    if target.exists():
        check(target, record)
        return target
    parts = record["parts"]
    for part in parts:
        check(directory / part["file"], part)
    temporary = directory / (record["file"] + ".joining")
    if temporary.exists():
        raise FileExistsError(f"Incomplete join exists: {temporary}. Move it aside before retrying.")
    with temporary.open("xb") as stream:
        for part in parts:
            with (directory / part["file"]).open("rb") as source:
                shutil.copyfileobj(source, stream, length=8 * 1024 * 1024)
    check(temporary, record)
    temporary.rename(target)
    return target


def safe_extract(archive: Path, output: Path) -> None:
    output = output.resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            relative = PurePosixPath(member.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts or any(":" in p for p in relative.parts):
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Archive symlink is not allowed: {member.filename}")
            target = output.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(output):
                raise ValueError(f"Archive path escapes target: {member.filename}")
            if member.is_dir():
                continue
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite an existing extracted file: {target}")
        z.extractall(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=Path("release_assets"))
    parser.add_argument("--output", type=Path, default=Path("research_package"))
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--offline", action="store_true", help="Use files already downloaded into --assets-dir")
    parser.add_argument("--verify-only", action="store_true", help="Verify upload parts without joining or extracting")
    args = parser.parse_args()
    manifest_path = args.assets_dir / "release_manifest.json"
    if not args.offline:
        download(manifest_path.name, args.assets_dir, args.base_url)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["archives"]:
        for part in record["parts"]:
            path = args.assets_dir / part["file"]
            if not args.offline:
                download(part["file"], args.assets_dir, args.base_url)
            check(path, part)
            print("Verified " + part["file"], flush=True)
    if args.verify_only:
        print("All release assets match their recorded sizes and SHA-256 checksums.")
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose an empty output directory; existing research files will not be overwritten.")
    args.output.mkdir(parents=True, exist_ok=True)
    for record in manifest["archives"]:
        archive = join_archive(record, args.assets_dir)
        print("Extracting " + archive.name, flush=True)
        safe_extract(archive, args.output)
    print(f"Prepared {args.output.resolve()}. Next: python run_release.py verify, then metrics.")
    print("Full frozen-model inference also needs the two withheld third-party rasters; see DATA_ACCESS.md.")


if __name__ == "__main__":
    main()
