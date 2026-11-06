"""Small offline tests for release integrity and safe extraction."""
import hashlib
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("assets", Path(__file__).resolve().parents[1] / "scripts/download_research_assets.py")
assets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assets)


class AssetTests(unittest.TestCase):
    def test_join_and_extract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.zip"
            with zipfile.ZipFile(original, "w") as z:
                z.writestr("nested/readme.txt", "test payload")
            data = original.read_bytes()
            parts = []
            for i, chunk in enumerate((data[:50], data[50:]), 1):
                name = f"rejoined.zip.part{i:03d}"
                (root / name).write_bytes(chunk)
                parts.append({"file": name, "bytes": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
            record = {"file": "rejoined.zip", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "parts": parts}
            joined = assets.join_archive(record, root)
            self.assertEqual(joined.read_bytes(), data)
            assets.safe_extract(joined, root / "output")
            self.assertEqual((root / "output/nested/readme.txt").read_text(), "test payload")
            with self.assertRaises(FileExistsError):
                assets.safe_extract(joined, root / "output")

    def test_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "part"
            path.write_bytes(b"abc")
            with self.assertRaises(ValueError):
                assets.check(path, {"bytes": 3, "sha256": "0" * 64})

    def test_zip_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("../escape.txt", "not allowed")
            with self.assertRaises(ValueError):
                assets.safe_extract(path, Path(tmp) / "output")
            self.assertFalse((Path(tmp) / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
