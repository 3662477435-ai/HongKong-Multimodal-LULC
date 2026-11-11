"""Offline checks for the separated data manifest loader."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import download_zenodo_data as data


class ZenodoManifestTests(unittest.TestCase):
    def fixture(self):
        records = []
        for archive in data.ARCHIVES:
            record = copy.deepcopy(archive)
            record['name'] = record.pop('file')
            record['parts'] = []
            records.append(record)
        return {'archives': records}

    def load(self, content):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            path.write_text(json.dumps(content), encoding='utf-8')
            with patch.object(data, 'check') as checked:
                records = data.load_manifest(path)
                checked.assert_called_once_with(path, data.MANIFEST)
                return records

    def test_preserves_expected_archive_names(self):
        records = self.load(self.fixture())
        self.assertEqual([r['file'] for r in records], [r['file'] for r in data.ARCHIVES])

    def test_wrong_archive_hash_rejected(self):
        content = self.fixture()
        content['archives'][0]['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            self.load(content)

    def test_missing_archive_rejected(self):
        content = self.fixture()
        content['archives'].pop()
        with self.assertRaises(ValueError):
            self.load(content)


if __name__ == '__main__':
    unittest.main()
