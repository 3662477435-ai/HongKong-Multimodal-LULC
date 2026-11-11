"""Download and verify the separately archived data (DOI publication pending).

Standard library only. This does not fetch analysis code; clone GitHub separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from download_research_assets import check, download, join_archive, safe_extract

BASE_URL = 'https://zenodo.org/records/24137833/files'
ARCHIVES = [
    {'file': '01_results_tables_metadata.zip', 'bytes': 3756134,
     'sha256': '674281247ba367cafa2d2a740f826ef86f2869b932d0b1581f23510119ba7de5'},
    {'file': '02_selected_model_checkpoints.zip', 'bytes': 3197087401,
     'sha256': '20180a0b5536dc1b79b8d3dbb6df65f581cf842465d9beee5d464f6c23522d04'},
    {'file': '03_open_provider_inputs_and_splits.zip', 'bytes': 2608711155,
     'sha256': 'e611cf0f91f76ddcfee11f720ff7791d4181f2aca31e954e0fba373ffacf0657'},
    {'file': '04_prediction_products_and_final_figures.zip', 'bytes': 270008767,
     'sha256': 'd0456890881bd49c397f8ba5bd9fea25c8fa974499e583b807ced3252c1bf6e6'},
]
MANIFEST = {'file': 'data_asset_manifest.json', 'bytes': 11239,
            'sha256': '9c5cc42a7c4453d1e40e72a2807172095fe26ee7560e6f840f6ce5f6d9b21031'}


def load_manifest(path):
    check(path, MANIFEST)
    content = json.loads(path.read_text(encoding='utf-8'))
    records = content['archives']
    if len(records) != len(ARCHIVES):
        raise ValueError('Unexpected archive count')
    for record, expected in zip(records, ARCHIVES):
        if any(record.get(key if key != 'file' else 'name') != value
               for key, value in expected.items()):
            raise ValueError('Manifest archive identity mismatch')
        record['file'] = record['name']
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path, default=Path('zenodo_assets'))
    parser.add_argument('--output', type=Path, default=Path('zenodo_data'))
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.offline:
        print('The DOI must be published before online downloads work. A draft is not public.', flush=True)
    if not args.offline:
        download(MANIFEST['file'], args.assets_dir, BASE_URL)
    records = load_manifest(args.assets_dir / MANIFEST['file'])
    for record in records:
        for part in record['parts']:
            path = args.assets_dir / part['file']
            if not args.offline:
                download(part['file'], args.assets_dir, BASE_URL)
            check(path, part)
            print('Verified ' + part['file'], flush=True)
    if args.verify_only:
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose an empty output directory. No existing files will be overwritten.')
    args.output.mkdir(parents=True, exist_ok=True)
    for record in records:
        archive = join_archive(record, args.assets_dir)
        safe_extract(archive, args.output)
    print('Data extracted to ' + str(args.output.resolve()))
    print('Analysis code stays on GitHub. See README_DATA.md for how to combine them.')


if __name__ == '__main__':
    main()
