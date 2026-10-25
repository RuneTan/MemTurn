"""Verify all delivered file bytes against MANIFEST.json; no network access."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
    failures = []
    for row in manifest['files']:
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(row['path'])
            continue
        data = path.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            failures.append(row['path'])
    return {'verified':not failures,'checked_files':len(manifest['files']),'failures':failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', default='.')
    args = parser.parse_args()
    result = verify(args.root)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['verified']: raise SystemExit(1)


if __name__ == '__main__': main()
