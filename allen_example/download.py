"""Resumable, checksum-verified downloads from the pinned Allen public release."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent


def digest(path, algorithm='md5'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def fetch(spec, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    marker = dest.with_name(dest.name + '.verified.json')
    if dest.exists():
        stat = dest.stat()
        if marker.exists():
            old = json.loads(marker.read_text())
            if old == dict(spec=spec, size=stat.st_size, mtime_ns=stat.st_mtime_ns):
                return
        if stat.st_size != spec['size'] or digest(dest) != spec['md5']:
            raise ValueError(f'Existing file fails the pinned checksum: {dest}; it was not overwritten')
    else:
        part = dest.with_name(dest.name + '.partial')
        missing = spec['size'] - (part.stat().st_size if part.exists() else 0)
        if missing < 0:
            raise ValueError(f'Partial download is larger than the pinned file: {part}')
        if shutil.disk_usage(dest.parent).free < missing + 5 * 1024**3:
            raise ValueError(f'Insufficient space for {dest.name} plus a 5-GiB reserve')
        if missing:
            subprocess.run(['curl', '--fail', '--location', '--retry', '8', '--retry-delay', '5',
                            '--connect-timeout', '60', '--continue-at', '-', '--output', str(part), spec['url']], check=True)
        if part.stat().st_size != spec['size'] or digest(part) != spec['md5']:
            raise ValueError(f'Download failed the published size/MD5 check: {part}')
        part.replace(dest)
    stat = dest.stat()
    tmp = marker.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(spec=spec, size=stat.st_size, mtime_ns=stat.st_mtime_ns), indent=2))
    tmp.replace(marker)


def main(root):
    manifest = json.loads((HERE / 'DOWNLOAD_MANIFEST.json').read_text())
    for spec in manifest['files']:
        print(f'Download/checksum: {spec["name"]} ({spec["size"] / 1e9:.3f} GB)', flush=True)
        fetch(spec, Path(root) / 'data/raw' / spec['name'])
    print('ALLEN 1 PUBLIC DATA DOWNLOAD: PASSED', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root.resolve())
