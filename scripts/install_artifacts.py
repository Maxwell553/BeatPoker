"""Download, verify, and install the immutable model bundle (standard library only)."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def destination(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError(f'Unsafe artifact path: {name}')
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Artifact escapes installation directory: {name}')
    return path


def valid(path, entry):
    return path.is_file() and path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256']


def verify(root, manifest):
    return [e['path'] for e in manifest['files'] if not valid(destination(root, e['path']), e)]


def extract(archive_path, root, manifest):
    expected = {e['path']: e for e in manifest['files']}
    seen = set()
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            entry = expected.get(member.name)
            if entry is None or member.name in seen or not member.isfile() or member.size != entry['bytes']:
                raise ValueError(f'Unexpected archive member: {member.name}')
            seen.add(member.name)
            path = destination(root, member.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
                temporary = Path(out.name)
                try:
                    with archive.extractfile(member) as source:
                        shutil.copyfileobj(source, out, 1024 * 1024)
                    out.flush()
                    if not valid(temporary, entry):
                        raise ValueError(f'Artifact checksum mismatch: {member.name}')
                    temporary.chmod(0o644)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
    if seen != set(expected):
        raise ValueError('Archive is incomplete')


def install(root=ROOT, source_dir=None, verify_only=False):
    manifest = json.loads((root / 'artifacts/manifest.json').read_text())
    if manifest['format'] != 1:
        raise ValueError('Unsupported artifact manifest')
    missing = verify(root, manifest)
    if not missing:
        print(f"All {len(manifest['files'])} model files verified.")
        return
    if verify_only:
        raise ValueError(f'{len(missing)} missing or changed model files; run scripts/install_artifacts.py')
    cache = root / '.cache/artifacts'
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for entry in manifest['parts']:
        name = entry['name']
        if PurePosixPath(name).name != name or '\\' in name:
            raise ValueError('Invalid artifact part name')
        path = (source_dir or cache) / name
        if not valid(path, entry):
            if source_dir:
                raise ValueError(f'Missing or corrupt local bundle part: {name}')
            url = f"https://github.com/{manifest['repository']}/releases/download/{manifest['release']}/{name}"
            print(f'Downloading {name} ({entry["bytes"] / 1024**2:.0f} MiB)…', flush=True)
            partial = path.with_suffix(path.suffix + '.partial')
            try:
                with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as out:
                    shutil.copyfileobj(response, out, 1024 * 1024)
                if not valid(partial, entry):
                    raise ValueError(f'Download checksum mismatch: {name}')
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
        parts.append(path)
    with tempfile.TemporaryDirectory(dir=cache) as directory:
        combined = Path(directory) / 'models.tar.gz'
        with combined.open('wb') as out:
            for path in parts:
                with path.open('rb') as source:
                    shutil.copyfileobj(source, out, 1024 * 1024)
        print('Installing and verifying model files…', flush=True)
        extract(combined, root, manifest)
    print(f"Installed {len(manifest['files'])} verified model files.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-dir', type=Path, help='Use downloaded release parts without network access')
    parser.add_argument('--verify', action='store_true', help='Verify installed models without downloading')
    args = parser.parse_args()
    try:
        install(source_dir=args.from_dir, verify_only=args.verify)
    except (OSError, ValueError, tarfile.TarError) as exc:
        parser.exit(1, f'Artifact installation failed: {exc}\n')


if __name__ == '__main__':
    main()
