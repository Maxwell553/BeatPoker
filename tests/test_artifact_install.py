import hashlib
import io
from pathlib import Path
import tarfile

import pytest

from scripts.install_artifacts import destination, extract, verify


def bundle(tmp_path, members):
    path = tmp_path / 'bundle.tar.gz'
    with tarfile.open(path, 'w:gz') as tar:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.type = kind
            if kind == tarfile.SYMTYPE:
                info.linkname = '/tmp/unrelated'
            tar.addfile(info, io.BytesIO(data))
    return path


def record(name='training/slumbot_data/model', data=b'weights'):
    return {'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def test_install_then_detect_model_corruption(tmp_path):
    root = tmp_path / 'install'
    root.mkdir()
    entry = record()
    manifest = {'files':[entry]}
    archive = bundle(tmp_path, [(entry['path'], b'weights', tarfile.REGTYPE)])
    extract(archive, root, manifest)
    assert verify(root, manifest) == []
    (root / entry['path']).write_bytes(b'changed')
    assert verify(root, manifest) == [entry['path']]


def test_corrupt_payload_does_not_replace_existing_model(tmp_path):
    entry = record()
    path = tmp_path / entry['path']
    path.parent.mkdir(parents=True)
    path.write_bytes(b'original')
    archive = bundle(tmp_path, [(entry['path'], b'corrupt', tarfile.REGTYPE)])
    with pytest.raises(ValueError, match='checksum'):
        extract(archive, tmp_path, {'files':[entry]})
    assert path.read_bytes() == b'original'


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a/../../escape', 'a\\escape'])
def test_reject_path_traversal(tmp_path, name):
    with pytest.raises(ValueError):
        destination(tmp_path, name)


def test_reject_symlink_parent(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'external').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        destination(root, 'external/escape')


@pytest.mark.parametrize('members', [[], [('unlisted', b'weights', tarfile.REGTYPE)],
    [('training/slumbot_data/model', b'', tarfile.SYMTYPE)],
    [('training/slumbot_data/model', b'weights', tarfile.REGTYPE)] * 2])
def test_reject_incomplete_extra_link_or_duplicate_members(tmp_path, members):
    archive = bundle(tmp_path, members)
    with pytest.raises(ValueError):
        extract(archive, tmp_path / 'root', {'files':[record()]})
