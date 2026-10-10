import hashlib
import json

import pytest

from hu_bot.native import SolverError,select_binary


def test_release_selection_verifies_immutable_binary_and_honors_explicit_override(tmp_path,monkeypatch):
    monkeypatch.delenv("HU_SOLVER_BINARY",raising=False)
    native=tmp_path/"native";native.mkdir();binary=native/"release-test";binary.write_bytes(b"solver fixture")
    manifest=native/"runtime.json"
    manifest.write_text(json.dumps({"binary":"native/release-test","sha256":hashlib.sha256(binary.read_bytes()).hexdigest()}))
    assert select_binary(tmp_path)==binary
    binary.write_bytes(b"modified")
    with pytest.raises(SolverError,match="hash mismatch"):select_binary(tmp_path)
    monkeypatch.setenv("HU_SOLVER_BINARY",str(tmp_path/"experimental"))
    assert select_binary(tmp_path)==tmp_path/"experimental"


def test_release_manifest_cannot_silently_select_missing_or_external_file(tmp_path,monkeypatch):
    monkeypatch.delenv("HU_SOLVER_BINARY",raising=False)
    native=tmp_path/"native";native.mkdir()
    (native/"runtime.json").write_text(json.dumps({"binary":"../outside","sha256":"0"*64}))
    with pytest.raises(SolverError,match="invalid release path"):select_binary(tmp_path)
