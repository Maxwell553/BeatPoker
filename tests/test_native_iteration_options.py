import pytest

from hu_bot.bot import SolverConfig
from hu_bot.full_blueprint import FullBlueprintBot


@pytest.mark.parametrize('value',[True,False,None,0,-1,1.5,1000.,2**32,'2000'])
def test_invalid_native_iteration_caps_fail_before_policy_server_is_started(value,monkeypatch):
    import hu_bot.full_blueprint as module
    monkeypatch.setattr(module,'PolicyServer',lambda *a:pytest.fail('Validation must precede loading'))
    with pytest.raises(ValueError,match='positive 32-bit integer'):SolverConfig(iterations=value)
    with pytest.raises(ValueError,match='positive 32-bit integer'):FullBlueprintBot(native_iterations=value)


def test_iteration_one_remains_available_for_explicit_diagnostics():
    assert SolverConfig(iterations=1).iterations==1


@pytest.mark.parametrize('value',[None,0,1,'false',[]])
def test_invalid_native_compression_fails_before_loading(value,monkeypatch):
    import hu_bot.full_blueprint as module
    monkeypatch.setattr(module,'PolicyServer',lambda *a:pytest.fail('Validation must precede loading'))
    with pytest.raises(ValueError,match='compression must be a boolean'):SolverConfig(compression=value)
    with pytest.raises(ValueError,match='compression must be a boolean'):FullBlueprintBot(native_compression=value)
