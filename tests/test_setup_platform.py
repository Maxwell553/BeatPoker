import pytest
from scripts.setup import needs_build


@pytest.mark.parametrize('system,machine,version,expected', [
    ('Darwin','arm64','26.6.2',False),
    ('Darwin','arm64','27.0',False),
    ('Darwin','arm64','15.7',True),
    ('Darwin','arm64','',True),
    ('Darwin','x86_64','26.0',True),
    ('Linux','x86_64','',True),
    ('Linux','aarch64','',True),
])
def test_prebuilt_policy_reader_requires_supported_macos_target(system,machine,version,expected):
    assert bool(needs_build(system,machine,version)) is expected
