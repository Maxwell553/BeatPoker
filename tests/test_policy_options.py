import argparse

import pytest

from hu_bot.options import add_policy_arguments,resolve_policy_arguments


def parse(arguments):
    parser=argparse.ArgumentParser();parser.add_argument('--rake',default='none')
    add_policy_arguments(parser)
    return resolve_policy_arguments(parser.parse_args(arguments),parser)


def test_unraked_launcher_uses_catalog_and_explicit_legacy_options_remain_available():
    assert parse([]).full_blueprint
    assert parse([]).native_iterations==1000
    assert parse([]).native_compression is True
    assert parse(['--native-uncompressed','--resolve-from','turn']).native_compression is False
    assert parse(['--native-iterations','2000','--resolve-from','turn']).native_iterations==2000
    assert parse(['--full-blueprint']).full_blueprint
    assert parse(['--resolve-from','turn']).full_blueprint
    assert not parse(['--native']).full_blueprint
    assert not parse(['--blueprint','preflop.json']).full_blueprint
    assert not parse(['--rake','casino']).full_blueprint


@pytest.mark.parametrize('arguments',[
    ['--native','--full-blueprint'],['--full-blueprint','--rake','casino'],
    ['--full-blueprint','--blueprint','preflop.json'],['--native','--resolve-from','turn'],
    ['--native','--river-gadget'],['--river-gadget','--resolve-from','river'],
    ['--native-iterations','0'],['--native-iterations','4294967296'],['--native-iterations','1.5']])
def test_conflicting_policy_modes_are_rejected(arguments):
    with pytest.raises(SystemExit):parse(arguments)
