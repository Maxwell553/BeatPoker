import pytest

from hu_bot.betting_config import load_betting_config
from hu_bot.card_config import ROOT, load_card_config
from training.export_checkpoint import export


def test_betting_families_keep_checkpoints_and_fallback_exports_separate():
    cards=load_card_config()
    old=load_betting_config(200)
    wider=load_betting_config(200,ROOT/"training/configs/hu200_pf6_params")
    assert old.stack_bb==wider.stack_bb==200
    assert old.name=="hu200" and wider.name=="hu200_pf6"
    assert cards.checkpoint(200)==cards.checkpoint(200,old.name)
    assert cards.checkpoint(200)!=cards.checkpoint(200,wider.name)
    assert cards.preflop_export(200,99)!=cards.preflop_export(200,99,wider.name)
    with pytest.raises(ValueError,match="stack mismatch"):
        load_betting_config(100,wider.path)


def test_export_rejects_cross_family_weights_before_invoking_native_reader(tmp_path):
    cards=load_card_config();wider=load_betting_config(200,ROOT/"training/configs/hu200_pf6_params")
    with pytest.raises(ValueError,match="abstractions disagree"):
        export(wider.path,cards.checkpoint(200),99,tmp_path/"invalid.json")
    assert not (tmp_path/"invalid.json").exists()


@pytest.mark.parametrize("change",[
    ("BettingAbstractionName hu200","BettingAbstractionName ../escape"),
    ("MinBet 2","MinBet 1"),("Asymmetric false","Asymmetric true"),
    ("StackSize 400","StackSize 9999"),("Limit false","Limit true"),
])
def test_incompatible_tree_parameters_are_rejected(tmp_path,change):
    path=tmp_path/"config";path.write_text(load_betting_config(200).path.read_text().replace(*change))
    with pytest.raises(ValueError):load_betting_config(200,path)
