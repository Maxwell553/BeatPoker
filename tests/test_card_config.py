from pathlib import Path

import pytest

from hu_bot.card_config import ROOT, load_card_config
from training.export_checkpoint import export


def test_abstractions_have_separate_checkpoint_and_fallback_paths():
    old=load_card_config()
    new=load_card_config(ROOT/"training/configs/cards_ochs256_params")
    assert old.name=="potential512_hs256"
    assert new.bucketings==("null","potential512","hs256","ochs256v1")
    assert old.checkpoint(200)!=new.checkpoint(200)
    assert old.preflop_export(200,99)!=new.preflop_export(200,99)
    assert old.preflop_export(200,99).name=="hu200_batch99.json"


@pytest.mark.parametrize("text",[
    "CardAbstractionName ../bad\nBucketings null,a,b,c\n",
    "CardAbstractionName valid\nBucketings null,a,b\n",
    "CardAbstractionName valid\nBucketings grouped,a,b,c\n",
    "CardAbstractionName valid\nCardAbstractionName second\nBucketings null,a,b,c\n",
])
def test_invalid_config_is_rejected(tmp_path,text):
    path=tmp_path/"cards";path.write_text(text)
    with pytest.raises(ValueError):load_card_config(path)


def test_export_rejects_mismatched_abstraction_before_running_native(tmp_path):
    new=load_card_config(ROOT/"training/configs/cards_ochs256_params")
    with pytest.raises(ValueError,match="disagree"):
        export(ROOT/"training/configs/hu200_params",new.checkpoint(200),9,tmp_path/"export.json")
    assert not (tmp_path/"export.json").exists()
