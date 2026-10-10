import pytest

from hu_bot.checkpoints import select_checkpoint,verify_checkpoint,digest


def test_stack_selection_uses_nearest_log_depth_and_latest_training():
    entries=[{"stack_bb":stack,"batch":batch,"training_deals":(batch+1)*1_000_000}
             for stack in (50,100,200,1000) for batch in (9,99)]
    for stack,expected in [(50,50),(70,50),(71,100),(141.4,100),(141.5,200),(447,200),(448,1000),(999.99,1000)]:
        entry=select_checkpoint(stack,entries)
        assert entry["stack_bb"]==expected and entry["batch"]==99
    for invalid in (49.99,1000.01,float("nan"),float("inf")):
        with pytest.raises(ValueError):select_checkpoint(invalid,entries)


def test_changed_or_missing_checkpoint_rejected(tmp_path,monkeypatch):
    import hu_bot.checkpoints as module
    monkeypatch.setattr(module,"ROOT",tmp_path)
    path=tmp_path/"weights"
    path.write_bytes(b"complete-checkpoint")
    entry={"sha256":{"weights":digest(path)}}
    verify_checkpoint(entry)
    path.write_bytes(b"partial-checkpoint")
    with pytest.raises(ValueError,match="changed or missing"):verify_checkpoint(entry)
    path.unlink()
    with pytest.raises(ValueError,match="changed or missing"):verify_checkpoint(entry)


def test_preferred_family_only_overrides_selection_at_its_own_depth():
    old={"stack_bb":200,"batch":299,"training_deals":300000000}
    refined={"stack_bb":200,"batch":99,"training_deals":100000000,"preferred":True}
    near={"stack_bb":100,"batch":99,"training_deals":100000000}
    assert select_checkpoint(200,[old,refined,near]) is refined
    assert select_checkpoint(100,[old,refined,near]) is near


def test_same_depth_and_batch_with_different_cards_reloads_policy(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import hu_bot.checkpoints as module
    import hu_bot.full_blueprint as full
    loaded=[]
    class FakePolicy:
        last_decision={}
        def __init__(self,**kwargs):loaded.append(kwargs)
        def act(self,view):return "test action"
        def close(self):pass
    monkeypatch.setattr(module,"ROOT",tmp_path)
    monkeypatch.setattr(full,"FullBlueprintBot",FakePolicy)
    catalog=tmp_path/"catalog.json"
    entry={"stack_bb":200,"batch":99,"training_deals":100000000,"sha256":{},
           "card_config":"first_cards","betting_config":"bets"}
    catalog.write_text(json.dumps({"format":1,"checkpoints":[entry]}))
    bot=module.StackAdaptiveBot(catalog=catalog,native_iterations=2000,native_compression=False)
    view=SimpleNamespace(stacks=[19950,19900],committed=[50,100],bb=100)
    bot.act(view);bot.act(view)
    assert len(loaded)==1
    bot.entries[0]={**entry,"card_config":"second_cards"}
    bot.act(view)
    assert len(loaded)==2 and loaded[1]['card_config']==tmp_path/'second_cards'
    assert loaded[1]['betting_config']==tmp_path/'bets'
    assert all(row['native_iterations']==2000 for row in loaded)
    assert all(row['native_compression'] is False for row in loaded)
    assert bot.selected==(200,99)
    bot.close()
