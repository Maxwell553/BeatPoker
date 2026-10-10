"""Explicitly published immutable checkpoints, selected once per bot session."""
import hashlib
import json
import math
from pathlib import Path
from .bot import SolverConfig

ROOT=Path(__file__).resolve().parents[1]
CATALOG=ROOT/"training/checkpoints/catalog.json"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def load_catalog(path=CATALOG):
    data=json.loads(Path(path).read_text())
    if data.get("format")!=1 or not data.get("checkpoints"):
        raise ValueError("No published full-game checkpoints")
    return data["checkpoints"]


def select_checkpoint(stack,entries):
    if not math.isfinite(stack) or not 50 <= stack <= 1000:
        raise ValueError("Starting stack must be 50–1000 bb")
    # Nearest multiplicative stack distance, then most training at that depth.
    # This translates strategies; it is not an equilibrium interpolation.
    return min(entries,key=lambda e:(abs(math.log(stack/e["stack_bb"])),
                                     -bool(e.get("preferred",False)),
                                     -e["training_deals"],-e["batch"]))


def verify_checkpoint(entry):
    for relative,expected in entry["sha256"].items():
        path=ROOT/relative
        if not path.is_file() or digest(path)!=expected:
            raise ValueError(f"Published checkpoint changed or missing: {relative}")


class StackAdaptiveBot:
    """Experimental nearest-depth full policy with exact-card search fallback."""
    name="StackAdaptiveBot"

    def __init__(self,seed=8128,seconds=30,catalog=CATALOG,rake=None,resolve_from_street=None,river_gadget=False,native_iterations=1000,native_compression=True):
        self.native_iterations=SolverConfig(iterations=native_iterations,compression=native_compression).iterations
        self.native_compression=native_compression
        if river_gadget and resolve_from_street is not None:raise ValueError("Choose one refinement method")
        self.entries=load_catalog(catalog)
        self.seed,self.seconds,self.rake=seed,seconds,rake
        self.resolve_from_street=resolve_from_street
        self.river_gadget=river_gadget
        self.bot=None
        self.selected=None
        self._selected_identity=None
        self.last_decision={}

    def reset(self):
        self.last_decision={}
        if self.bot:self.bot.reset()

    def close(self):
        if self.bot:self.bot.close()
        self.bot=None
        self.selected=None
        self._selected_identity=None

    def observe_action(self,*args,**kwargs):pass
    def end_hand(self,*args):pass

    def act(self,view):
        if self.rake is not None and self.rake.percent:
            raise ValueError("Published full-game checkpoints require unraked play")
        stack=min(s+c for s,c in zip(view.stacks,view.committed))/view.bb
        entry=select_checkpoint(stack,self.entries)
        key=(entry["stack_bb"],entry["batch"])
        identity=(*key,entry.get("card_config"),entry.get("betting_config"))
        if self._selected_identity!=identity:
            verify_checkpoint(entry)
            self.close()
            from .full_blueprint import FullBlueprintBot
            self.bot=FullBlueprintBot(stack=entry["stack_bb"],batch=entry["batch"],
                seed=self.seed,fallback_seconds=self.seconds,allow_stack_scaling=True,native_iterations=self.native_iterations,native_compression=self.native_compression,
                resolve_from_street=self.resolve_from_street,river_gadget=self.river_gadget,
                card_config=ROOT/entry["card_config"] if entry.get("card_config") else None,
                betting_config=ROOT/entry["betting_config"] if entry.get("betting_config") else None)
            self.selected=key
            self._selected_identity=identity
        action=self.bot.act(view)
        self.last_decision={**self.bot.last_decision,"checkpoint_selection":{
            "actual_stack_bb":stack,"training_stack_bb":entry["stack_bb"],
            "batch":entry["batch"],"training_deals":entry["training_deals"],
            "card_abstraction":entry.get("card_abstraction","potential512_hs256"),
            "betting_abstraction":entry.get("betting_abstraction",f"hu{entry['stack_bb']}"),
            "preferred":bool(entry.get("preferred",False)),
            "method":"nearest_log_stack","validated_pro_level":False}}
        return action
