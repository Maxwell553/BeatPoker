"""Explicit betting-tree identities keep incompatible strategy tables separate."""
from dataclasses import dataclass
from pathlib import Path
import re

from .card_config import ROOT


@dataclass(frozen=True)
class BettingConfig:
    path: Path
    name: str
    stack_bb: float

    @property
    def tree_path(self):
        return ROOT/f"training/slumbot_data/betting_tree.holdem.2.{self.name}"


def load_betting_config(stack=None,path=None):
    if path is None and stack is None:raise ValueError("Betting configuration or stack required")
    path=Path(path or ROOT/f"training/configs/hu{stack}_params").resolve()
    fields={}
    for line in path.read_text().splitlines():
        line=line.split("#",1)[0].strip()
        if not line:continue
        key,value=line.split(None,1)
        if key in fields:raise ValueError(f"Duplicate betting parameter: {key}")
        fields[key]=value.strip()
    name=fields.get("BettingAbstractionName","")
    if not re.fullmatch(r"[A-Za-z0-9_]+",name):raise ValueError("Invalid betting abstraction name")
    if fields.get("Limit")!="false" or fields.get("Asymmetric")!="false" or fields.get("MinBet")!="2":
        raise ValueError("Full policies require symmetric no-limit trees with a two-unit big blind")
    units=int(fields["StackSize"]);depth=units/2
    if not 50<=depth<=1000 or (stack is not None and depth!=stack):
        raise ValueError("Betting configuration stack mismatch or outside 50–1000 bb")
    return BettingConfig(path,name,depth)
