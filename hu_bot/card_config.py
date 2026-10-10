"""Names and paths that keep incompatible card abstractions isolated."""
from dataclasses import dataclass
from pathlib import Path
import re
import hashlib
import struct

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_NAME="potential512_hs256"
DEFAULT_CONFIG=ROOT/"training/configs/cards_params"


@dataclass(frozen=True)
class CardConfig:
    path: Path
    name: str
    bucketings: tuple[str, ...]

    def checkpoint(self,stack,betting_name=None):
        name=betting_name or f"hu{stack}"
        return ROOT/f"training/slumbot_data/holdem.2.{self.name}.13.4.3.{name}.ecfr"

    def preflop_export(self,stack,batch,betting_name=None):
        suffix="" if self.name==DEFAULT_NAME else f"_{self.name}"
        name=betting_name or f"hu{stack}"
        return ROOT/f"training/checkpoints/{name}_batch{batch}{suffix}.json"

    def artifacts(self):
        result={"selected_abstraction":self.name,"maps":{}}
        for street,bucketing in enumerate(self.bucketings):
            base=f"holdem.13.4.3.{bucketing}.{street}"
            counts=ROOT/f"training/slumbot_data/num_buckets.{base}"
            mapping=ROOT/f"training/slumbot_data/buckets.{base}"
            count=struct.unpack("<I",counts.read_bytes())[0]
            if not 1<=count<=65536 or (street==0 and count!=169):
                raise ValueError("Incompatible card abstraction dimensions")
            expected=(1326,1755*1176,63193*1128,2554656*1081)[street]*2
            if mapping.stat().st_size!=expected:raise ValueError(f"Incomplete card map: {mapping}")
            for path in (counts,mapping):
                with path.open("rb") as stream:checksum=hashlib.file_digest(stream,"sha256").hexdigest()
                result["maps"][str(path.relative_to(ROOT))]={"bytes":path.stat().st_size,"sha256":checksum}
        return result


def load_card_config(path=None):
    path=Path(path or DEFAULT_CONFIG).resolve()
    fields={}
    for line in path.read_text().splitlines():
        line=line.split("#",1)[0].strip()
        if not line:continue
        key,value=line.split(None,1)
        if key in fields:raise ValueError(f"Duplicate card parameter: {key}")
        fields[key]=value.strip()
    if set(fields)!={"CardAbstractionName","Bucketings"}:
        raise ValueError("Expected named four-street card bucketing configuration")
    name=fields["CardAbstractionName"]
    buckets=tuple(fields["Bucketings"].split(","))
    if len(buckets)!=4 or buckets[0]!="null" or any(
            not re.fullmatch(r"[A-Za-z0-9_]+",value) for value in (name,*buckets)):
        raise ValueError("Invalid card abstraction; preflop must use the 169 null classes")
    return CardConfig(path,name,buckets)
