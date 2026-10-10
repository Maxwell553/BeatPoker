"""Export and validate an experimental ECFR preflop checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hu_bot.card_config import load_card_config
from hu_bot.betting_config import load_betting_config
DATA=ROOT/"training/slumbot_data"


def export(betting, checkpoint, batch, output,card_config=None):
    betting,checkpoint,output=map(lambda p:Path(p).resolve(),(betting,checkpoint,output))
    abstraction=load_card_config(card_config)
    betting_model=load_betting_config(path=betting)
    if checkpoint.name!=f"holdem.2.{abstraction.name}.13.4.3.{betting_model.name}.ecfr":
        raise ValueError("Checkpoint directory and card/betting abstractions disagree")
    result=subprocess.run([str(ROOT/"vendor/slumbot2019/bin/export_preflop"),
        str(ROOT/"vendor/slumbot2019/runs/holdem_params"),str(betting),str(checkpoint),str(batch)],
        cwd=DATA,check=True,capture_output=True,text=True)
    model=json.loads(result.stdout)
    raw=(DATA/"buckets.holdem.13.4.3.null.0").read_bytes()
    if len(raw)!=1326*2:
        raise ValueError("Expected 1326 ushort preflop mappings")
    buckets=struct.unpack("<1326H",raw)
    mapping={}
    index=0
    for hi in range(1,52):
        for lo in range(hi):
            # Use suit-independent hand class names, eliminating runtime ID assumptions.
            a,b="23456789TJQKA"[hi//4],"23456789TJQKA"[lo//4]
            name=a+b+("" if a==b else "s" if hi%4==lo%4 else "o")
            bucket=buckets[index]
            if name in mapping and mapping[name]!=bucket:
                raise ValueError("Preflop buckets are not suit-invariant")
            mapping[name]=bucket
            index+=1
    if set(mapping.values())!=set(range(169)) or len(mapping)!=169:
        raise ValueError("Preflop mapping is incomplete")
    model["hand_buckets"]=mapping
    model["card_abstraction"]=abstraction.name
    model["betting_abstraction"]=betting_model.name
    files=[betting,abstraction.path,ROOT/"vendor/slumbot2019/src/ecfr.cpp"]
    files += list(checkpoint.glob(f"sumprobs.x.0.0.0.{batch}.p*.i"))
    model["sha256"]={str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for path,node in model["nodes"].items():
        if len(node["counts"])!=169 or any(len(row)!=len(node["actions"]) for row in node["counts"]):
            raise ValueError(f"Invalid policy table at {path}")
        node["sampled_buckets"]=sum(sum(row)>0 for row in node["counts"])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(model,separators=(",",":"))+"\n")
    return model


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("betting")
    parser.add_argument("checkpoint")
    parser.add_argument("batch",type=int)
    parser.add_argument("output")
    parser.add_argument("--card-config")
    args=parser.parse_args()
    model=export(args.betting,args.checkpoint,args.batch,args.output,args.card_config)
    print(json.dumps({"nodes":len(model["nodes"]),"stack_bb":model["stack_bb"],
                      "root_sampled_buckets":model["nodes"]["root"]["sampled_buckets"]}))
