"""Build and publish an immutable native executable without replacing live ones."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build(jobs=2,activate=True):
    if jobs<1:raise ValueError("Positive build parallelism required")
    target=ROOT/"native/target/publish-build"
    subprocess.run(["cargo","build","--release","--locked","--manifest-path",str(ROOT/"native/Cargo.toml"),
                    "--jobs",str(jobs)],check=True,cwd=ROOT,env={**os.environ,"CARGO_TARGET_DIR":str(target)})
    binary=target/"release/hu-solver";fingerprint=digest(binary)
    release=ROOT/"native/releases"/f"hu-solver-{fingerprint}"
    release.parent.mkdir(parents=True,exist_ok=True)
    if release.exists():
        if digest(release)!=fingerprint:raise ValueError("Existing immutable release is corrupt")
    else:
        temporary=release.with_suffix(f".tmp.{os.getpid()}")
        shutil.copy2(binary,temporary);temporary.replace(release)
    sources=[ROOT/"native/Cargo.toml",ROOT/"native/Cargo.lock",ROOT/"vendor/postflop-solver/Cargo.toml"]
    sources+=list((ROOT/"native/src").rglob("*.rs"))+list((ROOT/"vendor/postflop-solver/src").rglob("*.rs"))
    record={"binary":str(release.relative_to(ROOT)),"sha256":fingerprint,"built_unix":time.time(),
            "rustc":subprocess.check_output(["rustc","--version"],text=True).strip(),"platform":platform.platform(),
            "rustflags":os.environ.get("RUSTFLAGS",""),
            "source_sha256":{str(path.relative_to(ROOT)):digest(path) for path in sources}}
    encoded=json.dumps(record,indent=2)+"\n"
    release.with_suffix(".build.json").write_text(encoded)
    if activate:
        manifest=ROOT/"native/runtime.json";temporary=manifest.with_suffix(f".tmp.{os.getpid()}")
        temporary.write_text(encoded);temporary.replace(manifest)
    print(json.dumps({"binary":record["binary"],"sha256":fingerprint,"published":True,"activated":activate}))
    return record


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--jobs",type=int,default=2)
    parser.add_argument('--no-activate',action='store_true',help='Keep the current runtime selection unchanged')
    args=parser.parse_args();build(args.jobs,activate=not args.no_activate)
