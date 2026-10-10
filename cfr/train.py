"""Overnight linear MCCFR. Writes atomic checkpoints the bot can load mid-train.

    PYTHONPATH=. python -m cfr.train --game v3 --iters 0
    python -m cfr.train --game hu --iters 0
    python -m cfr.train --game 6maxv2 --iters 40000 --workers 8
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from cfr.mccfr import HUTrainer, SixMaxTrainer
from cfr.sixmax_v2 import SixMaxV2Trainer
from cfr.sixmax_v3 import (
    HU_V3_PATH,
    SIX_V3_PATH,
    TREE_VERSION,
    HU_TREE_VERSION,
    HUV3Trainer,
    SixMaxV3Trainer,
)
from strategy.card_abs import save_clusters
from strategy.library import LIB_PATH, build_tables

DATA = Path(__file__).resolve().parents[1] / "Final_Bot" / "data"
HU_PATH = DATA / "blueprint_hu_100.npz"
SIX_PATH = DATA / "blueprint_6max_100.npz"
SIX_V2_PATH = DATA / "blueprint_6max_v2_100.npz"


def _ensure_library() -> None:
    if not LIB_PATH.exists():
        build_tables()


def _worker_6maxv2(args):
    seed, n_iters, start_path = args
    from pathlib import Path as P

    from cfr.sixmax_v2 import SixMaxV2Trainer

    t = SixMaxV2Trainer.load(P(start_path)) if P(start_path).exists() else SixMaxV2Trainer()
    rng = np.random.default_rng(int(seed))
    t.iterate_batch(int(n_iters), rng)
    return t.sum_pf, t.sum_fl, t.reg_pf, t.reg_fl, t.cv_num, t.cv_den, t.iters


def train_6maxv2_parallel(iters: int, workers: int, seed: int) -> None:
    from concurrent.futures import ProcessPoolExecutor

    save_clusters()
    _ensure_library()
    path = SIX_V2_PATH
    workers = max(1, int(workers))
    per = max(1, int(iters) // workers)
    try:
        from cfr.kernel_6max import warmup

        print("compiling 6-max kernel...", flush=True)
        t_w = time.time()
        ok = warmup()
        print(f"  kernel compiled={ok}  {time.time() - t_w:.1f}s", flush=True)
    except Exception as e:
        print(f"  kernel warmup skipped: {e}", flush=True)
    jobs = [(seed + 17 * i + 3, per, str(path)) for i in range(workers)]
    print(f"MCCFR 6maxv2 parallel workers={workers} per={per} -> {path}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_worker_6maxv2, jobs))
    n = len(results)
    base = SixMaxV2Trainer.load(path) if path.exists() else SixMaxV2Trainer()
    base.sum_pf = sum(r[0] for r in results) / n
    base.sum_fl = sum(r[1] for r in results) / n
    base.reg_pf = sum(r[2] for r in results) / n
    base.reg_fl = sum(r[3] for r in results) / n
    base.cv_num = sum(r[4] for r in results) / n
    base.cv_den = sum(r[5] for r in results) / n
    base.iters = int(max(r[6] for r in results))
    base.save(path)
    dt = time.time() - t0
    print(
        f"done 6maxv2 iters={base.iters}  {iters / max(dt, 1e-6):.1f} job-iter/s  {dt:.1f}s",
        flush=True,
    )


def train_loop(game: str, iters: int, ckpt_every: int, seed: int) -> None:
    save_clusters()
    rng = np.random.default_rng(seed)
    _ensure_library()
    if game == "hu":
        path = HU_PATH
        trainer = HUTrainer.load(path) if path.exists() else HUTrainer()
        step = trainer.iterate
    elif game == "6maxv2":
        path = SIX_V2_PATH
        trainer = SixMaxV2Trainer.load(path) if path.exists() else SixMaxV2Trainer()
        step = trainer.iterate
    else:
        path = SIX_PATH
        trainer = SixMaxTrainer.load(path) if path.exists() else SixMaxTrainer()
        step = trainer.iterate
    n = 0
    t0 = time.time()
    target = iters if iters > 0 else 10**12
    print(f"MCCFR {game} start iters={trainer.iters} -> {path}", flush=True)
    while n < target:
        step(rng)
        n += 1
        if trainer.iters % ckpt_every == 0:
            trainer.save(path)
            dt = time.time() - t0
            rate = n / max(dt, 1e-6)
            print(f"  {game} iters={trainer.iters}  {rate:.0f}/s  wrote {path.name}", flush=True)
    trainer.save(path)
    print(f"done {game} iters={trainer.iters}", flush=True)


def _archive_stale_v3(path: Path, label: str, need: int) -> None:
    """Keep a copy of an older tree checkpoint before overwrite."""
    if not path.exists():
        return
    try:
        z = np.load(path)
        tree = int(z["tree"]) if "tree" in z.files else 0
        iters = int(z["iters"]) if "iters" in z.files else 0
    except Exception:
        return
    if tree >= need:
        return
    dest = path.with_name(f"{path.stem}.tree{tree}{path.suffix}")
    if not dest.exists():
        shutil.copy2(path, dest)
        print(f"archived {label} tree {tree} ({iters} iters) -> {dest.name}", flush=True)
    else:
        print(f"stale {label} tree {tree} already archived as {dest.name}", flush=True)


def train_v3(iters: int, ckpt_every: int, seed: int, batch: int) -> None:
    """All-street 6-max + HU blueprint. `--iters 0` runs until killed."""
    save_clusters()
    _ensure_library()
    try:
        from cfr.kernel_v3 import warmup

        print("compiling v3 kernel (first batch is slow)...", flush=True)
        t_w = time.time()
        ok = warmup()
        print(f"  kernel compiled={ok}  {time.time() - t_w:.1f}s", flush=True)
    except Exception as e:
        print(f"  kernel warmup failed: {e}", flush=True)
        raise
    _archive_stale_v3(SIX_V3_PATH, "6max", TREE_VERSION)
    _archive_stale_v3(HU_V3_PATH, "HU", HU_TREE_VERSION)
    six = SixMaxV3Trainer.load(SIX_V3_PATH) if SIX_V3_PATH.exists() else SixMaxV3Trainer()
    hu = HUV3Trainer.load(HU_V3_PATH) if HU_V3_PATH.exists() else HUV3Trainer()
    if SIX_V3_PATH.exists() and six.iters == 0:
        print(f"ignored stale 6max v3 checkpoint (need tree {TREE_VERSION})", flush=True)
    if HU_V3_PATH.exists() and hu.iters == 0:
        print(f"ignored stale HU v3 checkpoint (need tree {HU_TREE_VERSION})", flush=True)
    rng = np.random.default_rng(seed)
    target = iters if iters > 0 else 10**12
    n = 0
    t0 = time.time()
    batch = max(256, int(batch))
    hu_batch = max(64, batch // 4)
    print(
        f"MCCFR v3 start 6max={six.iters} hu={hu.iters} batch={batch} -> {SIX_V3_PATH.name}",
        flush=True,
    )
    while n < target:
        step = min(batch, target - n)
        six.iterate_batch(step, rng)
        hu.iterate_batch(min(hu_batch, step), rng)
        n += step
        if six.iters % max(ckpt_every, 1) < step or n >= target:
            six.save(SIX_V3_PATH)
            hu.save(HU_V3_PATH)
            dt = time.time() - t0
            rate = n / max(dt, 1e-6)
            print(
                f"  6max={six.iters} hu={hu.iters}  {rate:.0f} 6max-hands/s  wrote checkpoints",
                flush=True,
            )
    six.save(SIX_V3_PATH)
    hu.save(HU_V3_PATH)
    print(f"done v3 6max={six.iters} hu={hu.iters}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=("hu", "6max", "6maxv2", "v3", "both"), default="v3")
    ap.add_argument("--iters", type=int, default=0, help="0 = run until killed")
    ap.add_argument("--ckpt-every", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=8192, help="v3 hands per kernel call")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="Process workers for 6maxv2")
    args = ap.parse_args()
    if args.game == "v3":
        train_v3(args.iters, args.ckpt_every, args.seed, args.batch)
        return
    if args.game == "6maxv2" and args.workers > 1 and args.iters > 0:
        train_6maxv2_parallel(args.iters, args.workers, args.seed)
        return
    if args.game == "both":
        save_clusters()
        rng = np.random.default_rng(args.seed)
        hu = HUTrainer.load(HU_PATH) if HU_PATH.exists() else HUTrainer()
        sm = SixMaxTrainer.load(SIX_PATH) if SIX_PATH.exists() else SixMaxTrainer()
        n = 0
        t0 = time.time()
        target = args.iters if args.iters > 0 else 10**12
        print("MCCFR both start", flush=True)
        while n < target:
            for _ in range(8):
                hu.iterate(rng)
                n += 1
            sm.iterate(rng)
            n += 1
            if hu.iters % args.ckpt_every == 0:
                hu.save(HU_PATH)
                sm.save(SIX_PATH)
                dt = time.time() - t0
                print(f"  hu={hu.iters} 6max={sm.iters}  {n/max(dt,1e-6):.0f}/s", flush=True)
        hu.save(HU_PATH)
        sm.save(SIX_PATH)
        return
    train_loop(args.game, args.iters, args.ckpt_every, args.seed)


if __name__ == "__main__":
    main()
