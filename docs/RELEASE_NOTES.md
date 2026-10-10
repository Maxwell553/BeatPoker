# BeatPoker v1.0.0

Runnable heads-up no-limit Hold’em for 50–1000 bb, packaged from the Poker and PokerBot workspaces. Includes a local browser table, CLI, Python API, nine selected full-game checkpoints, exact-card native fallback, optional turn/river search, matching native source, and integrity-checked setup.

The attached `beatpoker-models-v1.part001` contains all 89 runtime model/card-map files (926,041,141 bytes compressed; approximately 6.47 GB installed). Clone the source and run `python3 scripts/setup.py` to download, verify, install, and check the package. No training or Git LFS is required.

For offline model installation, download the attachment into a directory and run `python3 scripts/setup.py --artifacts /path/to/assets`. The source repository contains the full checksum manifest at `artifacts/manifest.json`.

Apple Silicon macOS executables are included, with source builds for other supported Unix platforms. Validation: 122 tests passed in a clean installation; 34 runtime checks passed with rebuilt engines; browser smoke completed. Professional-level playing strength remains unproven. See `docs/MODEL_CARD.md` and `docs/VALIDATION.md` for limitations.

Model bundle SHA-256:

```text
35861ae6a7f875dc6e980debb08f1e4ebecdace597dffde2acf16cb7fbff1805  beatpoker-models-v1.part001
```
