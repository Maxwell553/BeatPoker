# BeatPoker v1.0.1

Runnable heads-up no-limit Hold’em for 50–1000 bb, packaged from the Poker and PokerBot workspaces. Includes a local browser table, CLI, Python API, nine selected full-game checkpoints, exact-card native fallback, optional turn/river search, matching native source, and integrity-checked setup.

This patch automatically builds from source on Macs older than macOS 26; the bundled C++ executable requires macOS 26+. Seven platform-selection checks cover that behavior. The playing policy and model files are unchanged.

The immutable [model bundle](https://github.com/Maxwell553/BeatPoker/releases/download/v1.0.0/beatpoker-models-v1.part001) from the [v1.0.0 model release](https://github.com/Maxwell553/BeatPoker/releases/tag/v1.0.0) contains all 89 runtime model/card-map files (926,041,141 bytes compressed; approximately 6.47 GB installed). Version 1.0.1 reuses that asset. Clone the source and run `python3 scripts/setup.py` to download, verify, install, and check the package. No training or Git LFS is required.

For offline model installation, download the linked bundle into a directory and run `python3 scripts/setup.py --artifacts /path/to/assets`. The source repository contains the full checksum manifest at `artifacts/manifest.json`.

Apple Silicon macOS 26+ executables are included, with source builds for other supported Unix platforms. Validation: 122 tests passed in a clean macOS installation; 34 runtime checks passed with rebuilt engines; browser smoke completed; the complete published-model installation passed all 122 tests and doctor on Linux. Seven additional installer platform checks passed. Professional-level playing strength remains unproven. See `docs/MODEL_CARD.md` and `docs/VALIDATION.md` for limitations.

Model bundle SHA-256:

```text
35861ae6a7f875dc6e980debb08f1e4ebecdace597dffde2acf16cb7fbff1805  beatpoker-models-v1.part001
```
