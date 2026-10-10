# BeatPoker

Local heads-up no-limit Hold’em, with a browser table, terminal interface, and Python API. Supports starting effective stacks from **50 to 1000 big blinds**, including fractional depths.

This release packages the trained full-game bot from the Poker workspace, built on PokerBot’s engine and range code. It uses the strongest published checkpoint selection supported by the existing comparisons. **Professional-level or superhuman strength has not been established.** See [model selection and evidence](docs/MODEL_CARD.md).

## Install and play

Requirements: Python 3.11+, macOS or Linux, at least 12 GB free disk space, and preferably 16 GB RAM. Apple Silicon macOS executables are included. Linux and Intel Macs additionally need a C++17 compiler, Make, and a stable Rust toolchain with Cargo; setup builds executables there. Windows requires WSL2. This release was tested on Apple Silicon macOS; other platforms have a source build path, not a validation claim.

```sh
git clone https://github.com/Maxwell553/BeatPoker.git
cd BeatPoker
python3 scripts/setup.py
.venv/bin/python play_gui.py --stack 200
```

Open **http://127.0.0.1:8765**. The table binds only to localhost. Stacks reset between hands. Initial Python JIT compilation can take extra time.

Setup creates `.venv`, installs Python dependencies, downloads the **926 MB model bundle** from this repository’s [v1.0.0 release](https://github.com/Maxwell553/BeatPoker/releases/tag/v1.0.0), verifies SHA-256 checksums, installs about 6.47 GB of runtime data, and runs the runtime checks. No training, paid service, API key, or Git LFS is needed. All required trained models are included in that release; they are stored separately from Git history.

For an offline model install, download the release asset(s) to a directory and run `python3 scripts/setup.py --artifacts /path/to/assets`. Python packages still require an installed package cache or network access. To rebuild executables on macOS, add `--build`.

Terminal play:

```sh
.venv/bin/python -m hu_bot --stack 137.5
```

Actions: `f` fold, `x` check, `c` call, `a` all-in, `b2.5` bet 2.5 bb, `r9` raise to 9 bb. Sizes are big blinds unless explicitly suffixed with `chips`.

## Policy options

The default is unraked heads-up play with trained strategies on all four streets. Models cover 50, 75, 100, 150, 200, 300, 500, 750, and 1000 bb. Other stack sizes use the nearest depth by proportional distance, with bet translation. Unsupported action histories trigger an explicit native-search fallback. Every decision records its source and any fallback in `last_decision`.

Optional exact-card turn and river search:

```sh
.venv/bin/python play_gui.py --stack 500 --resolve-from turn \
  --seconds 30 --native-iterations 1000
```

For more search effort, use `--seconds 120 --native-iterations 4000 --native-uncompressed`. Uncompressed search uses more RAM. Search budgets exclude some setup and measurement overhead, so they are not hard response deadlines. The default per-bot solver memory cap is 8 GiB. Solver errors are surfaced rather than hidden behind a heuristic.

Later-street search is optional: improved convergence in its restricted tree has not established stronger overall play. `--native` selects the older chart/preflop policy with native postflop search. Rake modes also select that chart mode; the released trained models are unraked. Experimental `--river-gadget` is retained as an opt-in feature, without a strength claim.

## Python API

```python
from hu_bot import StackAdaptiveBot
from poker.game import GameConfig, NLHGame

game = NLHGame(GameConfig(num_players=2, stack_bb=500))
game.start_hand()
bot = StackAdaptiveBot(seed=7)
try:
    game.apply(bot.act(game.view(game.to_act)))
    print(bot.last_decision)
    bot.reset()  # call between hands
finally:
    bot.close()
```

Only pass the bot its own `PlayerView`. The interface does not need an opponent’s private cards.

## Check the installation

```sh
.venv/bin/python scripts/doctor.py
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

Doctor hashes all model files, loads every selected checkpoint, checks legal preflop actions, and runs a converged native river solve. Tests cover engine rules, range parsing, stack selection, artifact integrity, and legal decisions on every street at twelve stack sizes in both positions. These are software checks, not a certification of playing strength.

To repair missing or changed model files, rerun `scripts/install_artifacts.py`. Downloads are cached in `.cache/artifacts`, which can be removed after installation. The model inventory is in [`artifacts/manifest.json`](artifacts/manifest.json); compiler/source provenance is in [`artifacts/provenance.json`](artifacts/provenance.json). Exact Python runtime versions used for release validation are in `requirements-tested.txt`.

## Source and licensing

The repository includes the matching source for both compiled engines. Original PokerBot modules remain as implementation dependencies; the supported product entry points are the ones above. Unused checkpoints, DecisionHoldem downloads, research runs, credentials, virtual environments, caches, and `reel/` are excluded.

Distributed under **AGPL-3.0-or-later**, with the MIT-licensed Slumbot component retaining its original license. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
