# Release validation

Validated locally on 2026-10-09, macOS 26.6.2 / Apple Silicon, Python 3.14.7.

* Installed into a separate clean checkout and a newly created virtual environment using `scripts/setup.py --artifacts ../assets`.
* Decompressed the release bundle and verified all 89 runtime data files against SHA-256 and expected byte lengths. Selected checkpoint configs and preflop exports also passed their catalog hashes.
* `scripts/doctor.py` loaded all nine published depths, returned legal actions, verified matching native source, and completed a converged native river diagnostic.
* `python -m pytest -q`: **122 passed in 20.63 seconds**, with no skips. Includes all-street legal actions at 50, 62.5, 75, 100, 137.5, 150, 200, 300, 500, 750, 999.5, and 1000 bb in both positions, plus engine/evaluator, parsing, configuration, and corrupt/incomplete/unsafe archive checks.
* Rebuilt the C++ policy reader from an empty object directory and rebuilt the Rust solver with `--locked` from the source matching the pinned binary. **34 native and packaged-runtime tests passed in 11.38 seconds** with the rebuilt executables.
* Browser smoke at 137.5 bb: loaded the local table and assets, called preflop, checked the flop, and observed legal bot responses and progression to the turn.
* Checked the tracked file list: no `reel/`, virtual environment, model working files, experiment reports, object/build caches, or credential files. A targeted common-secret-pattern scan found no matches in tracked files.

The included native executable is pinned by SHA-256 and has matching source provenance. Compiler warnings in vendored code did not prevent clean source builds.

## Published installation and compatibility follow-up

The [full Linux workflow](https://github.com/Maxwell553/BeatPoker/actions/runs/38018706171) built both engines, downloaded the actual published bundle, installed all 89 verified model files, passed **122 tests in 14.73 seconds with no skips**, and passed doctor. GitHub's release-asset digest independently matched the local bundle hash. A fresh public source clone matched the published commit and ignored `reel/`.

Version 1.0.1 adds automatic source builds on Apple Silicon Macs older than macOS 26, because the shipped C++ executable's deployment target is 26.0. **Seven platform-selection tests passed** for old/new Apple Silicon macOS, Intel macOS, and Linux. This adds installer coverage without changing the model bundle or playing policy. Intel macOS and WSL2 remain documented build paths without an end-to-end test on those platforms.

These results establish the package's local operability and tested software behavior. They do not establish professional playing strength, a guaranteed win rate, or full-game exploitability bounds.
