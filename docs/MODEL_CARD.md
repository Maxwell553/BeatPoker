# Model selection and limits

The release combines the corrected PokerBot engine and range machinery with the newer full-game policy catalog and its native exact-card fallback. This is the best-supported **published configuration** in the two local workspaces, not a claim that its global optimality has been established.

| Starting model depth (bb) | Checkpoint batch | Training deals |
| --- | --- | --- |
| 50, 75, 100, 150, 200 | 99 | 100 million each |
| 300, 500, 750, 1000 | 299 | 300 million each |

Each deeper model beat its previous 100-million-deal checkpoint in two separate 100,000-pair comparisons. Training longer did not consistently improve other depths, so their prior published selections remain. The older chart strategy from PokerBot is retained for native/raked modes and implementation compatibility. There is no statistically sound comparison establishing that every selected policy beats every possible alternative.

## Evidence

At 200 bb, the published baseline completed 9,992 independent Slumbot hands at **−15.26 bb/100**, with a descriptive 95% interval of **[−47.32, +16.80]**. Eight transport interruptions remained unknown. A 30-second turn-search cohort originally completed 7,445 hands at **+6.23 bb/100**, interval **[−32.35, +44.81]**, with fourteen unknown outcomes; its continuation was still research in progress when packaging began. These results do not establish winning play against that opponent.

A separate 500 bb paired pilot comparing 120-second / 4,000-iteration uncompressed search with the 30-second / 1,000-iteration compressed setting was inconclusive: adjusted **+5.86 bb/100**, interval **[−53.31, +65.03]**, over 1,000 pairs. Better restricted-tree convergence is a reason to offer the higher budget, not to label it a proven stronger default.

An imported DecisionHoldem policy supported only 200 bb. Its independent confirmation against the local policy was inconclusive, and targeted river opponents found substantial weaknesses. Its 35.6 GB import and experimental wrappers are excluded from this product. Experimental flop search, alternative card buckets, policy mixtures, and late training averages likewise lacked sufficient evidence for promotion.

## Practical limitations

* No version is validated as professional-level, superhuman, or approximately unexploitable across 50–1000 bb.
* Intermediate depths use nearest-depth translation, not a solved strategy for each continuous starting stack.
* Card buckets lose information. The trained betting tree and off-tree bet translation introduce approximation.
* Native search solves supplied public ranges in a restricted betting tree. Its gap is half NashConv for that subgame, not exploitability of the unrestricted full game.
* Range-based re-solving is not proven safe continual resolving; changed strategies can invalidate assumptions about earlier play.
* The native chart mode and raked play are not covered by the unraked trained-policy results.
* Latency and memory use vary sharply by position. A search error is reported to the caller.

Software validation of this packaged release is recorded in `docs/VALIDATION.md`. Research outputs in the original workspace were preserved separately; they are not required to run the product.
