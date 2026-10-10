# Third-party notices

BeatPoker combines the user's PokerBot engine/range implementation with a trained policy reader and a native postflop solver. The combined release and new integration code are distributed under AGPL-3.0-or-later; component notices are retained below.

* **postflop-solver**, b-inary, upstream revision `9d1509fe5077d019825f833eed04b16d342dfda1`: https://github.com/b-inary/postflop-solver. AGPL-3.0-or-later; full license at `vendor/postflop-solver/LICENSE`. Local changes add three explicit references for compatibility with newer Rust. The native wrapper source and pinned Cargo dependency lockfile are included.
* **Slumbot 2019**, Eric Jackson, upstream revision `a74c99dd5e6e2fb50118e3990a25210e9dd0f6b3`: https://github.com/ericgjackson/slumbot2019. MIT; full copyright and license at `vendor/slumbot2019/LICENSE`. Local additions include the read-only policy server, Apple random-number compatibility, deterministic training streams, checkpoint fixes, and memory-mapped card data.
* **PokerBot**, supplied by the repository owner: local engine, range/chart, GUI, and legacy modules reused with later correctness fixes. Prior internal benchmark labels are not claims about this release.

The packaged strategies and abstractions were generated locally using the modified Slumbot trainer. They are not the public Slumbot opponent's private strategy, a licensed commercial solver export, or an authenticated professional equilibrium. No DecisionHoldem strategy data is distributed in this release.

The corresponding source for the included native binary is `native/src/main.rs` plus `vendor/postflop-solver/`, `native/Cargo.toml`, and `native/Cargo.lock`. `artifacts/provenance.json` records their hashes. Python and Rust dependencies retain their respective upstream licenses.
