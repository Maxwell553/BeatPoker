"""Verify runtime files and exercise both compiled policy engines."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from scripts.install_artifacts import install, digest
    install(verify_only=True)
    from hu_bot import StackAdaptiveBot
    from hu_bot.checkpoints import load_catalog
    from hu_bot.native import NativeSolver
    from poker.game import GameConfig, NLHGame
    metadata = json.loads((ROOT / 'artifacts/provenance.json').read_text())
    for relative, expected in metadata['source_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f'Packaged native source changed: {relative}')
    for entry in load_catalog():
        game = NLHGame(GameConfig(num_players=2, stack_bb=entry['stack_bb'], rng_seed=7))
        game.start_hand()
        bot = StackAdaptiveBot(seed=7)
        try:
            game.apply(bot.act(game.view(game.to_act)))
            print(f"{entry['stack_bb']:4} bb: checkpoint {entry['batch']} loaded; legal action returned.")
        finally:
            bot.close()
    solver = NativeSolver(threads=2)
    try:
        response = solver.request(dict(cmd='solve', board=['Ks','7h','2c','3d','9s'],
            ranges=['AA,KK,QQ,AKs,A5s','QQ,JJ,TT,AKs,AQs'], pot=600, stack=9700,
            hole=['Ac','Ad'], seconds=3, iterations=1000, compact=True))
        if not response['solve']['converged']:
            raise ValueError('Native river diagnostic did not converge')
        print('Native solver: river diagnostic converged.')
    finally:
        solver.close()
    print('Runtime ready. These checks establish operability, not professional playing strength.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        sys.exit(f'Runtime check failed: {exc}')
