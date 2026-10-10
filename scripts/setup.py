"""Create the environment, install bundled models, and prepare executables."""
import argparse
from pathlib import Path
import platform
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def needs_build(system, machine, mac_version):
    # The shipped C++ reader targets macOS 26; Rust alone targets macOS 11.
    # Rebuild both together on older Macs instead of attempting an incompatible binary.
    return not (system == 'Darwin' and machine == 'arm64'
                and mac_version and int(mac_version.split('.')[0]) >= 26)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, help='Directory containing downloaded model bundle parts')
    parser.add_argument('--build', action='store_true', help='Rebuild both executables from source')
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error('Python 3.11 or newer is required')
    if platform.system() not in ('Darwin', 'Linux'):
        parser.error('Use macOS or Linux (Windows: run inside WSL2)')
    environment = ROOT / '.venv'
    if not (environment / 'bin/python').is_file():
        venv.EnvBuilder(with_pip=True).create(environment)
    python = str(environment / 'bin/python')
    def run(*command):
        subprocess.run(command, cwd=ROOT, check=True)
    run(python, '-m', 'pip', 'install', '-r', 'requirements.txt')
    artifact_args = ['--from-dir', str(args.artifacts.resolve())] if args.artifacts else []
    run(python, 'scripts/install_artifacts.py', *artifact_args)
    if args.build or needs_build(platform.system(), platform.machine(), platform.mac_ver()[0]):
        run('sh', 'training/build_slumbot.sh', 'bin/policy_server')
        run(python, 'training/build_native.py')
    run(python, 'scripts/doctor.py')
    print('\nReady. Start the table: .venv/bin/python play_gui.py --stack 200')


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(f'Setup failed (exit {exc.returncode}). Fix the reported error and rerun setup.')
