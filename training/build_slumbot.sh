#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root/vendor/slumbot2019"
mkdir -p obj bin
make -f Makefile -f ../../training/extra.mk -j2 CFLAGS='-std=c++17 -O3 -march=native -include ../../training/slumbot_compat.h' \
    LIBRARIES='-pthread' "$@"
