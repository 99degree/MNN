#!/usr/bin/env bash
# replace_libMNN.sh
# Utility to replace the built libMNN.so with a compatible version.
# Usage: ./replace_libMNN.sh /path/to/new/libMNN.so
# This script copies the provided library into the build output directory
# used by the profiling examples (build_nodeprof/OFF/).

set -e

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/new/libMNN.so"
    exit 1
fi

NEW_LIB="$1"

if [[ ! -f "$NEW_LIB" ]]; then
    echo "Error: File not found: $NEW_LIB"
    exit 1
fi

TARGET_DIR="$(dirname "$(realpath "$0")")/build_nodeprof/OFF"
TARGET_LIB="$TARGET_DIR/libMNN.so"

mkdir -p "$TARGET_DIR"
cp -f "$NEW_LIB" "$TARGET_LIB"
chmod 755 "$TARGET_LIB"

echo "Replaced libMNN.so with $NEW_LIB"
