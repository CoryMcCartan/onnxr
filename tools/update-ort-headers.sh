#!/usr/bin/env bash
#
# Download the ONNX Runtime C/C++ headers from the latest GitHub release
# and install them into src/onnxruntime/.
#
# Usage: bash tools/update-ort-headers.sh [VERSION]
#   VERSION  Optional ORT version (e.g. 1.26.0). Defaults to latest release.

set -euo pipefail
cd "$(dirname "$0")/.."

DEST="src/onnxruntime"

# Headers we need (closed dependency set for onnxruntime_cxx_api.h)
KEEP_HEADERS=(
    onnxruntime_c_api.h
    onnxruntime_cxx_api.h
    onnxruntime_cxx_inline.h
    onnxruntime_float16.h
    onnxruntime_ep_c_api.h
)

# Resolve version
if [ -n "${1:-}" ]; then
    VERSION="$1"
else
    echo "Fetching latest release tag..."
    VERSION=$(curl -sL https://api.github.com/repos/microsoft/onnxruntime/releases/latest \
        | grep '"tag_name"' | sed 's/.*"v\(.*\)".*/\1/')
fi
echo "Using ONNX Runtime v${VERSION}"

# Pick any platform tarball (headers are identical across platforms)
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-osx-arm64-${VERSION}.tgz"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "Downloading ${URL}..."
curl -sL "$URL" | tar xz -C "$TMPDIR"

INCLUDE_DIR="$TMPDIR/onnxruntime-osx-arm64-${VERSION}/include"
if [ ! -d "$INCLUDE_DIR" ]; then
    echo "ERROR: include/ not found in tarball" >&2
    exit 1
fi

# Copy only the headers we need
rm -rf "$DEST"
mkdir -p "$DEST"
for h in "${KEEP_HEADERS[@]}"; do
    if [ ! -f "$INCLUDE_DIR/$h" ]; then
        echo "ERROR: $h not found in release" >&2
        exit 1
    fi
    cp "$INCLUDE_DIR/$h" "$DEST/$h"
done

# Strip diagnostic-suppressing pragmas that CRAN flags
echo "Stripping diagnostic-suppressing pragmas..."
for f in "$DEST"/*.h; do
    sed -i '' \
        -e '/#pragma.*diagnostic.*push/d' \
        -e '/#pragma.*diagnostic.*pop/d' \
        -e '/#pragma.*diagnostic.*ignored/d' \
        "$f"
done

echo "Done. Headers installed to ${DEST}/:"
ls -1 "$DEST"
echo ""
echo "ORT version: ${VERSION}"
echo "Remember to update .ort_version in R/runtime.R if the version changed."
