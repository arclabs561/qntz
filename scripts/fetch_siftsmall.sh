#!/usr/bin/env bash
# Fetch the SIFT-small ANN dataset (10k base + 100 query 128-d vectors).
#
# Source: TEXMEX corpus (IRISA), the standard small SIFT benchmark. fvecs format
# (per vector: int32 dim, then dim float32). Data lands in qntz/data/siftsmall/
# which is gitignored.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data"
URL="ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"

mkdir -p "$DEST"
if [ -f "$DEST/siftsmall/siftsmall_base.fvecs" ]; then
  echo "have siftsmall/ -> $DEST/siftsmall"
  exit 0
fi

echo "fetching siftsmall.tar.gz"
curl -sSL --fail -o "$DEST/siftsmall.tar.gz" "$URL"
tar xzf "$DEST/siftsmall.tar.gz" -C "$DEST"
rm -f "$DEST/siftsmall.tar.gz"
echo "done -> $DEST/siftsmall"
