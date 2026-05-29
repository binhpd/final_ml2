#!/bin/bash
# Wrapper train kèm caffeinate -i để Mac không sleep khi train xuyên đêm
# Usage: ./caffeinate_train.sh u2net|yolo

set -e

cd "$(dirname "$0")/../.."

# Fix libexpat trên macOS Sequoia/Tahoe + Homebrew python@3.12
# Note: /usr/bin/caffeinate strip DYLD_* do SIP. Phải pass qua env bên trong.
DYLD_FIX="env DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib"

VENV=venv_ml2/bin/python

case "$1" in
  u2net)
    echo "[caffeinate] U2-Net train..."
    /usr/bin/caffeinate -i $DYLD_FIX "$VENV" ml2/u2net/train.py --config ml2/u2net/configs/doc_lite_planB.yaml
    ;;
  yolo)
    echo "[caffeinate] YOLO train..."
    /usr/bin/caffeinate -i $DYLD_FIX "$VENV" ml2/yolo_seg/train.py --data ml2/data/yolo_doc/doc.yaml --epochs 150 --device mps
    ;;
  *)
    echo "Usage: $0 {u2net|yolo}"
    exit 1
    ;;
esac
