#!/usr/bin/env bash
# Apply neoquadrant's patches to the vendored Qdrant submodule. Safe to run repeatedly.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
qdrant="$root/vendor/qdrant"

git -C "$root" submodule update --init --depth 1 vendor/qdrant

for patch in "$root"/patches/*.patch; do
  name="$(basename "$patch")"
  if git -C "$qdrant" apply --reverse --check "$patch" 2>/dev/null; then
    echo "already applied: $name"
  elif git -C "$qdrant" apply --check "$patch" 2>/dev/null; then
    git -C "$qdrant" apply "$patch"
    echo "applied:         $name"
  else
    echo "FAILED to apply: $name (Qdrant version changed? rebase the patch)" >&2
    exit 1
  fi
done
