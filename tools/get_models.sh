#!/bin/sh
# Download the model files the tests use into build/models/, pinned to the
# revisions in tests/fixtures/*/meta.txt. Skips files already present.
set -eu
for meta in tests/fixtures/*/meta.txt; do
  name=$(basename "$(dirname "$meta")")
  model=$(awk '$1 == "model" {print $2}' "$meta")
  rev=$(awk '$1 == "revision" {print $2}' "$meta")
  dir="build/models/$name"
  for f in config.json model.safetensors vocab.txt 1_Pooling/config.json; do
    [ -s "$dir/$f" ] && continue
    mkdir -p "$(dirname "$dir/$f")"
    curl -fsSL --retry 3 -o "$dir/$f.part" \
      "https://huggingface.co/$model/resolve/$rev/$f"
    mv "$dir/$f.part" "$dir/$f"
  done
done
