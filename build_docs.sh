#!/bin/bash
set -e

if ! command -v uv &> /dev/null
then
    echo "UV is not installed. Please follow the documentation on the contributing page to install UV."
    exit 1
fi

cd "$(dirname "$0")"

uv venv
uv sync --dev

uv run zensical build

echo "Documentation built successfully in the 'site' directory."
echo "To preview, run: uv run zensical serve"
