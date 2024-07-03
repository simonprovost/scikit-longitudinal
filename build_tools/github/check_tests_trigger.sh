#!/bin/bash

set -e
set -x

COMMIT_MSG=$(git log --no-merges -1 --oneline)

# The commit marker "[cd build]" or "[cd build gh]" will trigger the build when required
if [[ "$COMMIT_MSG" =~ \[cd\ tests\] ]]; then
    echo "tests=true" >> $GITHUB_OUTPUT
fi
