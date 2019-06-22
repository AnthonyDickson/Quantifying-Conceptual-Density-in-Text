#!/usr/bin/env bash

rm -rf out/
mkdir out/
cd out/

for script in ../*.py; do
	echo "Running '$script'..."
	python $script
done

find . -type f ! -name '*.png' -delete
