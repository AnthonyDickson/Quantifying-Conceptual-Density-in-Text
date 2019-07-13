#!/usr/bin/env bash

# Create an empty directory
rm -rf out/
mkdir out/

# Create soft links for text files
for doc in *.txt; do
	ln $doc out/$doc
done

# Create soft links for XML files
for doc in *.xml; do
	ln $doc out/$doc
done

# Add root directory to python path
export PYTHONPATH=$PYTHONPATH:$(realpath ..)

# Run all of the scripts in figures/
cd out/

for script in ../*.py; do
	echo "Running '$script'..."
	python $script
done

# Delete all intermediate files (i.e. anything but .png files).
find . -type f ! -name '*.png' -delete