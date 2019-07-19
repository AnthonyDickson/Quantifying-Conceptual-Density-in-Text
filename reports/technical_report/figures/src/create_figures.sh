#!/usr/bin/env bash

# Check we are in the correct directory
cwd=${PWD##*/} # the last part of the path to the working directory

if [[ ${cwd} == "src" ]]; then
    echo "ERROR: Wrong directory." 1>&2
    echo "Do not call this script from within the src/ directory." 1>&2
    echo "Change directory to the parent directory." 1>&2
    exit 1
fi

# Clean the directory
find ./* -maxdepth 0 -type f -delete

# Create soft links for text files
input="src/documents.txt"

while IFS= read -r file
do
    ln -sf ../../../docs/${file} ${file}
done < "$input"

# Add root directory to python path
export PYTHONPATH=${PYTHONPATH}:$(realpath ../../../)

# Run all of the scripts in figures/src
for script in src/*.py; do
	echo "Running '$script'..."
	python ${script}
done

# Delete all intermediate files (i.e. anything but .png files).
find ./* -maxdepth 0 -type f,l ! -name '*.png' -delete

echo "Done!"