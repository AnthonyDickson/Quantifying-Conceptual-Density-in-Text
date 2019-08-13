#!/usr/bin/env bash

path_to_pdfs="${1}"

mkdir -p documents

for d in "${path_to_pdfs}"/*; do
  echo Processing documents in "${d}"

  for filename in "${d}"/*.pdf; do
      path="${filename##*${path_to_pdfs}/}"
      output_path=${PWD}/documents/textbooks/"${path%.*}.txt"

      echo Creating file at ${output_path}...
      mkdir -p $(dirname ${output_path})
      touch ${output_path}
      python qcd/parse_pdf.py "${filename}" -o "${output_path}"
  done
done
