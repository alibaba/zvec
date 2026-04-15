#!/bin/bash

project_name=proxima-zvec
gcov_tool=gcov
zip_html=false
output_name=html
keep_info=false

script_dir=$(cd "$(dirname "$0")"; pwd)
source_base=$(dirname "$script_dir")
filter_list="'*/tests/*' '*/thirdparty/*' '*/deps/*' '*/proto/*' '*/external/*' '*/sqlengine/antlr/gen/*'"

while getopts t:p:o:zk option; do
  case "$option" in
  t)
    gcov_tool=$OPTARG;;
  p)
    project_name=$OPTARG;;
  o)
    output_name=$OPTARG;;
  z)
    zip_html=true;;
  k)
    keep_info=true;;
  esac
done

# Process sources
lcov -c -b "$source_base" -d . -o $project_name.lcov.info --gcov-tool=$gcov_tool --no-external --keep-going --ignore-errors unused,mismatch || exit 1
eval $(echo lcov -r $project_name.lcov.info -o $project_name-filtered.lcov.info $filter_list --ignore-errors unused,mismatch) || exit 1

# Gather HTML files
genhtml -t "$project_name" -o $output_name $project_name-filtered.lcov.info || exit 1
if [ "$keep_info" = false ]; then
  rm -rf *.lcov.info
fi

# Zip HTML files
if $zip_html ; then
  zip -r $output_name.zip $output_name/
fi
