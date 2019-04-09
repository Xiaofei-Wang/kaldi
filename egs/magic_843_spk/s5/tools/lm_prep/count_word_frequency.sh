#!/bin/bash

input_text=$1 # data/train/text
output_count=$2

cut -d " " -f2- $input_text | tr ' ' '\n' | sort | uniq -c | \
    awk '{print $2" "$1}' | sort -k2rn > $output_count
