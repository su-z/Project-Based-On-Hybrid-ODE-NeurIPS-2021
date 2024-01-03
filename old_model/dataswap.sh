#!/bin/bash

SUFFIX=$1
PREFIXES=(data results experiments/Fig3.png experiments/Fig3.ipynb)

for PREFIX in "${PREFIXES[@]}"
do
  lnswap "$(basename "$PREFIX").$SUFFIX" "$PREFIX"
done
