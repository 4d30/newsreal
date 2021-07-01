#!/bin/sh

wc -w $1/* |\
awk 'BEGIN {MAX = 0};
    /txt/{SUM+=$1;
    COUNT+=1; 
    if ($1 > MAX)
      MAX = $1}; 
    END {AVG = SUM/COUNT; printf "Avg = %d MAX = %d\n", AVG, MAX}'
