#!/bin/bash
for i in {0..10}; do
    python align_punctuate.py --start $i &
done
wait
