#!/bin/bash
for i in {51..54}; do
    python download_YT_videos.py --input-file YouTube/links/duration-question-check-ids.json --start $i --bin-size 100
    aws s3 cp YouTube/audios/ s3://meeting-datasets/YouTube/audios --recursive
    rm YouTube/audios/* 
done
wait