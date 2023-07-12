import argparse
import getpass
import glob
import os
import json
import logging
import random
import shutil
import sys
import time
from tqdm import tqdm
from yt_dlp import YoutubeDL
from urllib.request import urlopen
import codecs
from xml.dom import minidom
import datetime
from youtube_transcript_api import YouTubeTranscriptApi

class YTLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)

def convert_to_time(time_str):
    h,m,s = time_str.split(':')
    return datetime.timedelta(hours=int(h),minutes=int(m),seconds=int(s))


def get_all_downloaded_videos(log_dir):
    return [os.path.basename(x).split('.json')[0] for x in glob.glob(os.path.join(log_dir, '*.json'))]


def sample_video(video_list, log_dir, refresh_frequency=125):
    counter = 0
    while True:
        video_id = random.choice(video_list)
        log_filename = os.path.join(log_dir, '{}.json'.format(video_id))
        # Skip video with existing log.
        if os.path.exists(log_filename):
            continue
        counter += 1
        # Refresh list of available videos.
        if (counter % refresh_frequency) == 0:
            downloaded_videos = set(get_all_downloaded_videos(log_dir))
            video_list = [x for x in video_list if x not in downloaded_videos]
        yield video_id


if __name__ == '__main__':
    p = argparse.ArgumentParser('YT download batch script.')
    p.add_argument('--input-file', help='JSON file containing list of youtube IDs.')
    p.add_argument('--start', type=int, help='start indices')
    p.add_argument('--bin-size', type=int, default=100, help='bin size')
    p.add_argument('--output-dir', default=os.path.join('/home/arprasad/YouTube', 'audios'),
                   required=False, help='Output directory where videos will be stored.')
    p.add_argument('--log-dir', default=os.path.join('/home/arprasad/YouTube', 'logs'),
                   required=False, help='Log dir to keep track of downloaded videos so far.')

    args = p.parse_args()
    time.sleep(random.random()) # lazy way to prevent race condition 
    if not os.path.isdir(args.output_dir): os.makedirs(args.output_dir)
    time.sleep(random.random()) # lazy way to prevent race condition
    if not os.path.isdir(args.log_dir): os.makedirs(args.log_dir)
    opts = {'format': '18/22', 'outtmpl': '{}/%(id)s.%(ext)s'.format(args.output_dir),
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
    'prefer_ffmpeg': True, 'logger': YTLogger()}
    # , 
    # 'postprocessor_args': [
    #     '-ar', '16000'
    # ],
    url = 'https://www.youtube.com/watch?v={}'
    with open(args.input_file, 'r') as f: video_list = json.load(f) 
    transcripts_dir = '/home/arprasad/YouTube/transcripts'
    start = args.start*args.bin_size
    bin_size = args.bin_size

    with YoutubeDL(opts) as ydl:
        for video_id in tqdm(video_list[start: start + bin_size]): #sample_video(video_list, args.log_dir)
            try: 
                ydl.download(url.format(video_id))
                with open(os.path.join(args.log_dir, '{}.json'.format(video_id)), 'w') as f: f.write(json.dumps({'user': getpass.getuser()}))
            except: 
                logging.warning('Video {} failed.'.format(video_id))
            
            time.sleep(random.random())
            disk_free = shutil.disk_usage('/').free / (1024*1024*1024)
            if disk_free < 5:
                sys.exit('Disk space is low. Free space and re-run this script.')
        