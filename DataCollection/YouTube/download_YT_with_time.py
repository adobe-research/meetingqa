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

import traceback

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
    p.add_argument('--output-dir', default=os.path.join('/home/arprasad/YouTube', 'videos'),
                   required=False, help='Output directory where videos will be stored.')
    p.add_argument('--log-dir', default=os.path.join('/home/arprasad/YouTube', 'logs'),
                   required=False, help='Log dir to keep track of downloaded videos so far.')

    args = p.parse_args()
    time.sleep(random.random()) # lazy way to prevent race condition 
    if not os.path.isdir(args.output_dir): os.makedirs(args.output_dir)
    time.sleep(random.random()) # lazy way to prevent race condition
    if not os.path.isdir(args.log_dir): os.makedirs(args.log_dir)
    opts = {'format': '18/22', 'outtmpl': '{}/%(id)s.%(ext)s'.format(args.output_dir), 'logger': YTLogger(), 'writesubtitles': True,
'subtitle': '--write-sub --sub-lang en', 'skip-download': True}
    url = 'https://www.youtube.com/watch?v={}'
    with open(args.input_file, 'r') as f: video_list = json.load(f) 
    transcripts_dir = '/home/arprasad/YouTube/timedTranscripts'

    with YoutubeDL(opts) as ydl:
        for video_id in tqdm(video_list): #sample_video(video_list, args.log_dir)
            try: 
                info_dict = ydl.extract_info(url.format(video_id), download=False) 
                channel_id = info_dict['channel_id']   
                video_title = info_dict['title']
                duration = info_dict['duration_string']
                cc_license = False
                try: 
                    if info_dict['license'] == 'Creative Commons Attribution license (reuse allowed)': cc_license = True
                except: pass
                punctuated = False
                # utterances = []
                try:
                # if 'subtitles' in info_dict.keys() and info_dict['subtitles']:
                    caption_url = ''
                    captions_info = info_dict['subtitles'] 
                    for key, entry in captions_info.items():
                        if 'en' in key:
                            punctuated = True
                            for t in entry:
                                if t['ext'] == 'json3': 
                                    caption_url = t['url']
                                    language = t['name'].split()[0].lower()
                                    break
                            
                    captions_dict = eval(urlopen(caption_url).read())
                    utterances = []
                    for line in captions_dict['events']:
                        startMs = line['tStartMs']
                        durationMs = line['dDurationMs']
                        utt_dict = {'startMs': startMs, 'durationMs': durationMs}
                        for utterance in line['segs']:
                            utt_dict['text'] = utterance['utf8']
                        utterances.append(utt_dict)
                    text = " ".join(utterance['text'] for utterance in utterances)
                
                # elif 'en' in info_dict['automatic_captions'].keys():
                except:
                    captions_list = info_dict['automatic_captions']['en']
                    for t in captions_list:
                        if t['ext'] == 'json3': 
                            caption_url = t['url']
                            language = t['name'].split()[0].lower()
                            break
                    
                    captions_dict = eval(urlopen(caption_url).read())
                    utterances = []
                    for line in captions_dict['events']:
                        if 'segs' in line.keys():
                            segment = []
                            for utterance in line['segs']:
                                if utterance['utf8'] != '\n': segment.append(utterance['utf8'].strip(" "))
                            segment = ' '.join(segment)
                            if len(segment): 
                                utt_dict = {'startMs': line['tStartMs'], 'durationMs': line['dDurationMs'], 'text': segment}
                                utterances.append(utt_dict)
                    
                    text = " ".join(utterance['text'] for utterance in utterances)
                    

                # else: print('Some problem with transcripts for ', video_id)

                transcript_dict = {'channelID': channel_id, 'title': video_title, 'duration': duration, 'language': language, 'ccLicense': cc_license, 'punctuated': punctuated,'utterances':utterances, 'text': text}
                f = open(os.path.join(transcripts_dir, video_id + '.json'), "w+")
                json.dump(transcript_dict, f, indent=4)    
                f.close()         
            
                # ydl.download(url.format(video_id))
                # with open(os.path.join(args.log_dir, '{}.json'.format(video_id)), 'w') as f: f.write(json.dumps({'user': getpass.getuser()}))
            except Exception: 
                # traceback.print_exc()
                logging.warning('Video {} failed.'.format(video_id))
            
            time.sleep(random.random())
            disk_free = shutil.disk_usage('/').free / (1024*1024*1024)
            if disk_free < 5:
                sys.exit('Disk space is low. Free space and re-run this script.')
        