import os
import json
import argparse
from tqdm import tqdm

trans_path = 'YouTube/timedTranscripts'
diar_path = '/home/arprasad/sensei-fs-symlink/users/arprasad/YouTube/speakers'
dest_path = 'YouTube/cache'
if not os.path.exists(dest_path): os.makedirs(dest_path)

files = os.listdir(diar_path)
files.sort()

for fname in tqdm(files):
    try:
        diarization = json.load(open(os.path.join(diar_path, fname)))
        data = json.load(open(os.path.join(trans_path, fname)))
        assert len(diarization)
    except: print(fname); continue
    utterances = data['utterances']
    for u, utterance in enumerate(utterances):
        speaker = None
        if u == 0: start = -1
        else: start = utterance['startMs']/1000
        end = utterance['startMs']/1000 + utterance['durationMs']/1000
        for segments in diarization:
            # here alignment based only on end times
            if  end <= segments[2]: #end > segments[1] and
                speaker = segments[0]
                break
        try:
            if speaker == None and end >= diarization[-1][2]: speaker = diarization[-1][0] # for the ends that are not captured
        except: import pdb; pdb.set_trace()
        data['utterances'][u]['speaker'] = speaker
        # print(speaker, start, end, utterance['text'])
    f = open(os.path.join(dest_path, fname), "w+")
    json.dump(data,f, indent=4)
    f.close()
        

           
            
            