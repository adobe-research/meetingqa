import json
import argparse
from tqdm import tqdm
import os

source_path = 'YouTube/processed'
dest_path = "ProcessedTranscripts/YouTube"
if not os.path.exists(dest_path): os.makedirs(dest_path)

files = os.listdir(source_path)
files.sort()

for file in tqdm(files):
    data = json.load(open(os.path.join(source_path, file)))
    data = data['speakerAssigned']
    stored_data = {'sentences':[], 'displayText':''}
    speakers = list(set([d['speaker'] for d in data]))
    for entry in data:
        stored_data['sentences'].append({'displayText': entry['text'], 'speakerFaceId': speakers.index(entry['speaker'])})
    stored_data['displayText'] = " ".join([s['displayText'] for s in stored_data['sentences']])

    f = open(os.path.join(dest_path, file), "w+")
    json.dump(stored_data, f, indent = 4)
    

