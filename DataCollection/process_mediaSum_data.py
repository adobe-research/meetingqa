# -*- coding: utf-8 -*-

import json
import os
from tqdm import tqdm
import spacy
from spacy.language import Language
import argparse

def fix_quotes(text):
    text = text.replace('."', '".')
    text = text.replace('?"', '"?')
    text = text.replace('!"', '"!')
    text = text.replace(',"', '",')
    return text

# This function adds costum rules to spaCy sentence segmentation.
@Language.component('custom_boundaries')
def set_custom_boundaries(doc):
    for i, token in enumerate(doc):
        # Do not let spaCy to break a sentence at a comma
        if token.text.endswith(',') and i < len(doc) - 1:
            doc[i + 1].is_sent_start = False
            doc[i].is_sent_start = False
        if token.text[-1] not in {'.', '!', '?'} and i < len(doc) - 1:
            doc[i + 1].is_sent_start = False
    return doc

nlp = spacy.load("en_core_web_lg")
# Custom rules are added to spaCy
nlp.add_pipe("custom_boundaries", before="parser")

dest_path = "/home/arprasad/ProcessedTranscripts/MediaSumInterviews"
source_path = "/home/arprasad/Interviews/news_dialogue.json"

interviews = json.load(open(source_path))

parser = argparse.ArgumentParser(description='Take arguments from commandline')
parser.add_argument('--start', default=0, type=int)
parser.add_argument('--bin-size', default=100000, type=int)
args = parser.parse_args()
print('Started ', args.start)
bin_size = args.bin_size
start = args.start*bin_size

for interview in tqdm(interviews[start : min(start + bin_size, len(interviews))]):
    ijson = {'sentences':[]}
    iname = interview['id'] + '.json'
    assert len(interview['utt']) == len(interview['speaker'])
    speakers = list(set(interview['speaker']))
    for s, segment in enumerate(interview['utt']):
        # speakerID = speakers.index(interview['speaker'][s])
        speakerID = interview['speaker'][s]
        utterances = []
        segment = fix_quotes(segment)
        doc = nlp(segment)
        for sent in doc.sents: 
            sentence_dict = {'displayText': sent.text, 'speakerFaceId':speakerID}
            ijson['sentences'].append(sentence_dict)

    ijson['displayText'] = " ".join([sentence['displayText']  for sentence in ijson['sentences']])
    iname = os.path.join(dest_path, iname)
    with open(iname, "w+") as j:
        json.dump(ijson, j, indent=4)
    

        
        







