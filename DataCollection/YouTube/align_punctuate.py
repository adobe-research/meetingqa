import os
import json
# from fuzzyset import FuzzySet
from rpunct import RestorePuncts
import spacy
from spacy.language import Language
import re
import time
from tqdm import tqdm
import argparse

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

def repunctuate(text):
    rpunct = RestorePuncts()
    text = re.sub("[\[].*?[\]] ", "", text) 
    punct_text = rpunct.punctuate(text)
    return punct_text
    # doc = nlp(punct_text)
    # sentences = []
    # for sent in doc.sents:
    #     sentences.append(sent.text)


repunct_dir = 'YouTube/logs/sentences'
diar_dir = 'YouTube/cache'
dest_path = 'YouTube/processed'
if not os.path.exists(dest_path): os.makedirs(dest_path)

files = os.listdir(diar_dir)
files.sort()

rpunct = RestorePuncts()

# import pdb; pdb.set_trace()
def run(start, bin_size):
    start = start*bin_size
    print('Started for {}'.format(start))
    already_there = [t for t in os.listdir(dest_path)]
    for file in files[start : start + bin_size]:#files[start : start + bin_size]: #[start: start + bin_size]
        print("Begin ", file)
        if file in already_there: continue
    # for file in tqdm(files[:100]):
        try:
            stime = time.time()
            passage = ''
            lines = open(os.path.join(repunct_dir, file.split('.')[0] + '.txt')).readlines()
            lines = [line.strip() for line in lines]
            passage = " ".join(lines)

            data = json.load(open(os.path.join(diar_dir, file)))
            data['speakerAssigned'] = []
            utterances = data['utterances']

            speaker_combined = []
            prev_speaker = ''
            current_speaker = ''
            for utterance in utterances:
                text = utterance['text']
                current_speaker = utterance['speaker']
                if current_speaker != prev_speaker: 
                    speaker_combined.append([current_speaker, text])
                else:
                    # try:
                        # print(all_text)
                    all_text = speaker_combined[-1][1]
                    all_text = " ".join([all_text, text])
                    # except: import pdb; pdb.set_trace()
                    speaker_combined[-1][1] = all_text
                prev_speaker = current_speaker

            for i, inst in enumerate(speaker_combined):
                speaker, text = inst
                text = re.sub("[\[].*?[\]]", "", text) 
                if text == '': continue
                punct_text = rpunct.punctuate(text, lang='en')
                doc = nlp(punct_text)
                for sent in doc.sents:
                    data['speakerAssigned'].append({'text':sent.text,'speaker':speaker})
            # print('### Time Taken: \t', time.time() - stime, ' ###')
            print('End ', file)
            f = open(os.path.join(dest_path, file), "w+")
            json.dump(data, f, indent = 4)
        except: continue

    print('Ended for {}'.format(start))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take arguments from commandline')
    parser.add_argument('--start', default=0)
    parser.add_argument('--bin-size', default=100)
    args = parser.parse_args()

    run(int(args.start), int(args.bin_size))   