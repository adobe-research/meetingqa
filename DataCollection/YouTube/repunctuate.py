import os
import json
from rpunct import RestorePuncts
import torch
import spacy
from spacy.language import Language
import time
import re
import argparse
from tqdm import tqdm
from multiprocessing import Process,Lock


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



source_dir = '/home/arprasad/YouTube/transcripts'
fnames = [f.split('.')[0] for f in os.listdir(source_dir)]
fnames.sort()

# bin_size = args.bin_size

def repunctuate(start, bin_size):
    print('Started for {}'.format(start))
    rpunct = RestorePuncts()
    start = start*bin_size
    already_there = [t.split('.')[0] for t in os.listdir('/home/arprasad/YouTube/logs/sentences/')]
    for fname in fnames[start : start + bin_size]: #[start: start + bin_size]
        print("Begin ", fname)
        data = json.load(open(os.path.join(source_dir, fname + '.json')))
        if data['punctuated'] or fname in already_there: 
            print('End ', fname)
            continue
        text = data['text']
        text = re.sub("[\[].*?[\]] ", "", text)
        try:    
            punct_text = rpunct.punctuate(text)
            doc = nlp(punct_text)
            sentences = []
            for sent in doc.sents:
                sentences.append(sent.text)
            dest_path = '/home/arprasad/YouTube/logs/sentences/{}.txt'.format(fname)
            f = open(dest_path, "w+")
            for s in sentences:
                f.write(s + '\n')
            f.close()
            print('End ', fname)
        except: 
            print('End ', fname)
            continue
    print('Ended for {}'.format(start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take arguments from commandline')
    parser.add_argument('--start', default=0)
    parser.add_argument('--bin-size', default=1000)
    args = parser.parse_args()

    repunctuate(int(args.start), int(args.bin_size))

    # processes = [Process(target=thread_task, args=(i,)) for i in range(0, len(fnames), bin_size)]
    # for process in processes:
    #     process.start()
    # for process in processes:
    #     process.join()