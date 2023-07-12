
# This is an example code that shows how to call the disfluency detector.
import os
import re
from tqdm import tqdm
import time
import json
import random
import pdb
import string
from disfluency_detection import run_parse, process_sentences

def restore_clitics(line):
    line = re.sub("\s\s+", " ", line)
    line = line.replace(" 're", "'re")
    line = line.replace(" 've", "'ve")
    line = line.replace(" n't", "n't")
    line = line.replace(" 'll", "'ll")
    line = line.replace(" 'd", "'d")
    line = line.replace(" 'm ", "'m ")
    line = line.replace(" 's", "'s")
    line = line.replace("i 'm", "i'm")
    return line

def remove_space(line):
    line = line.replace(" !", "!")
    line = line.replace(" .", ".")
    line = line.replace(" ?", "?")
    line = line.replace(" ,", ",")
    line = line.replace(" :", ":")
    line = line.replace(" ;", ";")
    line = line.replace(" 'll ", "'ll ")
    line = line.replace(" 've ", "'ve ")
    line = line.replace(" 's", "'s")
    line = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', line)
    line = line.replace(' " ', ' ')
    line = line.replace('""', '')
    line = line.replace(' "?', '"?')
    line = line.replace(' ".', '".')
    line = line.replace(' ",', ',')
    line = line.lstrip(string.punctuation + " ")
    return line


def remove_disfluency(line):
    content = line.split()
    words = content[::2]
    labels = content[1::2]
    assert len(words) == len(labels)
    removed = " ".join([words[w] for w in range(len(words)) if labels[w] != 'E'])
    return restore_clitics(removed)

def obtain_mask(orig, removed):
    words = orig.split()
    left = removed.split()
    idx, lidx = 0, 0
    mask = []
    while idx < len(words):
        if lidx < len(left) and left[lidx] == words[idx]: 
            mask.append(True)
            idx += 1
            lidx += 1
        else:
            mask.append(False)
            idx += 1
    assert len(mask) == len(words)
    return mask

def apply_mask(orig, mask):
    words = orig.split()
    words = [words[w] for w in range(len(words)) if mask[w]]
    line = " ".join(words)
    if line == '': return line
    if (line[-1] != "." and line[-1] != '?'): line = line + orig[-1]
    line = line.replace(',.', '.'); line = line.replace(',?', '?'); line = line.replace(',,', ',')
    if line[0].islower(): line = line[0].upper() + line[1:]
    return line


model_path = 'model'
data_path = '/home/arprasad/ProcessedTranscripts/' #'/home/arprasad/YouTube/logs' 

fnames = os.listdir(data_path + '/AMI') #/sentences
fnames = [f for f in fnames if f not in os.listdir(data_path + '/DisfluencyRemoved/AMI')]
# fnames = random.sample(fnames, 10)
# sentence_dict = {}
lengths = []
all_orig_sentences = []
all_processed_sentences = []
delete_log = []
# INPUT_FILE_NAME = 'raw_sentences.txt'
for fname in fnames:
    delete = []
    INPUT_FILE_NAME = fname
    # input_path = os.path.join(data_path + '/sentences', INPUT_FILE_NAME)
    # with open(input_path) as input_file:
    #     sentences = input_file.readlines()
    input_path = os.path.join(data_path + '/AMI', INPUT_FILE_NAME)
    with open(input_path) as input_file:
        data = json.load(input_file)
        sentences = [s['displayText'].replace("_", "") for s in data['sentences']]
        # sentences = input_file.readlines()
    # pre-process sentences: remove punctuation, lowercase the words, etc.

    
    sentences_processed = process_sentences(sentences)
    for s, sent in enumerate(sentences_processed): 
        if not len(sent): 
           delete.append(s)
    sentences = [sentences[s] for s in range(len(sentences)) if s not in delete]
    sentences_processed = [sentences_processed[s] for s in range(len(sentences_processed)) if s not in delete]
    all_orig_sentences.append(sentences)
    delete_log.append(delete)
    all_processed_sentences.extend(sentences_processed)
    lengths.append(len(sentences_processed))

    

    # Run the disfluency detector
start_time = time.time()
all_parse_trees, all_df_labels = run_parse(model_path, all_processed_sentences)
print( '----- time taken: {} -----'.format(time.time() - start_time))
start = 0

all_final_sentences = []
for f, fname in enumerate(fnames):
    input_path = os.path.join(data_path + '/AMI', fname)
    data = json.load(open(input_path))
    final_sentences = []
    DISFLUENCIES_FILE_NAME = fname #'disfluencies.txt' 
    # disfluencies_path = os.path.join(data_path + '/disfluency', DISFLUENCIES_FILE_NAME)
    disfluencies_path = os.path.join(data_path + '/DisfluencyRemoved/AMI', DISFLUENCIES_FILE_NAME)
    processed = [restore_clitics(p) for p in all_processed_sentences[start:start + lengths[f]]]
    original = [remove_space(o) for o in all_orig_sentences[f]]

    assert len(processed) == len(original)
    for l, line in enumerate(processed):
        assert len(line.split()) == len(original[l].split()), pdb.set_trace()

    df_labels = all_df_labels[start:start + lengths[f]]
    start += lengths[f]
    for l in range(lengths[f]):
        disfluency_removed = remove_disfluency(df_labels[l])
        line = apply_mask(original[l], obtain_mask(processed[l], disfluency_removed))
        if len(line):
            final_sentences.append(line)
        else:
            delete_log[f].append(l)

    if len(data['sentences']) == len(final_sentences):
        for s, sdict in enumerate(data['sentences']):
            sdict['displayText'] = final_sentences[s]
        data['displayText'] = " ".join(final_sentences)
    elif len(delete_log[f]): 
        print(delete_log[f])
        print(len(data['sentences']))
        data['sentences'] = [data['sentences'][d] for d in range(len(data['sentences'])) if d not in delete_log[f]]
        print(len(data['sentences']))
        assert len(data['sentences']) == len(final_sentences), pdb.set_trace()
        for s, sdict in enumerate(data['sentences']):
            sdict['displayText'] = final_sentences[s]
        data['displayText'] = " ".join(final_sentences)
    else:
        print("Error in building delete logs, nothing registered but lengths don't match")

    
    out = open(disfluencies_path, 'w+')
    json.dump(data, out, indent=4)
    



        
    
    # with open(disfluencies_path, 'w+') as result_file:
    #     for line in df_labels:
    #         result_file.write(remove_disfluency(line))
    #         result_file.write('\n')
    
    # print("Disfluencies written to:", disfluencies_path)
