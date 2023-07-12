# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
import json
import numpy as np
import argparse
import random

def get_num_words(text):
    return len(text.split(" "))

def check_if_question(text):
    if text[-1] == "?": 
        if len(text.split(" ")) > 2: return True #Orig: 2
        # else: print(text)
    return False

def compute_stat(counter):
    counter = np.array(counter)
    return round(counter.mean(), 1)

parser = argparse.ArgumentParser(description='Take arguments from commandline')
parser.add_argument('--source-path', default='/home/arprasad/ProcessedTranscripts/DisfluencyRemoved', help='Path to where processed transcripts lie')
parser.add_argument('--dataset', default='AMI', help='Name of dataset')
parser.add_argument('--destination-path', default='/home/arprasad/ProcessedTranscripts/AnnotationProposal/FinalAMI', help='Path to store data for annotation')
args = parser.parse_args()

source_path = os.path.join(args.source_path, args.dataset)
destination_path = os.path.join(args.destination_path, args.dataset)
if not os.path.exists(destination_path): os.makedirs(destination_path)
print(destination_path)

words_counter = []
sentences_counter = []
questions_counter = [] 
adjusted_questions_counter = []

meetings = os.listdir(source_path)
meetings.sort()
remove_meetings = [] # Was used to create multiple batches for human annotation, remove files from previous batch(es).
# remove_meetings = ['IS1006b.json', 'ES2006d.json', 'ES2009c.json', 'IS1009a.json', 'ES2003c.json']
# remove_meetings.extend(os.listdir('/home/arprasad/ProcessedTranscripts/CalibrationBatchJson/AMI'))
print(len(remove_meetings))
meetings = [m for m in meetings if not m in remove_meetings]
meetings_subsample = meetings
print('Selected ', len(meetings_subsample), ' meetings.')
# meetings_subsample = ['IS1006b.json', 'ES2006d.json', 'ES2009c.json', 'IS1009a.json', 'ES2003c.json']
# print('Selected Meeting IDs: ', meetings_subsample)
# import pdb; pdb.set_trace()
for meeting in meetings_subsample:
    num_words = 0
    num_questions = 0
    num_sentences = 0
    seq_questions = []
    local_seq_len = 1
    last_was_question = False
    data = json.load(open(os.path.join(source_path, meeting)))
    # new_data = {'displayText':data['displayText']}
    f = open(os.path.join(destination_path, meeting), 'w+')
    sentences = data['sentences']
    for sid, sentence_dict in enumerate(sentences):
        text = sentence_dict['displayText']
        sentence_dict['sentenceId'] = sid
        sentence_dict['question'] = {'possible': False, 'meaningful':False, 'questionContext':[], 'combinedQuestion':[], 'answerSpan':[], 'unanswerableReason':'', 'confidentAnswer': True, 'externalReview': False}
        if not len(text): continue
        num_words += get_num_words(text)
        num_sentences += 1   
        is_question = check_if_question(sentence_dict['displayText'])
        if is_question: 
            sentence_dict['question']['possible'] = True
            if last_was_question:
                local_seq_len += 1
            last_was_question = True  
        else: 
            if last_was_question: 
                seq_questions.append(local_seq_len)
                local_seq_len = 1
            last_was_question = False
        num_questions += int(is_question)  
        data['sentences'][sid] = sentence_dict
    words_counter.append(num_words)
    sentences_counter.append(num_sentences)
    questions_counter.append(num_questions)
    adjusted_questions_counter.append(num_questions - sum(seq_questions) + len(seq_questions))
    
    json.dump(data, f, indent=4)
    f.close()
    
    


print('Displaying Statistics for {} Dataset (per meeting)....'.format(args.dataset))
print('Total Num Files: \t', str(len(meetings)))
print('Total Sentences: \t', compute_stat(sentences_counter))
print('Total Words: \t', compute_stat(words_counter))
print('Num Identified Questions: \t', compute_stat(questions_counter))
print('% Questions: \t', 100*compute_stat(questions_counter)/compute_stat(sentences_counter))
print('Num Identified Questions (collapsing sequential): \t', compute_stat(adjusted_questions_counter))
print('% Questions (collapsing sequential): \t', 100*compute_stat(adjusted_questions_counter)/compute_stat(sentences_counter))