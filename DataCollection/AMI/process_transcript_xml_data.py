# -*- coding: utf-8 -*-

import codecs
from xml.dom import minidom
import os
from tqdm import tqdm
import json
import numpy as np

def get_file_meta(words_path):
    all_files = os.listdir(words_path)
    file_meta = {}
    filenames = list(set([f.split('.')[0] for f in all_files]))
    filenames.sort()
    for fname in filenames: 
        fspeakers = [f.split('.')[1] for f in all_files if fname in f]
        fspeakers.sort()
        file_meta[fname] = fspeakers
    return file_meta

def get_id(name):
    name = name.replace(',', '')
    return int(name.split('.')[-1])

def get_speakerId(speaker, speakers):
    return speakers.index(speaker)

def get_start_time(elem):
    return elem[2]

def get_utterance_for_onepeople(people, path, icsi = False):
    """
    get utterances for one person {ES2002a.A.words.xml}
    :param people: people
    :param paths: words file path
    :return:
    """
    utterances = []  # sequence of tuples

    # parse xml doc
    word_doc = minidom.parse(path)

    # init
    utterance_word_ids = []
    utterance = ""
    start_time = 0
    words = word_doc.getElementsByTagName('w')

    for word in words:
        if word.hasAttribute("starttime"):
            text = word.firstChild.data
            # if end with '.' or '?'. it will be an utterance
            seg = True if word.hasAttribute(
                'punc') and (text == "." or text == "?") else False
            if icsi: seg = True if word.getAttribute('c') == '.' and (text == "." or text == "?") else False
            if seg:
                utterance = utterance[:-1] + text
                utterances.append((people, utterance, start_time))
                # reset
                utterance = ""
                utterance_word_ids = []
                start_time = 0
            else:
                
                if not word.hasAttribute('trunc') and not (icsi and word.getAttribute('c') == 'TRUNCW') and not (icsi and text == "@"): 
                    if word.getAttribute('punc') or (icsi and (word.getAttribute('c') == 'CM' or word.getAttribute('c') == 'APOSS')): utterance = utterance[:-1] + text
                    elif icsi and (text[0] == "'"): utterance = utterance[:-1] + text
                    elif icsi and (word.getAttribute('c') == 'HYPH'): utterance = utterance[:-1] + text
                    else: utterance += text
                    if not (icsi and word.getAttribute('c') == 'HYPH'): utterance += " "
                    utterance_word_ids.append(word.getAttribute("nite:id"))
                    
                # when an new utterance start
                if start_time == 0:
                    try: start_time = float(word.getAttribute("starttime"))
                    except: 
                        if icsi:
                            id = get_id(word.getAttribute("nite:id"))
                            seg_path = path.replace('Words', 'Segments')
                            seg_path = seg_path.replace('words', 'segs')
                            seg_doc = minidom.parse(seg_path)
                            segments = seg_doc.getElementsByTagName('segment')
                            for segment in segments:
                                seg_start = float(segment.getAttribute('starttime'))
                                try: seg_end = float(segment.getAttribute('endtime'))
                                except: seg_end = seg_start
                                ref =  segment.getElementsByTagName('nite:child')[0].getAttribute('href')
                                if 'words' in ref:
                                    splits = ref.split('id(')
                                    if len(splits) < 3: continue
                                    id_start = splits[1].split(')')[0]
                                    id_end = splits[2].split(')')[0]
                                    if '.w.' not in id_start or '.w.' not in id_end: continue
                                    id_start = get_id(id_start)
                                    id_end = get_id(id_end)
                                if id >= id_start and id < id_end:
                                    start_time = seg_start + (id - id_start)*(seg_end - seg_start)/(id_end - id_start)

        else:
            # some error datas
            print("word {} does not have start time, pass it!".format(word.getAttribute("nite:id")))
    
        
    return utterances

def get_num_words(text):
    return len(text.split(" "))

def check_if_question(text):
    if text[-1] == "?": return True
    return False

def compute_stat(counter):
    counter = np.array(counter)
    return round(counter.mean(), 1)


words_path = '/home/arprasad/icsi/ICSIplus/Words' # Options: '~/icsi/ICSIplus/Words' '~/ami/ami_public_manual_1.6.2/words'
dest_path = '/home/arprasad/ProcessedTranscripts/ICSI' # Change based on options
if "icsi" in words_path: icsi = True
else: icsi = False

# Could be a different file too for generating statistics, but need annotations for questions here.
words_counter = []
sentences_counter = []
questions_counter = [] 
adjusted_questions_counter = []

file_meta = get_file_meta(words_path)
for fname in tqdm(file_meta.keys()):
    num_words = 0
    num_questions = 0
    num_sentences = 0
    seq_questions = []
    local_seq_len = 1
    last_was_question = False
    meeting_utterances = []
    meeting_json = {}
    speakers = file_meta[fname]
    for speak in speakers:
        speak_utterances = get_utterance_for_onepeople(speak, "{}/{}.{}.words.xml".format(words_path, fname, speak), icsi = icsi)
        meeting_utterances.extend(speak_utterances)

    meeting_utterances.sort(key=get_start_time)
    meeting_utterances = [m for m in meeting_utterances if m[-1] > 0]
    entire_transcript = " ".join([m[1] for m in meeting_utterances])
    meeting_json['displayText'] = entire_transcript
    meeting_json['sentences'] = []
    for utterance in meeting_utterances:
        sentence_dict = {'displayText': utterance[1], 'speakerFaceId': get_speakerId(utterance[0], speakers)}
        meeting_json['sentences'].append(sentence_dict) 
        num_words += get_num_words(sentence_dict['displayText'])
        num_sentences += 1   
        is_question = check_if_question(sentence_dict['displayText'])
        if is_question: 
            if last_was_question:
                local_seq_len += 1
            last_was_question = True  
        else: 
            if last_was_question: 
                seq_questions.append(local_seq_len)
                local_seq_len = 1
            last_was_question = False

        num_questions += int(is_question)    
    words_counter.append(num_words)
    sentences_counter.append(num_sentences)
    questions_counter.append(num_questions)
    adjusted_questions_counter.append(num_questions - sum(seq_questions) + len(seq_questions))

    jname = os.path.join(dest_path, fname + '.json')
    with open(jname, "w+") as j:
        json.dump(meeting_json, j, indent = 4)
print("\n")
print('Displaying Statistics (per meeting)....')
print('Total Sentences: \t', compute_stat(sentences_counter))
print('Total Words: \t', compute_stat(words_counter))
print('Num Identified Questions: \t', compute_stat(questions_counter))
print('% Questions: \t', 100*compute_stat(questions_counter)/compute_stat(sentences_counter))
print('Num Identified Questions (collapsing sequential): \t', compute_stat(adjusted_questions_counter))
    

