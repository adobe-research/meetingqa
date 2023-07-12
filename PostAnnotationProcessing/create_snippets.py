# -*- coding: utf-8 -*-
import argparse
import os
import json
from tqdm import tqdm
from generate_synthetic_qa import club_speakers, find_paragraph
from itertools import groupby
import random
import numpy as np
import pdb

def pretty_name(name):
    """
    Clean the speaker names
    """
    name = name.split(',')[0]
    return name.title()

def avg_club_length(clubs):
    lengths = []
    for club in clubs:
        lengths.append(club['end'] - club['start'] + 1)
    return np.mean(lengths)

def split_clubs(clubs, sentences):
    """
    For multi-speaker data-augmentation in which we force more speaker turns. Takes in original utterances
    and splits them such that there are more speaker turns and each utterance is smaller.
    """
    revised_clubs = []
    speakers = sorted(list(set(pretty_name(club['speaker']) for club in clubs)))
    def get_subclubs(c_sentences, c_start, c_end, c_speaker):
        c_len = c_end - c_start + 1
        sample = np.random.choice([1,2], c_len) # instead of random max number, build custom for each club
        run_sum, start = 0, 0
        sub_club = []
        for s, size in enumerate(sample):
            if c_len - run_sum > 1:
                run_sum += size
                sclub = {'displayText': ' '.join([sent['displayText'] for sent in c_sentences[start: start + size]])}
                sclub['start'] = c_start + start
                sclub['end'] = sclub['start'] + size - 1
                if s%2 == 0: sclub['speaker'] = c_speaker
                else: sclub['speaker'] = np.random.choice([speak for speak in speakers if speak != c_speaker])
                sub_club.append(sclub)
                start += size
            elif c_len - run_sum == 1:
                run_sum += 1
                sclub = {'displayText': ' '.join([sent['displayText'] for sent in c_sentences[start: start + 1]])}
                sclub['start'] = c_start + start
                sclub['end'] = sclub['start']
                if s%2 == 0: sclub['speaker'] = c_speaker
                else: sclub['speaker'] = np.random.choice([speak for speak in speakers if speak != c_speaker])
                sub_club.append(sclub)
                start += size
            elif c_len == run_sum:
                break
        return sub_club
        
    for club in clubs:
        c_speaker = club['speaker']
        c_sentences = sentences[club['start']: min(club['end'] + 1, len(sentences))]
        split_club = get_subclubs(c_sentences, club['start'], club['end'], c_speaker)
        revised_clubs.extend(split_club)
    return revised_clubs

def find_prefix_paragraph(pid, already, threshold, clubs):
    count = already
    start = None
    if pid > 0:
        for c, club in enumerate(clubs[pid - 1::-1]):
            count += len(club['displayText'].split())
            if count >= threshold: break
        start = pid - 1 -c
        
    return start

def find_suffix_paragraph(pid, already, threshold, clubs):
    count = already
    end = None
    if pid < len(clubs) - 1:
        for c, club in enumerate(clubs[pid + 1:]):
            count += len(club['displayText'].split())
            if count >= threshold: break
        end = c + pid + 1
        
    return end

def create_no_speaker_context(sentences, clubs, pid, prefix_pid, suffix_pid):
    """
    Context after removing speaker information and turns.
    """
    paragraph = ''
    for club in clubs[prefix_pid : min(suffix_pid + 1, len(clubs))]:
        start = club['start']
        end = club['end']
        paragraph += '\n'.join([sentences[i]['displayText'] for i in range(start, end + 1)]) + '\n'
    return paragraph

def create_context(clubs, pid, prefix_pid, suffix_pid, separate_qa = False, separate_speaker = False, mask_identity = False):
    """
    Create passage for QA based on location of question, prefix and suffix budge. Apply different data augmentations appropriately.
    """
    # assume both separate attributes are never true together
    paragraph = ''
    speakers = sorted(list(set(pretty_name(club['speaker']) for club in clubs)))
    if not separate_qa and not separate_speaker:
        for club in clubs[prefix_pid : min(suffix_pid + 1, len(clubs))]:
            sname = pretty_name(club['speaker'])
            if mask_identity: sname = 'Speaker ' + str(speakers.index(sname))
            if paragraph: paragraph += " " + sname + ': ' + club['displayText'] + "\n"
            else: paragraph += sname + ': ' + club['displayText'] + "\n"

    elif separate_qa:
        if prefix_pid > 0 or suffix_pid + 1 < len(clubs):
            remaining_clubs = clubs[:prefix_pid]
            remaining_clubs.extend(clubs[suffix_pid +1:])

            

            for club in clubs[prefix_pid : pid + 1]:
                sname = pretty_name(club['speaker'])
                if mask_identity: sname = 'Speaker ' + str(speakers.index(sname))
                if paragraph: paragraph += " " + sname + ': ' + club['displayText'] + "\n"
                else: paragraph += sname + ': ' + club['displayText'] + "\n"

            num_clubs = random.sample(list(range(min(5, len(remaining_clubs)))), 1)[0]
            dummy_clubs = random.sample(remaining_clubs, num_clubs)

            
            for club in dummy_clubs:
                sname = pretty_name(club['speaker'])
                if mask_identity: sname = 'Speaker ' + str(speakers.index(sname))
                if paragraph: paragraph += " " + sname + ': ' + club['displayText'] + "\n"
                else: paragraph += sname + ': ' + club['displayText'] + "\n"
                
            for club in clubs[pid + 1: min(suffix_pid + 1, len(clubs))]:
                sname = pretty_name(club['speaker'])
                if mask_identity: sname = 'Speaker ' + str(speakers.index(sname))
                if paragraph: paragraph += " " + sname + ': ' + club['displayText'] + "\n"
                else: paragraph += sname + ': ' + club['displayText'] + "\n"

    elif separate_speaker:
        remaining_sentences = []
        if prefix_pid > 0:
            remaining_sentences.extend(sentences[:clubs[prefix_pid - 1]['end'] + 1])
        if suffix_pid + 1 < len(clubs):
            remaining_sentences.extend(sentences[clubs[suffix_pid + 1]['start']:])

        decision = np.random.random() < 0.5

        if len(remaining_sentences) >= 1:
            if decision or len(remaining_sentences) == 1: dummy_sentences = random.sample(remaining_sentences, 1)
                # add one   
            else: dummy_sentences = random.sample(remaining_sentences, 2)

            for club in clubs[prefix_pid : pid + 1]:
                sname = pretty_name(club['speaker'])
                if mask_identity: sname = 'Speaker ' + str(speakers.index(sname))
                if paragraph: paragraph += " " + sname + ': ' + club['displayText'] + "\n"
                else: paragraph += sname + ': ' + club['displayText'] + "\n"
            
            for c, club in enumerate(clubs[pid + 1: min(suffix_pid + 1, len(clubs))]):
                sname = pretty_name(club['speaker'])
                if mask_identity: sname = 'Speaker ' + str(speakers.index(sname))
                random_phrase = " ".join([d['displayText'] for d in dummy_sentences])
                new_text = " ".join([random_phrase, club['displayText']])
                if c == 0:
                    if paragraph: paragraph += " " + sname + ': ' + new_text + "\n"
                    else: paragraph += sname + ': ' + new_text + "\n"
                else:
                    if paragraph: paragraph += " " + sname + ': ' + club['displayText'] + "\n"
                    else: paragraph += sname + ': ' + club['displayText'] + "\n"
  
    return paragraph

# def update_context(alt_dict, sentences, context):
    
#     new_context = ''
#     last_sentence = sentences[alt_dict['ids_1'][-1]]['displayText']
#     idx = context.find(last_sentence) + len(last_sentence)
#     new_context = context[:idx]
#     new_context += '\n ' + alt_dict['alt_name'] + ':'
#     new_context += context[idx:]
#     return new_context

    
def obtain_formatted_answers(ids, sentences, clubs, separate = False, mask_identity = False, multi_speakers = False, remove_speakers = False):
    """
    Takes in the answer spans and processes them to return answer sentences and other meta information. Also apply data-augmentation
    compatible with the ones applied in context in order ensure maximum lexical overlap with context.
    """
    # add support for is multi-span answer and also find continuous spans
    answers = []
    answer_spans = []
    pids = set()
    span_tracker = []
    is_multispan = False
    prev_id = None
    prev_id_speaker = None
    span_id = 0
    speakers = sorted(list(set(pretty_name(club['speaker']) for club in clubs)))
    alt_dict = {}
    for id in ids:
        pid = find_paragraph(id, clubs)
        pids.add(pid)
        id_speaker = clubs[pid]['speaker']
        if not prev_id is None: 
            if id - prev_id > 1 or id_speaker != prev_id_speaker:  
                is_multispan = True
                span_id +=1 
            span_tracker.append(span_id)
        else: span_tracker.append(span_id)
        try: sname = pretty_name(clubs[pid]['speaker'])
        except: import pdb; pdb.set_trace()
        if mask_identity: sname = 'Speaker ' + str(speakers.index(sname))

        if remove_speakers: 
            answers.append(sentences[id]['displayText'] + '\n')
        elif separate:
            if id == ids[0] and id == clubs[pid]['start'] and id == clubs[pid]['end']: answers.append(sentences[id]['displayText'] + '\n')
            elif id == ids[0] and id == clubs[pid]['start']: answers.append(sentences[id]['displayText'])
            elif id == clubs[pid]['start'] and id == clubs[pid]['end']: answers.append(sname + ": " + sentences[id]['displayText'] + '\n')
            elif id == clubs[pid]['start']: answers.append(sname + ": " + sentences[id]['displayText'])
            elif id == clubs[pid]['end']: answers.append(sentences[id]['displayText'] + '\n')
            else: answers.append(sentences[id]['displayText'])
        else:
            if id == clubs[pid]['start'] and id == clubs[pid]['end']: answers.append(sname + ": " + sentences[id]['displayText'] + '\n')
            elif id == clubs[pid]['start']: answers.append(sname + ": " + sentences[id]['displayText'])
            elif id == clubs[pid]['end']: answers.append(sentences[id]['displayText'] + '\n')
            else: answers.append(sentences[id]['displayText'])
        prev_id = id
        prev_id_speaker = id_speaker
    
    if len(pids) > 1: is_multispan = True

    sub_answer = answers[0]
    for i in range(1,len(span_tracker)):
        if span_tracker[i] == span_tracker[i-1]:
            sub_answer += ' ' + answers[i]
        else:
            answer_spans.append(sub_answer)
            sub_answer = answers[i]
    answer_spans.append(sub_answer)

    
    return answers, is_multispan, answer_spans, alt_dict

if __name__ == "__main__":

    random.seed(2022)
    np.random.seed(2022)

    pre_word_count = 50 
    post_word_count = 250 

    source_path = "AllData/Annotated"
    dest_path = 'AllData/Snippets'
    if not os.path.exists(dest_path): os.makedirs(dest_path)

    files = os.listdir(source_path)
    files.sort()
    

    parser = argparse.ArgumentParser(description='Take arguments from commandline')
    parser.add_argument('--answer-type', default='concat', choices=['concat', 'cont', 'multispans'], help='type of answers to feed in')
    args = parser.parse_args()

    total = 0
    answerable = 0

    choose_per_file = True #for main synthetic data = True
    answer_lengths = []
    for_test = False
    word_counts_per_file = True
    hit_max = 0
    force_unanswerable = True # For Split, force unanswerable = True
    split_utterances = False
    remove_speakers = False
    club_lengths = []

    for file in tqdm(files):
        if word_counts_per_file:
            pre_word_count = np.random.choice([25, 50, 75])
            post_word_count = np.random.choice([200, 250, 300])

        data = json.load(open(os.path.join(source_path, file)))
        clubs = club_speakers(data)
        club_lengths.append(avg_club_length(clubs))
        
        sentences = data['sentences']

        separate_qa = False # If True, adds random sentences between question and beginning of answer.
        separate_speaker_label = False # If True, adds random sentences between speaker name (label) and the actual content of answer.
        mask_identity = False # If True, replaces speaker names with Speaker #, done to simulate test time settings with AMI.
        multi_speaker = False # If True, split utterances into smaller chunks and simulates larger speaker turns.
        no_speakers = False # If True, removes all speaker and transition/turn information.
        if not for_test:
            if np.random.random() < 0.5: mask_identity = True
            if remove_speakers: 
                if np.random.random() < 0.20: no_speakers = True
            if split_utterances: 
                if np.random.random() < 0.25: multi_speaker = True
            
            if not no_speakers: 
                option = np.random.choice([0, 1, 2])
                if option == 1: separate_qa = True
                if option == 2: separate_speaker_label = True

        else: mask_identity = True
        case = '{}-{}-{}-'.format(int(separate_qa), int(separate_speaker_label), int(mask_identity))
        if split_utterances: case = '{}-{}-{}-{}-'.format(int(separate_qa), int(separate_speaker_label), int(mask_identity), int(multi_speaker))
        if no_speakers: case = 'NoSpeaker_' + case
        for sid, sentence_dict in enumerate(sentences):
            if not choose_per_file and not for_test:
                separate_qa = False
                separate_speaker_label = False
                if split_utterances: 
                    if np.random.random() < 0.25: multi_speaker = True
                
                if not no_speakers:
                    option = np.random.choice([0, 1, 2])
                    if option == 1: separate_qa = True
                    if option == 2: separate_speaker_label = True
                case = '{}-{}-{}-'.format(int(separate_qa), int(separate_speaker_label), int(mask_identity))
                if split_utterances: case = '{}-{}-{}-{}-'.format(int(separate_qa), int(separate_speaker_label), int(mask_identity), int(multi_speaker))
                if no_speakers: case = 'NoSpeaker_' + case
            question = sentence_dict['question']
            text = sentence_dict['displayText']
            
            if question['possible']:
                marker = False
                snippet = {}
                speakers = sorted(list(set(pretty_name(club['speaker']) for club in clubs)))
                if multi_speaker and len(speakers) > 1: clubs = split_clubs(clubs, sentences)
                pid = find_paragraph(sid, clubs) #change to clubs
                if pid is None: import pdb; pdb.set_trace()
                prev = ''
                idx = clubs[pid]['displayText'].find(text)
                prev = clubs[pid]['displayText'][:idx]
                after = clubs[pid]['displayText'][idx + len(text):]
                prefix_pid = find_prefix_paragraph(pid, len(prev.split()), pre_word_count, clubs)
                suffix_pid = find_suffix_paragraph(pid, len(after.split()), post_word_count, clubs)
                
                if prefix_pid is None or suffix_pid is None: hit_max += 1
                if prefix_pid == None: prefix_pid = pid
                if suffix_pid == None: suffix_pid = pid
                assert prefix_pid <= pid, 'Problem with prefix pid'
                assert suffix_pid >= pid, 'Problem with suffix pid'
                
                # create paragraphs for q/a
                if not no_speakers: context = create_context(clubs, pid, prefix_pid, suffix_pid, separate_qa, separate_speaker_label, mask_identity)
                else: context = create_no_speaker_context(sentences, clubs, pid, prefix_pid, suffix_pid)
                if not context: continue
                total += 1
                
                if len(question['answerSpan']): is_impossible = False
                else: is_impossible = True; answers = []; snippet = {'context': context, 'question':text, 'id':case + file.split('.')[0] + '-' +  str(total), 'title':file.split('.')[0], 'is_impossible':is_impossible, 'answers': {'text': [], 'answer_start':[]}}
                if not is_impossible:
                    answerable += 1
                    if args.answer_type == 'cont': 
                        # no multi-speaker support
                        new_span_ids = list(range(question['answerSpan'][0], question['answerSpan'][-1] + 1))
                        if question['answerSpan'] != new_span_ids: import pdb; pdb.set_trace()
                        answers, _, answer_spans, _ = obtain_formatted_answers(new_span_ids, sentences, clubs, separate_speaker_label, mask_identity, multi_speaker, no_speakers)
                    else:
                        answers, is_multispan, answer_spans, alt_dict = obtain_formatted_answers(question['answerSpan'], sentences, clubs, separate_speaker_label, mask_identity, multi_speaker, no_speakers)
                    
                    if args.answer_type == 'concat' or args.answer_type == 'cont':
                        
                        if no_speakers: answer_text = "".join(answers).rstrip()
                        else: answer_text = " ".join(answers).rstrip()
                        identifier = case + file.split('.')[0] + '-' + str(total)
                        
                        aid = context.find(answer_text)
                        
                        if aid < 0:
                            # print('used partial answers for', file)
                            # consider overflow cases when entire answer does not fit in context
                            for i in range(len(answer_spans) - 1, -1, -1):
                                
                                answer_text = " ".join(answer_spans[:i]).rstrip()
                                aid = context.find(answer_text)
                                if aid >= 0: break
                        assert aid >= 0, pdb.set_trace() 
                        answer_start = aid
                        if force_unanswerable and np.random.random() < 0.25:
                            identifier = identifier + '-' + 'forceNoAns'
                            marker = True
                            original_context = context
                            if separate_speaker_label: 
                                context = context[:answer_start] + '\n' + context[answer_start:]
                            try: 
                                context = context.replace(answer_text + '\n', '' )
                            except: context = context.replace(answer_text, '' )
                            
                            answer_text = ''
                        if answer_text == '':
                            snippet = {'context': context, 'question':text, 'title': file.split('.')[0], 'id': identifier, 'is_impossible':True, 'answers':{'text': [], 'answer_start': []} }
                        else:
                            answer_lengths.append(len(answer_text.split()))
                            snippet = {'context': context, 'question':text, 'title': file.split('.')[0], 'id':case + file.split('.')[0] + '-' + str(total), 'is_impossible':is_impossible, 'answers':{'text': [answer_text], 'answer_start': [answer_start]} }
                            
                    elif args.answer_type == 'multispans':
                        answer_start = []
                        for span in answer_spans:
                            span = span.rstrip()
                            aid = context.find(span)
                            if aid < 0: 
                                pass
                                # print('Alert: overflow for ', file)
                            
                            answer_start.append(aid)
                        # accounting for context cutting off the answer!
                        answer_spans = [span for s, span in enumerate(answer_spans) if answer_start[s] >= 0]
                        answer_start = [strt for s, strt in enumerate(answer_start) if answer_start[s] >= 0]
                        if len(answer_spans) == 0:
                            snippet = {'context': context, 'question':text, 'title': file.split('.')[0], 'id':case + file.split('.')[0] + '-' + str(total), 'is_impossible':True, 'answers':{'text': [], 'answer_start': []} }
                        else: snippet = {'context': context, 'question':text, 'title': file.split('.')[0],'id':case + file.split('.')[0] + '-' +  str(total), 'is_impossible':is_impossible, 'answers':{'text':answer_spans, 'answer_start':answer_start}}
                
                if marker: dname = file.split('.')[0] + '-' + str(total) + '-forceNoAns' + '.json'
                else: dname = case + file.split('.')[0] + '-' + str(total) + '.json'
                d = open(os.path.join(dest_path, dname), 'w+')
                json.dump(snippet, d, indent = 4)    

    print('---- Statistics ----')
    print('For {} files ...'.format(len(files)))
    print('Created {} snippets'.format(total))
    print('Out of which {}% are answerable'.format(round(answerable*100/total,2)))
    print('Avg. Club Length: ', np.mean(club_lengths))
    if not len(answer_lengths):
        print('With avg. answer length of {} words'.format(sum(answer_lengths)/len(answer_lengths)))                
    print('Hit Max {}% of times'.format(round(hit_max*100/total, 2)))          
    
        
