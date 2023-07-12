import os
import json
import argparse
from create_snippets import create_context, obtain_formatted_answers, pretty_name, find_prefix_paragraph, find_suffix_paragraph
from generate_synthetic_qa import club_speakers, find_paragraph
from tqdm import tqdm 
import pdb
import numpy as np

def fix_speakers(data):
    sentences = data['sentences']
    for sent in sentences:
        sent['speakerFaceId'] = str(sent['speakerFaceId'])
    data['sentences'] = sentences
    return data

def process_answers(qid, answers, sentences):
    multispan = False
    speakers = set()
    offset = 0
    prev_ans = None
    answer_text = []
    for a, ans in enumerate(answers):
        ans_sentence = sentences[ans]
        ans_speaker = pretty_name(ans_sentence['speakerFaceId'])
        answer_text.append(ans_sentence['displayText'])
        speakers.add(ans_speaker)
        if a and ans - prev_ans > 1:
            multispan = True
        if not a: offset = ans - qid
        prev_ans = ans
    num_speakers = len(speakers)
    return multispan, num_speakers, offset, " ".join(answer_text)

pre_word_count = 50 #50
post_word_count = 250 #200

source_path = "ProcessedTranscripts/Annotated-AMI-QA/test" 
dest_path = "AllData/Dataset/" 

# if not os.path.exists(dest_path): os.makedirs(dest_path)

parser = argparse.ArgumentParser(description='Take arguments from commandline')
parser.add_argument('--dataset', default='AMI')
parser.add_argument('--multispan', action='store_true')
# parser.add_argument('--answer-type', default='concat', choices=['concat', 'cont', 'multispans'], help='type of answers to feed in')
args = parser.parse_args()

# source_path = os.path.join(source_path, args.dataset)
# if os.path.exists(source_path): source_path = new_source_path
test_file = os.path.join(dest_path, 'final-' + args.dataset + "-ms-test" +  '.json') # 'ms' denotes multispan

files = os.listdir(source_path)
files = [f for f in files if '.json' in f]
files.sort()

# files = [f for f in files if 'IB4003' in f]
print(files)

total = 0
answerable = 0
hit_max = 0

separate_qa = False
separate_speaker_label = False
mask_identity = True
multi_speaker = False
no_speakers = False

test = {'version': '0.1.0', 'data':[]}

mistakes = []
num_questions = 0
num_answerable = 0
num_multispan = 0
num_wh_questions = 0
answer_length = []
max_distance = 50
missed = 0

for file in tqdm(files):
    print(file)
    data = json.load(open(os.path.join(source_path, file)))
    data = fix_speakers(data)
    clubs = club_speakers(data)
    sentences = data['sentences']
    case = '{}-{}-{}-'.format(int(separate_qa), int(separate_speaker_label), int(mask_identity))
    for sid, sentence_dict in enumerate(sentences):
        question = sentence_dict['question']
        text = sentence_dict['displayText']

        if question['possible']:
            num_questions += 1
            snippet = {}
            pid = find_paragraph(sid, clubs)
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
            context = create_context(clubs, pid, prefix_pid, suffix_pid, separate_qa, separate_speaker_label, mask_identity)
            total += 1
            if not context: continue
            if len(question['answerSpan']): is_impossible = False
            else: 
                is_impossible = True; answers = []; 
                snippet = {'context': context, 'question':text, 'id':case + file.split('.')[0] + '-' +  str(sid), 'title':file.split('.')[0], 'is_impossible':is_impossible, 'answers': {'text': [], 'answer_start':[]}, 'meta':{'impossible': True, 'multispan': False, 'multispeaker':False}}
                test['data'].append(snippet)   
            if not is_impossible:
                if sorted(question['answerSpan']) != sorted(list(set(question['answerSpan']))): mistakes.append((file, sid, question['answerSpan']))
                question['answerSpan'] = sorted(list(set(question['answerSpan']))) #Omit copies in answerSpan if they occur by mistake
                for ans in question['answerSpan']:
                    if abs(ans - sid) > max_distance: 
                        mistakes.append((file, sid, question['answerSpan']))
                answer_length.append(len(question['answerSpan']))
                answerable += 1
                multispan, num_speakers, _, _ = process_answers(sid, question['answerSpan'], sentences)
                if num_speakers > 1: multispeaker = True
                else: multispeaker = False
                if 'train' in test_file: question['answerSpan'] = list(range(question['answerSpan'][0], question['answerSpan'][-1] + 1))
                try: answers, is_multispan, answer_spans, _ = obtain_formatted_answers(question['answerSpan'], sentences, clubs, separate_speaker_label, mask_identity, multi_speaker, no_speakers)
                except: 
                    mistakes.append((file, sid, question['answerSpan']))
                    continue
                    # import pdb; pdb.set_trace()
                # import pdb; pdb.set_trace()
                if text == 'Has anyone actually looked at the Java code for the?': desired = True
                else: desired = False
                if args.multispan:
                    snippet = {'context': context, 'question':text, 'title': file.split('.')[0], 'id': case + file.split('.')[0] + '-' + str(sid), 'answers': {'text':[], 'answer_start':[]}, 'meta':{'impossible': True, 'multispan': False, 'multispeaker':False}}
                    c_start = 0 # Corner case when people repeat the same phrase and both are marked as answers
                    for answer in answer_spans:
                        # if desired: print(c_start, answer)
                        aid = context[c_start:].find(answer)
                        if aid < 0:
                            print('Answer segment not in context')
                            continue
                        aid += c_start
                        c_start = aid + len(answer)
                        snippet['answers']['text'].append(answer)
                        snippet['answers']['answer_start'].append(aid)
                    if len(snippet['answers']) == 0: snippet['is_impossible'] = True
                    else: snippet['is_impossible'] = False
                    # if len(answer_spans) >1: import pdb; pdb.set_trace()
                    # if desired: import pdb; pdb.set_trace()
                else:
                    if is_multispan: num_multispan += 1
                    answer_text = " ".join(answers).rstrip()
                    aid = context.find(answers[0])
                    if aid < 0:
                        print('Answer not in context')
                        missed += 1
                        answer_start = 0
                    answer_start = aid
                    if 'train' in test_file:
                        filtered_answers = []
                        for answer in answers:
                            aaid = context.find(answer)
                            if aaid >= 0: filtered_answers.append(answer)
                            else: break
                        answer_text = " ".join(filtered_answers).rstrip()
                    if answer_text == '':
                        snippet = {'context': context, 'question':text, 'title': file.split('.')[0], 'id': identifier, 'is_impossible':True, 'answers':{'text': [], 'answer_start': []}, 'meta':{'impossible': True, 'multispan': False, 'multispeaker':False}}
                        # never enters this case
                    else:
                        num_answerable += 1
                        # answer_lengths.append(len(answer_text.split()))
                        snippet = {'context': context, 'question':text, 'title': file.split('.')[0], 'id':case + file.split('.')[0] + '-' + str(sid), 'is_impossible':is_impossible, 'answers':{'text': [answer_text], 'answer_start': [answer_start]}, 'meta':{'impossible': False, 'multispan': multispan, 'multispeaker': multispeaker} }
                    # if desired: import pdb; pdb.set_trace()
                test['data'].append(snippet)
    
t = open(test_file, 'w+')
json.dump(test, t, indent = 4)
print(len(mistakes))
for mistake in mistakes:
    print(mistake)

print(missed)
print(num_questions)
print(len(test['data']))
print(num_answerable)
print(100*num_answerable/num_questions)
print(100*num_multispan/num_questions)
print(np.mean(answer_length))
