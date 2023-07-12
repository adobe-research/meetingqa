import os
import json
import argparse
from create_snippets import create_context, obtain_formatted_answers, pretty_name, find_prefix_paragraph, find_suffix_paragraph
from generate_synthetic_qa import club_speakers, find_paragraph
from tqdm import tqdm 
import pdb
import numpy as np
# from datasets import load_metric
import evaluate
from sentence_transformers import SentenceTransformer, util
# from qaCode.evaluate import compute_metrics


def fix_speakers(data):
    sentences = data['sentences']
    for sent in sentences:
        sent['speakerFaceId'] = str(sent['speakerFaceId'])
    data['sentences'] = sentences
    return data

pre_word_count = 50 #50
post_word_count = 250 #200
total_word_count = pre_word_count + post_word_count

source_path = "ProcessedTranscripts/Annotated-AMI-QA/test" 
dest_path = "AllData/Dataset/" 

# if not os.path.exists(dest_path): os.makedirs(dest_path)

parser = argparse.ArgumentParser(description='Take arguments from commandline')
parser.add_argument('--dataset', default='AMI')
parser.add_argument('--score', default='rouge')
parser.add_argument('--multispan', action='store_true')

args = parser.parse_args()


test_file = os.path.join(dest_path, 'final-' + args.dataset + "-test" +  '.json') # 'ms' denotes multispan

files = os.listdir(source_path)
files = [f for f in files if '.json' in f]
files.sort()


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
intersections = []
predictions = {}

for file in tqdm(files):
    print(file)
    data = json.load(open(os.path.join(source_path, file)))
    data = fix_speakers(data)
    clubs = club_speakers(data)
    sentences = data['sentences']
    sentence_list = [sentence['displayText'] for sentence in sentences]
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
            if args.score == 'rouge':
                rouge = evaluate.load('rouge')
                references = [text] * len(sentence_list)
                scores = rouge.compute(predictions=sentence_list, references=references, use_aggregator=False)['rouge1']
            elif args.score == 'sentence-embed':
                model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
                query_emb = model.encode(text)
                doc_emb = model.encode(sentence_list)
                scores = util.cos_sim(query_emb, doc_emb)[0].cpu().tolist()
            if args.score == 'location':
                context_idxs = list(range(clubs[prefix_pid]['start'], clubs[suffix_pid]['end'] + 1))
            else:
                candidate_idxs = np.argsort(scores)[::-1][:total_word_count]
                budget = total_word_count
                context_idxs = [sid]
                for cand in candidate_idxs[1:]:
                    if budget <= 0: break
                    context_idxs.append(cand) 
                    budget = budget - len(sentence_list[cand].split())
                
                context_idxs.sort()
            context = create_context(clubs, pid, prefix_pid, suffix_pid, separate_qa, separate_speaker_label, mask_identity)
            total += 1
            if not context: continue
            if len(question['answerSpan']): is_impossible = False
            else: 
                is_impossible = True; answers = []; 
                snippet = {'context': context, 'question':text, 'id':case + file.split('.')[0] + '-' +  str(sid), 'title':file.split('.')[0], 'is_impossible':is_impossible, 'answers': {'text': [], 'answer_start':[]}}
                test['data'].append(snippet)   
            if not is_impossible:
                intersect = list(set(context_idxs) & set(question['answerSpan'])) 
                intersections.append(len(intersect) / len(question['answerSpan']))
                
                intersect_text = " ".join([sentence_list[j] for j in intersect])
                predictions[case + file.split('.')[0] + '-' +  str(sid)] = intersect_text
                continue
                if sorted(question['answerSpan']) != sorted(list(set(question['answerSpan']))): mistakes.append((file, sid, question['answerSpan']))
                question['answerSpan'] = sorted(list(set(question['answerSpan']))) #Omit copies in answerSpan if they occur by mistake
                for ans in question['answerSpan']:
                    if abs(ans - sid) > max_distance: 
                        mistakes.append((file, sid, question['answerSpan']))
                answer_length.append(len(question['answerSpan']))
                answerable += 1
                question['answerSpan'] = list(range(question['answerSpan'][0], question['answerSpan'][-1] + 1))
                try: answers, is_multispan, answer_spans, _ = obtain_formatted_answers(question['answerSpan'], sentences, clubs, separate_speaker_label, mask_identity, multi_speaker, no_speakers)
                except: 
                    mistakes.append((file, sid, question['answerSpan']))
                    continue
                    
                if text == 'Has anyone actually looked at the Java code for the?': desired = True
                else: desired = False
                if args.multispan:
                    snippet = {'context': context, 'question':text, 'title': file.split('.')[0], 'id': case + file.split('.')[0] + '-' + str(sid), 'answers': {'text':[], 'answer_start':[]}}
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
                    
                else:
                    if is_multispan: num_multispan += 1
                    answer_text = " ".join(answers).rstrip()
                    aid = context.find(answers[0])
                    if aid < 0:
                        print('Answer not in context')
                        answer_start = 0
                    answer_start = aid
                    if answer_text == '':
                        snippet = {'context': context, 'question':text, 'title': file.split('.')[0], 'id': identifier, 'is_impossible':True, 'answers':{'text': [], 'answer_start': []} }
                        # never enters this case
                    else:
                        num_answerable += 1
                        
                        snippet = {'context': context, 'question':text, 'title': file.split('.')[0], 'id':case + file.split('.')[0] + '-' + str(sid), 'is_impossible':is_impossible, 'answers':{'text': [answer_text], 'answer_start': [answer_start]} }
                    
                test['data'].append(snippet)
    

if args.score == 'rouge':
    t = open('rouge1-retrieved-upperbound.json', 'w+')
    json.dump(predictions, t, indent = 4)
elif args.score == 'sentence-embed':
    t = open('sentembed-retrieved-upperbound.json', 'w+')
    json.dump(predictions, t, indent = 4)
elif args.score == 'location':
    t = open('location-retrieved-upperbound.json', 'w+')
    json.dump(predictions, t, indent = 4)
print(np.mean(intersections))
print(len(mistakes))
for mistake in mistakes:
    print(mistake)

print(num_questions)
print(len(test['data']))
print(num_answerable)
print(100*num_answerable/num_questions)
print(100*num_multispan/num_questions)
print(np.mean(answer_length))
