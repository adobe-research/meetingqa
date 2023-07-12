import json
import os
from tqdm import tqdm
import pdb

def get_prefix_id(sentences, id, threshold):
    start = None
    count = 0
    if id > 0:
        for c, club in enumerate(sentences[id - 1::-1]):
            count += len(club.split())
            if count >= threshold: break
        start = id - 1 -c
        # print('sanity check: ', club['displayText'])
    # else: print('hit min')
    if start is None: start = id
    return start

def get_suffix_id(sentences, id, threshold):
    end = None
    count = 0
    if id < len(sentences) - 1:
        for c, club in enumerate(sentences[id + 1:]):
            count += len(club.split())
            if count >= threshold: break
        end = c + id + 1
        # print('sanity check: ', club['displayText'])
    # else: print('hit max')
    if end is None: end = id
    return end

def create_context(sentences, prefix_id, suffix_id):
    # context = 'Speaker 0: '
    # remaining = ' '.join(sentences[prefix_id: min(suffix_id + 1, len(sentences))])
    # return context + ' ' + remaining
    return "\n".join(sentences[prefix_id: min(suffix_id + 1, len(sentences))]) + '\n'

source_path = "BehanceQA/data/test"
dest_file = 'AllData/Dataset/test-nospeak-behanceQA.json'
id_name = '-'.join(source_path.split('/')[0::2])
files = os.listdir(source_path)
files.sort()

prefix_context = 25
suffix_context = 250

total = 0
weird_question = 0
answerable = 0
answer_lengths = []

tjson = {'version':'0.1.0', 'data':[]}

for f, file in tqdm(enumerate(files)):
    data = json.load(open(os.path.join(source_path, file)))
    sentences = data['sentence']
    questions_dict = {k:d for k,d in data['entity'].items() if 'question' in d['label'].lower()}
    for relation in data['relation'].values():
        questionId = relation['question']
        if questionId not in questions_dict.keys():
            # weird_question += 1
            continue
        answer = data['entity'][relation['answer']]
        questions_dict[questionId]['answer'] = answer
    
    for question in questions_dict.values():
        
        sentenceId = question['span'][0]
        assert sentences[sentenceId] == question['sentence'][0]
        prefix_id = get_prefix_id(sentences, sentenceId, prefix_context)
        suffix_id = get_suffix_id(sentences, sentenceId, suffix_context)
        if 'answer' in question.keys(): is_impossible = False
        else: is_impossible = True
        if not is_impossible and suffix_id < question['answer']['span'][-1]: suffix_id = question['answer']['span'][-1]
        context = create_context(sentences, prefix_id, suffix_id)
        if is_impossible:
            snippet = {'context': context, 'question':sentences[sentenceId], 'id': id_name + '-' + str(total), 'title':file.split('.')[0] + '-' + str(total), 'is_impossible':is_impossible, 'answers': {'text': [], 'answer_start':[]}}
        else:
            # answer_text = " ".join(question['answer']['sentence'])
            answer_text = "\n".join(question['answer']['sentence'])
            answer_lengths.append(len(answer_text.split()))
            aid = context.find(answer_text)
            if aid < 0: 
                print('overflowing file:', file)
            snippet = {'context': context, 'question':sentences[sentenceId], 'id': id_name + '-' + str(total), 'title':file.split('.')[0] + '-' + str(total), 'is_impossible':is_impossible, 'answers': {'text': [answer_text], 'answer_start':[aid]}}
            answerable += 1

        tjson['data'].append(snippet)
        total += 1
t = open(dest_file, 'w+')
json.dump(tjson, t, indent=4)

print('---- Statistics ----')
print('For {} files ...'.format(len(files)))
print('Created {} snippets'.format(total))
print('Out of which {}% are answerable'.format(round(answerable*100/total,2)))
print('With avg. answer length of {} words'.format(sum(answer_lengths)/len(answer_lengths)))    
