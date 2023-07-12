import os
import argparse
import json
import pdb
from tqdm import tqdm

def nice_split(my_list):
    return [l for l in my_list if len(l)]

parser = argparse.ArgumentParser(description='Take arguments from commandline')
parser.add_argument('--source-file', default='AllData/Dataset/final-AMI-ms-test.json', type=str)
parser.add_argument('--dest-path', default='AllData/MultiSpanDataset/', type=str)
args = parser.parse_args()

dest_path = args.dest_path
if not os.path.exists(dest_path): os.makedirs(dest_path)

content = json.load(open(args.source_file))
data = {'version': '0.1.0', 'data':[]}

desired = False

weird = 0
total = 0
for entry in tqdm(content['data']):
    total += 1
    question = entry['question']
    
    context = entry['context']
    answers = entry['answers']
    
    if not entry['is_impossible']: 
        
        answer_starts = [int(a) for a in answers['answer_start']]
        answer_ends = [answer_starts[a] + len(answers['text'][a]) for a in range(len(answer_starts))]
        prev_end = 0
        
        entry_dict = {'id': entry['id'], 'question':nice_split(question.split(" ")), 'context':nice_split(context.split(" ")), 'labels':[]}
        # Labeling for question is simple, all are "other" tags
        for word in entry_dict['question']:
            entry_dict['labels'].append("O")
        
        for s, strt in enumerate(answer_starts):   
            for word in nice_split(context[prev_end:strt].split(" ")):
                entry_dict['labels'].append("O")
            for word in nice_split(context[strt: answer_ends[s]].split(" ")):
                entry_dict['labels'].append("I_ANSWER")
            prev_end = answer_ends[s]

        for word in nice_split(context[prev_end:].split(" ")):
                entry_dict['labels'].append("O")

            
        
    else:
        entry_dict = {'id': entry['id'], 'question':nice_split(question.split(" ")), 'context':nice_split(context.split(" ")), 'labels':[]}
        for word in entry_dict['question']:
            entry_dict['labels'].append("O")
        for word in entry_dict['context']:
            entry_dict['labels'].append("O")

    combined = entry_dict['question'] + entry_dict['context']
    if len(entry_dict['labels']) != len(entry_dict['question']) + len(entry_dict['context']):
        
        weird += 1
        continue
    
    answer_text = [c for i, c in enumerate(combined) if entry_dict['labels'][i] == 'I_ANSWER']
    
    
    if desired: pdb.set_trace()
    data['data'].append(entry_dict)

fname = args.source_file.split('/')[-1]
f = open(os.path.join(dest_path, fname), "w+")
json.dump(data, f, indent = 4)

print(weird)
print(total)

    



    

