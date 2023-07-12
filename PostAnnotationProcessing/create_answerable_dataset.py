import os
import argparse
import json
import pdb
from tqdm import tqdm

def nice_split(my_list):
    return [l for l in my_list if len(l)]

parser = argparse.ArgumentParser(description='Take arguments from commandline')
parser.add_argument('--source-file', default='AllData/Dataset/final-AMI-train.json', type=str)
parser.add_argument('--dest-path', default='AllData/AnswerableDataset/', type=str)
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
    
    answerable = entry['is_impossible']
    entry_dict = {'id': entry['id'], 'question': question, 'context': context, 'label': int(answerable)}
    data['data'].append(entry_dict)

fname = args.source_file.split('/')[-1]
f = open(os.path.join(dest_path, fname), "w+")
json.dump(data, f, indent = 4)

print(weird)
print(total)

    



    

