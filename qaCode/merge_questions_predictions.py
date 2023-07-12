import os
import json
import argparse

parser = argparse.ArgumentParser(description='Take arguments from commandline')
parser.add_argument('--questions-file', default='../AllData/Dataset/questions-test-blinkdemo.json', type=str)
parser.add_argument('--predictions-file', default='', type=str)
# parser.add_argument('--dest-path', default='AllData/Dataset/', type=str)
args = parser.parse_args()

questions_dicts = json.load(open(args.questions_file))['data']
predictions_dict = json.load(open(args.predictions_file))

dest_path = "/".join(args.predictions_file.split('/')[:-1])
dest_file = open(os.path.join(dest_path, 'merged-questions-preds.json'), 'w+')

merged_dict = {}

for qdict in questions_dicts:
    id, question = qdict.values()
    merged_dict[id] = {'question': question, 'pred': predictions_dict[id]}

json.dump(merged_dict, dest_file, indent=4)
dest_file.close()
