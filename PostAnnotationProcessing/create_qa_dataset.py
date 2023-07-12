import argparse
import os
import json
from tqdm import tqdm
import random

random.seed(2022)

source_path = 'AllData/AMI-Train'
dest_path = 'AllData/Dataset'
if not os.path.exists(dest_path): os.makedirs(dest_path)

all_test = True

files = os.listdir(source_path)
files.sort()

answerable_files = []
unanswerable_files = []
forced_unanswerable_files = []

for file in tqdm(files):
    data = json.load(open(os.path.join(source_path, file)))
    if 'is_impossible' not in data.keys():
        print(file)
        continue
    if not data['is_impossible']: answerable_files.append(file)
    else: 
        if 'forceNoAns' in data['id']: forced_unanswerable_files.append(file)
        else: unanswerable_files.append(file)

forced = len(forced_unanswerable_files) > 0
# import pdb; pdb.set_trace()
print('Total of {} answered snippets'.format(len(answerable_files)))

# unanswerable_files = [f for f in tqdm(files) if f not in answerable_files]

print('Creating file splits...')

if all_test:
    test = {'version':"0.1.0", 'data':[]}
    for file in tqdm(files):
        data = json.load(open(os.path.join(source_path, file)))
        test['data'].append(data)
    # key = source_path.split('/')[-1].lower()
    # te = open(os.path.join(dest_path, 'test-{}.json'.format(key)), "w+")
    te = open(os.path.join(dest_path, 'final-AMI-aug-train.json'), "w+")
    json.dump(test, te, indent = 4)
    te.close()
    exit()

num_train = round(len(answerable_files)*0.8)
num_test = round(len(answerable_files)*0.1)
num_dev = len(answerable_files) - num_train - num_test

num_un_train = int(num_train/3)
num_un_test = int(num_test/3)
num_un_dev = int(num_dev/3)

if forced:
    num_un_train = int(3*num_train/14)
    num_un_test = int(3*num_test/14)
    num_un_dev = int(3*num_dev/14)

train = {'version':"0.1.0", 'data':[]}
dev = {'version':"0.1.0", 'data':[]}
test = {'version':"0.1.0", 'data':[]}

random.shuffle(answerable_files)
random.shuffle(unanswerable_files)

count = 0
for file in tqdm(answerable_files):
    data = json.load(open(os.path.join(source_path, file)))
    if count < num_train:
        train['data'].append(data)
    elif count < num_train + num_dev:
        dev['data'].append(data)
    else:
        test['data'].append(data)
    count += 1

unanswerable_files = random.sample(unanswerable_files, num_un_train + num_un_dev + num_un_test)

count = 0
for file in tqdm(unanswerable_files):
    data = json.load(open(os.path.join(source_path, file)))
    if count < num_un_train:
        train['data'].append(data)
    elif count < num_un_train + num_un_dev:
        dev['data'].append(data)
    else:
        test['data'].append(data)
    count += 1

count = 0
for file in tqdm(forced_unanswerable_files):
    data = json.load(open(os.path.join(source_path, file)))
    if count < num_un_train:
        train['data'].append(data)
    elif count < num_un_train + num_un_dev:
        dev['data'].append(data)
    else:
        test['data'].append(data)
    count += 1

random.shuffle(train['data'])
random.shuffle(dev['data'])
random.shuffle(test['data'])

# import pdb; pdb.set_trace()

print('Statistics...')
print('Train:\t', len(train['data']))
print('Dev:\t', len(dev['data']))
print('Test:\t', len(test['data']))

if forced: #forced:
    tr = open(os.path.join(dest_path, 'multispan-train-med.json'), "w+")
    d = open(os.path.join(dest_path, 'multispan-dev-med.json'), "w+")
    te = open(os.path.join(dest_path, 'multispan-test-med.json'), "w+")

    train_med = {'version':'0.1.0', 'data':train['data'][:round(len(train['data'])/5)]}
    dev_med = {'version':'0.1.0', 'data':dev['data'][:round(len(dev['data'])/5)]}
    test_med = {'version':'0.1.0', 'data':test['data'][:round(len(test['data'])/5)]}

    json.dump(train_med, tr, indent=4)
    json.dump(dev_med, d, indent=4)
    json.dump(test_med, te, indent=4)

    tr.close()
    d.close()
    te.close()


import pdb; pdb.set_trace()

# tr = open(os.path.join(dest_path, 'train.json'), "w+")
# d = open(os.path.join(dest_path, 'dev.json'), "w+")
# te = open(os.path.join(dest_path, 'test.json'), "w+")

# json.dump(train, tr, indent=4)
# json.dump(dev, d, indent=4)
# json.dump(test, te, indent=4)

# tr.close()
# d.close()
# te.close()

tr = open(os.path.join(dest_path, 'multi-speakerrevised-train-med.json'), "w+")
d = open(os.path.join(dest_path, 'multi-speakerrevised-dev-med.json'), "w+")
te = open(os.path.join(dest_path, 'multi-speakerrevised-test-med.json'), "w+")

train_med = {'version':'0.1.0', 'data':train['data'][:round(len(train['data'])/5)]}
dev_med = {'version':'0.1.0', 'data':dev['data'][:round(len(dev['data'])/5)]}
test_med = {'version':'0.1.0', 'data':test['data'][:round(len(test['data'])/5)]}

json.dump(train_med, tr, indent=4)
json.dump(dev_med, d, indent=4)
json.dump(test_med, te, indent=4)

tr.close()
d.close()
te.close()

import pdb; pdb.set_trace()

tr = open(os.path.join(dest_path, 'train-small.json'), "w+")
d = open(os.path.join(dest_path, 'dev-small.json'), "w+")
te = open(os.path.join(dest_path, 'test-small.json'), "w+")

train_small = {'version':'0.1.0', 'data':train['data'][:5000]}
dev_small = {'version':'0.1.0', 'data':dev['data'][:500]}
test_small = {'version':'0.1.0', 'data':test['data'][:500]}

json.dump(train_small, tr, indent=4)
json.dump(dev_small, d, indent=4)
json.dump(test_small, te, indent=4)

tr.close()
d.close()
te.close()



