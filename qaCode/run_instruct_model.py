import os
import re
import json
import numpy as np
import pdb
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from instruct_utils import MeetingQAQuestionDataset, prepare_question_dataset, prepare_answer_dataset, get_sentence_speaker, get_question_utterance
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from nltk import tokenize
from custom_evaluate import compute_metrics

question_prompt = "[CONTEXT]\nBased on the conversation above, what question did [SPEAKER] ask?"
answerable_prompt = "[CONTEXT]\nBased on the conversation above, did anyone answer [SPEAKER]'s question: " + '"[QUESTION]". Respond "yes" if answered, "no" otherwise.'
# answer_prompt = "[CONTEXT] \nIn the conversation above, list the sentences that answer [SPEAKER]'s question: " + '"[QUESTION]"'
answer_prompt = "[CONTEXT]\nBased on the conversation above, which sentences from the converstation answer [SPEAKER]'s question: " + '"[QUESTION]"'

dest_path = 'Results/flan-t5-xxl/'
if not os.path.exists(dest_path): os.makedirs(dest_path)
response_dict = {'question_prompt': question_prompt, 'answer_prompt': answer_prompt, 'predictions':{}}
vanilla_predict, vanilla_filtered, answerable_predict, answerable_filtered, oracle_predict, oracle_filtered = {}, {}, {}, {}, {}, {}
sub_predict, sub_filtered, extans_predict, extans_filtered = {}, {}, {}, {}

def apply_question_prompt(contexts, speakers):
    batch = []
    assert len(contexts) == len(speakers), pdb.set_trace()
    for i in range(len(contexts)):
        context = contexts[i]
        speaker = speakers[i]
        instance = question_prompt
        instance = instance.replace('[SPEAKER]', speaker).replace('[CONTEXT]', context)
        batch.append(instance)
    return batch

def apply_answer_prompt(contexts, speakers, questions):
    batch = []
    assert len(contexts) == len(speakers), pdb.set_trace()
    for i in range(len(contexts)):
        context = contexts[i]
        speaker = speakers[i]
        question = questions[i]
        instance = answer_prompt
        instance = instance.replace('[SPEAKER]', speaker).replace('[CONTEXT]', context).replace('[QUESTION]', question)
        batch.append(instance)
    return batch

def apply_answerable_prompt(contexts, speakers, questions):
    batch = []
    assert len(contexts) == len(speakers), pdb.set_trace()
    for i in range(len(contexts)):
        context = contexts[i]
        speaker = speakers[i]
        question = questions[i]
        instance = answerable_prompt
        instance = instance.replace('[SPEAKER]', speaker).replace('[CONTEXT]', context).replace('[QUESTION]', question)
        batch.append(instance)
    return batch

def apply_substituted_prompt(contexts, speakers, questions, response):
    batch = []
    assert len(contexts) == len(speakers), pdb.set_trace()
    for i in range(len(contexts)):
        context = contexts[i]
        speaker = speakers[i]
        question = questions[i]
        context = context.replace(question, response[i])
        question = response[i]
        instance = answer_prompt
        instance = instance.replace('[SPEAKER]', speaker).replace('[CONTEXT]', context).replace('[QUESTION]', question)
        batch.append(instance)
    return batch

def exact_search(context, answer):
    filtered_answer = []
    for sent in tokenize.sent_tokenize(answer):
        if ":" in sent: speaker = get_sentence_speaker(sent)
        else: speaker = ""
        filt = re.sub(r'Speaker [0-9]*\: ', '', sent)
        idx = context.find(sent)
        if idx >= 0:
            orig_speaker = get_sentence_speaker(get_question_utterance(context, sent))
            if  not len(speaker) or speaker == orig_speaker: filtered_answer.append(sent)
    
    return " ".join(filtered_answer)


num_gen_tokens = 1024

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token 
if torch.cuda.is_available():
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl", device_map="auto", torch_dtype=torch.float16)
else:
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl")

data_files = '/home/arprasad/AllData/Dataset/final-AMI-test-meta.json'
answerable_preds = json.load(open('/home/arprasad/qaCode/Results/roberta-answerable-f1/predictions_answerable.json'))
dataset = load_dataset('json', data_files = {'test': data_files}, field="data")
dataset = dataset['test'].map(prepare_question_dataset)
raw_dataset = dataset
dataset = dataset.map(lambda example: {'answer_text': " ".join(example['answers']['text'])}, remove_columns=['answers'])
data_loader = DataLoader(dataset, batch_size=4, shuffle=False)


for batch in tqdm(data_loader):
    
    input_text = apply_question_prompt(batch['context'], batch['question_speaker'])
    if torch.cuda.is_available():
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to("cuda")
    else: 
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=num_gen_tokens)
    questions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # import pdb; pdb.set_trace()

    input_text = apply_answerable_prompt(batch['context'], batch['question_speaker'], batch['question'])
    if torch.cuda.is_available():
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to("cuda")
    else: 
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=num_gen_tokens)
    is_answerable = tokenizer.batch_decode(outputs, skip_special_tokens=True)


    input_text = apply_answer_prompt(batch['context'], batch['question_speaker'], batch['question'])
    if torch.cuda.is_available():
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to("cuda")
    else: 
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=num_gen_tokens)
    answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    #Begin Pipeline
    input_text = apply_substituted_prompt(batch['context'], batch['question_speaker'], batch['question'], questions)
    if torch.cuda.is_available():
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to("cuda")
    else: 
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)

    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=num_gen_tokens)
    sub_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for i, id in enumerate(batch['id']):
        response_dict['predictions'][id] = {'context': batch['context'][i], 'question': batch['question'][i], 'answer_text': batch['answer_text'][i], 'model': {'predict_question': questions[i], 'predict_answer': answers[i], 'predict_answerable': is_answerable[i], 'predict_sub_answer': sub_answers[i]}}
        vanilla_predict[id] = answers[i]
        vanilla_filtered[id] = exact_search(batch['context'][i], answers[i])
        sub_predict[id] = sub_answers[i]
        sub_filtered[id] = exact_search(batch['context'][i], sub_answers[i])
        answerable_predict[id] = answers[i] if is_answerable[i].lower() == 'yes' else ""
        answerable_filtered[id] = exact_search(batch['context'][i], answers[i]) if is_answerable[i].lower() == 'yes' else ""
        extans_predict[id] = answers[i] if answerable_preds[id]  else ""
        extans_filtered[id] = exact_search(batch['context'][i], answers[i]) if answerable_preds[id]  else ""
        oracle_predict[id] = answers[i] if not batch['is_impossible'][i] else ""
        oracle_filtered[id] = exact_search(batch['context'][i], answers[i]) if not batch['is_impossible'][i] else ""
    # break
    # import pdb; pdb.set_trace()
print('Just Answers')
vanilla_metrics = compute_metrics(raw_dataset, vanilla_filtered)
print('\nSub Answers')
sub_metrics = compute_metrics(raw_dataset, sub_filtered)
print('\nAsk Answerable')
answerable_metrics = compute_metrics(raw_dataset, answerable_filtered)
print('\nExt Answerable')
ext_metrics = compute_metrics(raw_dataset, extans_filtered)
print('\nOracle Answerable')
oracle_metrics = compute_metrics(raw_dataset, oracle_filtered)
# import pdb; pdb.set_trace
f = open(os.path.join(dest_path, 'predictions_dump.json'), 'w+')
json.dump(response_dict, f, indent=4)

f = open(os.path.join(dest_path, 'vanilla_predictions.json'), 'w+')
json.dump(vanilla_filtered, f, indent=4)
f = open(os.path.join(dest_path, 'sub_predictions.json'), 'w+')
json.dump(sub_filtered, f, indent=4)
f = open(os.path.join(dest_path, 'answerable_predictions.json'), 'w+')
json.dump(answerable_filtered, f, indent=4)
f = open(os.path.join(dest_path, 'extans_predictions.json'), 'w+')
json.dump(extans_filtered, f, indent=4)
f = open(os.path.join(dest_path, 'oracle_predictions.json'), 'w+')
json.dump(oracle_filtered, f, indent=4)
