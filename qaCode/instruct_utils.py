import torch
import os
import json
import re
import numpy as np

class MeetingQAQuestionDataset(torch.utils.data.Dataset):
    def __init__(self, data_file):
        super(MeetingQAQuestionDataset, self).__init__()
        self.data_file = data_file
        self.raw_data = json.load(open(self.data_file))['data']

    def get_question_utterance(self, context, question):
        utterances = context.split('\n')
        for utt in utterances:
            if question in utt:
                return utt
        return ''

    def get_sentence_speaker(self, text):
        speakerRegex = re.compile('Speaker [0-9]*:')
        speaker = speakerRegex.search(text).group()
        return speaker.rstrip(':')

    def __getitem__(self, idx):
        item = {'context':'', 'question':'', 'question_speaker':'', 'is_impossible':False}
        raw_item = self.raw_data[idx]
        item['context'] = raw_item['context']
        item['question'] = raw_item['question']
        item['question_speaker'] = self.get_sentence_speaker(self.get_question_utterance(item['context'], item['question']))
        item['is_impossible'] = raw_item['is_impossible']
        return item

    def __len__(self):
        return len(self.raw_data)

def get_question_utterance(context, question):
    utterances = context.split('\n')
    for utt in utterances:
        if question in utt:
            return utt
    return ''

def get_sentence_speaker(text):
    if ":" in text:
        speakerRegex = re.compile('Speaker [0-9]*:')
        try: speaker = speakerRegex.search(text).group()
        except: speaker = ''
    else: speaker = ''
    return speaker.rstrip(':')

def prepare_question_dataset(example):
    example['question_speaker'] = get_sentence_speaker(get_question_utterance(example['context'], example['question']))
    return example

def prepare_answer_dataset(example):
    example['answer_text'] = example['answers']['text']

