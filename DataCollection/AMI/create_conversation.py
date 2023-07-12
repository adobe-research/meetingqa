import os
import json
import pdb
import argparse
import random 

parser = argparse.ArgumentParser(description='Take arguments from commandline')
parser.add_argument('--source-path', default='/home/arprasad/ProcessedTranscripts/AnnotationProposal/FinalAMI', help='Path to where processed transcripts lie')
parser.add_argument('--dataset', default='AMI', help='Name of dataset')
args = parser.parse_args()

source_path = os.path.join(args.source_path, args.dataset)

files = list(set([f.split('.')[0] for f in os.listdir(source_path)]))
for file in files:
    source_file = '{}/{}.json'.format(source_path,file)
    data = json.load(open(source_file))
    dest_file = '{}/{}.tsv'.format(source_path, file)
    f = open(dest_file, 'w+')

    prev_speaker = ''
    current_speaker = ''

    sentences = data['sentences']
    f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('Speaker ID', 'Text/Utterance', 'Sentence Number', 'Possible Question', 'Meaningful', 'Context', 'Combined Question IDs', 'AnswerSpan', 'Unanswerable Reason', 'Confident Answer', 'External Review'))
    for sentence in sentences:
        curr_speaker = sentence['speakerFaceId']
        text = sentence['displayText']
        id = sentence['sentenceId']
        if curr_speaker != prev_speaker: 
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format("Speaker " + str(curr_speaker) + ':', text,id,  sentence['question']['possible'], sentence['question']['meaningful'], sentence['question']['questionContext'], sentence['question']['combinedQuestion'], sentence['question']['answerSpan'], '', 'TRUE', 'FALSE'))
        else:
            f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format('', text, id, sentence['question']['possible'], sentence['question']['meaningful'], sentence['question']['questionContext'], sentence['question']['combinedQuestion'], sentence['question']['answerSpan'], '', 'TRUE', 'FALSE'))
        prev_speaker = curr_speaker

    f.close()

        
