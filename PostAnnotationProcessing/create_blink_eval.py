import os
import json

def check_if_question(text):
    if text[-1] == "?": 
        # if len(text.split()) > 2: return True #Orig: 2
        return True
        # else: print(text)
    return False

source_path = "ProcessedTranscripts/BlinkDemo"
dest_path = 'ProcessedTranscripts/BlinkDemo/DummyAnnote'
if not os.path.exists(dest_path): os.makedirs(dest_path)

files = os.listdir(source_path)
files.sort()

for file in files:
    num_questions = 0
    data = json.load(open(os.path.join(source_path, file)))
    new_data = {'sentences':[]}
    sentences = data['sentences']
    new_data['displayText'] =  data['displayText']
    for s, sentence in enumerate(sentences):
        sentences_dict = {}
        if not len(sentence['displayText']): continue
        # import pdb; pdb.set_trace()
        sentences_dict['displayText'] = sentence['displayText']
        sentences_dict['speakerFaceId'] = sentence['speakerFaceId']
        sentences_dict['sentenceId'] = s
        sentences_dict['question'] = {'possible': False, 'meaningful':False, 'questionContext':[], 'combinedQuestion':[], 'answerSpan':[]}
        if check_if_question(sentence['displayText']):
            sentences_dict['question']['possible'] = True
            num_questions += 1
        new_data['sentences'].append(sentences_dict)
    print('Num Questions: ', num_questions)
    f = open(os.path.join(dest_path, file), "w+")
    json.dump(new_data, f, indent = 4)
    f.close()
        




