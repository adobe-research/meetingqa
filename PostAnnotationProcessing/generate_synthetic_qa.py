import argparse
import os
import json
import spacy
import en_core_web_sm
import re
from tqdm import tqdm
nlp = en_core_web_sm.load()

def collapse_speakers(speakers):
    speakers = sorted(speakers, key=lambda x: (-len(x), x))
    
    trunc_speakers = []
    for speaker in speakers:
        insert = True
        if not len(trunc_speakers): trunc_speakers.append(speaker)
        else:
            for trunc in trunc_speakers:
                if speaker in trunc: 
                    insert = False
                    break
        if insert: trunc_speakers.append(speaker)
    trunc_speakers = list(set(trunc_speakers))
    trunc_speakers = sorted(trunc_speakers, key=lambda x: (-len(x), x))
    return trunc_speakers

def club_speakers(data):
    sentences = data['sentences']
    prev_speaker = None
    clubs = []
    speaker_dict = {}
    for s, sentence in enumerate(sentences):
        current_speaker = re.sub("[\(\[].*?[\)\]]", "", str(sentence['speakerFaceId']))
        
        if prev_speaker != current_speaker:
            
            if speaker_dict: clubs.append(speaker_dict)
            speaker_dict = {}
            speaker_dict['speaker'] = current_speaker
            speaker_dict['displayText'] = sentence['displayText']
            speaker_dict['start'] = s
            speaker_dict['end'] = s
        else:
            speaker_dict['displayText'] = " ".join([speaker_dict['displayText'], sentence['displayText']])
            speaker_dict['end'] = s
        prev_speaker = current_speaker
        if s == len(sentences) -1: clubs.append(speaker_dict)
    return clubs

def find_paragraph(sid, clubs):
    for c, club in enumerate(clubs):
        if club['start'] <= sid and sid <= club['end']:
            return c
            
def is_question(text):
    if text[-1] == '?': return True
    return False

def is_host(sname, speakers):
    
    if 'host' in sname.lower(): return True
    hname = ''
    for speaker in speakers:
        if 'host' in speaker.lower(): hname = speaker
    if len(hname) and sname in hname: return True
    return False
    

def match_speaker(sname, speakers):
    sname = re.sub("[\(\[].*?[\)\]]", "", sname)
    for speaker in speakers:
        if sname.lower() in speaker.lower():
            return speaker
    return None

def identify_person(question, text, speakers):
    doc = nlp(question)
    person_ents = [str(e) for e in doc.ents]
    person_ents = [e for e in person_ents if match_speaker(e, speakers)]
    person_ents = [match_speaker(e, speakers) for e in person_ents]

   
    for speaker in speakers:
        splits = speaker.split()
        for split in splits: 
            if len(split) and split.isupper() and split.capitalize() in question:
                person_ents.append(speaker)
    person_ents = [e for e in person_ents if 'host' not in e]
    person_ents = list(set(person_ents))
    
    if len(person_ents): 
        return person_ents

    doc = nlp(text)
    person_ents = [str(e) for e in doc.ents]
    person_ents = [e for e in person_ents if match_speaker(e, speakers)]
    person_ents = [match_speaker(e, speakers) for e in person_ents]

    
    for speaker in speakers:
        splits = speaker.split()
        for split in splits: 
            if len(split) and split.isupper() and split.capitalize() in text:
                person_ents.append(speaker)
    person_ents = list(set(person_ents))
    return person_ents

if __name__ == "__main__":

    source_path = 'ProcessedTranscripts/MediaSumInterviews'
    dest_path = 'AllData/Annotated'
    if not os.path.exists(dest_path): os.makedirs(dest_path)

    files = os.listdir(source_path)
    files.sort()

    desired = False
    print(files[:10])

    for file in tqdm(files):    
        
        data = json.load(open(os.path.join(source_path, file)))
        clubs = club_speakers(data)
        sentences = data['sentences']
        speakers = list(set(sentence['speakerFaceId'] for sentence in sentences))
        speakers = [re.sub("[\(\[].*?[\)\]]", "", speaker) for speaker in speakers]
        
        speakers = collapse_speakers(speakers)
        
        questions_dict = {}
        for sid, sentence_dict in enumerate(sentences):
            text = sentence_dict['displayText']
            speaker = match_speaker(sentence_dict['speakerFaceId'], speakers) 
            
            sentence_dict['sentenceId'] = sid
            sentence_dict['question'] = {'possible': False, 'meaningful':False, 'questionContext':[], 'combinedQuestion':[], 'answerSpan':[], 'answerStartId':None, 'answerEndId':None }
            if is_question(text) and is_host(speaker, speakers):
                sentence_dict['question']['possible'] = True
                sentence_dict['question']['meaningful'] = True
                answers = []
            
                already_answered = False
                pid = find_paragraph(sid, clubs)
                if pid in questions_dict: 
                    answers = data['sentences'][questions_dict[pid][0]]['question']['answerSpan']
                    already_answered = True
                    questions_dict[pid].append(sid)
                else: questions_dict[pid] = [sid]
                entities = identify_person(text, clubs[pid]['displayText'], speakers)
                entities = [e for e in entities if e != speaker]
                
                if entities and not already_answered:
                    for entity in entities:
                        for c, club in enumerate(clubs[pid+1:]):
                            club_speaker = match_speaker(club['speaker'], speakers) 
                            if club_speaker == entity:
                                answers.extend(list(range(club['start'], club['end'] + 1)))
                                break
                            elif is_host(club_speaker, speakers):
                               
                                break
                        
            
                if not already_answered and not len(answers):
                    if pid >= 1 and pid < len(clubs) -1 and clubs[pid-1]['speaker'] == clubs[pid+1]['speaker']:
                        club = clubs[pid + 1]
                        answers.extend(list(range(club['start'], club['end'] + 1)))
                        

                    else:
                        for c, club in enumerate(clubs[pid+1:]):
                            club_speaker = match_speaker(club['speaker'], speakers) 
                            if is_host(club_speaker, speakers):
                                break
                            else: answers.extend(list(range(club['start'], club['end'] + 1)))
                answers.sort()
                sentence_dict['question']['answerSpan'] = answers
                
                if len(questions_dict[pid]) > 1:
                    for sid in questions_dict[pid]:
                        data['sentences'][sid]['question']['combinedQuestion'] = questions_dict[pid]
            elif is_question(text) and not is_host(speaker, speakers) and len(text.split()) > 2:
                sentence_dict['question']['possible'] = True
                
        if questions_dict:
            f = open(os.path.join(dest_path, file), 'w+')
            json.dump(data, f, indent = 4)
        
        
    
