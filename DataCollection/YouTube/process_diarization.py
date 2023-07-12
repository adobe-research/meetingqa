from pyannote.audio import Pipeline
import os
import pickle
import json
import argparse
import time
import boto3

s3_client = boto3.client('s3')
s3 = boto3.resource('s3')
bucket = s3.Bucket('meeting-datasets')

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

source_path = '/home/arprasad/YouTube/tmp'
# files = os.listdir(source_path)
objects = [obj for obj in bucket.objects.all()]
objects = [obj for obj in objects if '.wav' in obj.key]

dest_path = '/home/arprasad/YouTube/speakers'
if not os.path.exists(dest_path): os.makedirs(dest_path)

def run(start, bin_size):
    print('Started for {}'.format(start))
    start = start*bin_size
    already_there = [t for t in os.listdir(dest_path)]
    # for fname in fnames[start : start + bin_size]: #[start: start + bin_size]
    # for fname in files[start : start + bin_size]:
    for obj in objects[start : start + bin_size]:
    # apply pretrained pipeline
        stime = time.time()
        fname = obj.key.split('/')[-1]
        s3_client.download_file('meeting-datasets', obj.key, os.path.join(source_path, fname))
        print("Begin ", fname)
        if fname in already_there: continue
        import pdb; pdb.set_trace()
        diarization = pipeline(os.path.join(source_path, fname))
        print('### Time elapsed: \t', time.time() - stime, ' ###')
        os.system('rm {}/*'.format(source_path))
        speakers = []
        speaker_segments = []
        all_turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            all_turns.append(turn.start, turn.end, speaker)
        # all_turns = pickle.load(open('diarization_example.pkl', 'rb'))
        speakers = list(set([t[-1] for t in all_turns]))
        prev_speaker = None
        current_segment = -1
        for t, turn in enumerate(all_turns[:-1]):
            start, end, speaker = turn
            if t == 0:
                current_segment = 0
                speaker_segments.append([speaker, start, end])
            else:
                if speaker == prev_speaker:
                    speaker_segments[current_segment][2] = end
                else:
                    speaker_segments.append([speaker, start, end])
                    current_segment += 1

            prev_speaker = speaker

        # print('List of speakers:\t', str(speakers))
        f = open(os.path.join(dest_path, fname.replace('.wav', '.json')), 'w+')
        json.dump(speaker_segments, f)
    print('Ended for {}'.format(start))
    
    return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Take arguments from commandline')
    parser.add_argument('--start', default=0)
    parser.add_argument('--bin-size', default=100)
    args = parser.parse_args()

    run(int(args.start), int(args.bin_size))           


