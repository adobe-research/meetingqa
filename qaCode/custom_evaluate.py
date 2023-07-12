import datasets
from datasets import load_dataset, load_metric
import argparse
import os
import json
import textdistance
import numpy as np
import re

def remove_speakers(text):
    return re.sub(r'Speaker [0-9]*\: ', '', text)

def compute_metrics(examples, predictions, pred_ref_type=False):
    if pred_ref_type:
        predictions = predictions['data']
        for r,ref in enumerate(predictions):
            if not len(ref["answers"]['text']): predictions[r]['answers']['text'] = []
        predictions = {ex["id"]: " ".join(ex['answers']['text']) for ex in predictions}

    valid_ids = list(predictions.keys())
    references = [{"id": ex["id"], "answers": ex['answers']} for ex in examples if ex["id"] in valid_ids]
    ref_ids = [r["id"] for r in references]

    remove = []
    for pred in predictions.keys():
        if pred not in ref_ids: remove.append(pred)
    print('Removed ids due to error: ', remove)
    for pred in remove: del predictions[pred]

    for r,ref in enumerate(references):
        for i in range(len(ref['answers']['text'])):
             references[r]['answers']['text'][i] = remove_speakers(ref['answers']['text'][i])
        
    
    formatted_predictions = [{"id": k, "prediction_text": remove_speakers(v), "no_answer_probability": 0.0} for k, v in predictions.items()]
    metric = load_metric("squad_v2")
    metrics = metric.compute(predictions=formatted_predictions, references=references)
    references = sorted(references, key= lambda d: d['id'])
    formatted_predictions = sorted(formatted_predictions, key= lambda d: d['id'])
    jaccard = []
    for ref, pred in zip(references, formatted_predictions):
        ref = ref['answers']['text'][0] if len(ref['answers']['text']) else ""
        pred = pred['prediction_text']
        jaccard.append(textdistance.jaccard(ref.split(), pred.split()))
    metrics['jaccard'] = 100*np.mean(jaccard)
    for key in metrics.keys():
        metrics[key] = np.round(metrics[key], 2)
    if 'meta' in examples[0].keys():
        keep = ['f1', 'exact', 'jaccard']
        revised_metrics = {k: metrics[k] for k in keep}
        print(revised_metrics)
        print("{},{},{}".format(revised_metrics['f1'], revised_metrics['exact'], revised_metrics['jaccard']))
    else: print(metrics)

def compute_split_metrics(examples, predictions, pred_ref_type=False):
    if pred_ref_type:
        predictions = predictions['data']
        for r,ref in enumerate(predictions):
            if not len(ref["answers"]['text']): predictions[r]['answers']['text'] = []
        predictions = {ex["id"]: " ".join(ex['answers']['text']) for ex in predictions}
    valid_ids = list(predictions.keys())
    impossible_ids = [ex["id"] for ex in examples if ex['meta']["impossible"] and ex['id'] in valid_ids]
    possible_ids = [ex["id"] for ex in examples if not ex['meta']["impossible"] and ex['id'] in valid_ids]
    print(len(impossible_ids), len(examples))
    predictions_noans = {id:predictions[id] for id in impossible_ids}
    predictions_ans = {id:predictions[id] for id in possible_ids}
    predictions_multispan = {ex['id']: predictions[ex['id']] for ex in examples if ex['meta']['multispan'] and ex['id'] in valid_ids}
    predictions_multispeaker = {ex['id']: predictions[ex['id']] for ex in examples if ex['meta']['multispeaker'] and ex['id'] in valid_ids}
    print(len(predictions_multispan), len(predictions_multispeaker))

    
    print('')
    print('1. For Unanswerable Qs')
    compute_metrics(examples, predictions_noans)
    print('')
    print('2. For All Answerable Qs')
    compute_metrics(examples, predictions_ans)
    print('')
    print('3. For MultiSpan Qs')
    compute_metrics(examples, predictions_multispan)
    print('')
    print('4. For MultiSpeaker Qs')
    compute_metrics(examples, predictions_multispeaker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation-file', required=False, default=None)
    parser.add_argument('--test-file', required=False, default=None)
    parser.add_argument('--validation-predictions', required=False, default=None)
    parser.add_argument('--test-predictions', required=False, default=None)
    parser.add_argument('--cache_dir', required=False, default=None)
    parser.add_argument('--pred-ref-type', action='store_true', default=False)
    args = parser.parse_args()

    data_files = {}
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    if args.test_file is not None:
        data_files["test"] = args.test_file

    raw_datasets = load_dataset('json', data_files=data_files, field="data", cache_dir=args.cache_dir)

    if args.validation_file is not None: eval_examples = raw_datasets["validation"]
    if args.test_file is not None: test_examples = raw_datasets["test"]

    if args.validation_predictions is not None:
        predictions = json.load(open(args.validation_predictions))
        print("Eval Metrics.....")
        compute_metrics(raw_datasets['validation'], predictions)
        print('')

    if args.test_predictions is not None:
        predictions = json.load(open(args.test_predictions))
        print("Test Metrics.....")
        compute_metrics(raw_datasets['test'], predictions, args.pred_ref_type)
        print('\n\n')
        print("Split by Cases...")
        compute_split_metrics(raw_datasets['test'], predictions, args.pred_ref_type)


