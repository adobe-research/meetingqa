#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""

import os
os.environ['TRANSFORMERS_CACHE'] = '../temp/'
import shutil

import argparse
import logging
import math
import random
import json

import datasets
import torch
from datasets import ClassLabel, load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--question_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The column name of label to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lenght` is passed."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help=(
            "The maximum number of evaluations to wait for if there is no imporvement in eval metric before terminating training."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help=""
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help=""
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
    # 'tokens' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field="data")
    # Trim a number of training examples
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    elif raw_datasets["validation"] is not None:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features
    elif raw_datasets["test"] is not None:
        column_names = raw_datasets["test"].column_names
        features = raw_datasets["test"].features

    if args.text_column_name is not None:
        text_column_name = args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if args.question_column_name is not None:
        question_column_name = args.question_column_name
    elif "tokens" in column_names:
        question_column_name = "question"
    else:
        question_column_name = column_names[1]

    if args.label_column_name is not None:
        label_column_name = args.label_column_name
    elif f"{args.task_name}_tags" in column_names:
        label_column_name = f"{args.task_name}_tags"
    else:
        label_column_name = column_names[-1]

    def get_text(batch, labels):
        batch_text = []
        for x in range(len(labels)):
            filtered = batch['input_ids'][x][labels[x] == 0] # Label list sorting ensures this corresponds to "I_ANSWER" and we have only 1 label.
            text = tokenizer.decode(filtered, skip_special_tokens=True)
            batch_text.append(text)         
        return batch_text

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, num_labels=num_labels)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    if not tokenizer_name_or_path:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    try:
        if config.model_type in {"gpt2", "roberta", "longformer", "bigbird"}:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True)
    # For BigBird
    except: tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True)

    if args.model_name_or_path:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForTokenClassification.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[question_column_name],
            examples[text_column_name],
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            orig = word_ids.copy()
            correction = len(examples[question_column_name][i])
            context_idx = 1 + len(tokenizer(examples[question_column_name][i], is_split_into_words = True)['input_ids'])
            for j, w in enumerate(word_ids):
                if w is not None and j >= context_idx: word_ids[j] = correction + w
            
            for w, word_idx in enumerate(word_ids):
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)

                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if args.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )
    
    eval_ids = raw_datasets["validation"]["id"]
    test_ids = raw_datasets["test"]["id"]

    train_dataset = processed_raw_datasets["train"]
    eval_dataset = processed_raw_datasets["validation"]
    test_dataset = processed_raw_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Metrics
    # metric = load_metric("seqeval")
    metric = load_metric("squad_v2")

    def get_labels(predictions, references):
        # Transform predictions and references tensos to numpy arrays
        if device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

    def compute_metrics(pred_text, ref_text, stage = 'eval'):
        predictions = [{"id": str(i), "prediction_text": pred_text[i], "no_answer_probability":0.0} for i in range(len(pred_text))]
        references = [{"id": str(i), "answers": {"text": [ref_text[i]], "answer_start": []}} for i in range(len(ref_text))]
        for r, ref in enumerate(references): 
            if not len(ref['answers']['text'][0]): references[r]['answers']['text'] = []
        metrics = metric.compute(predictions=predictions, references=references)
        metrics = {stage + '_' + k: v for k,v in metrics.items()}
        return metrics
        

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    best_f1_metric = None
    patience_counter = 0
    best_model_checkpoint = None
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break
        
        checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-' + str(completed_steps))
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
        model.eval()
        pred_text = []
        ref_text = []
        logger.info("***** Running validation *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        stage = "eval"
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            
            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)
            # import pdb; pdb.set_trace()
            ref_text.extend(get_text(batch, labels))
            pred_text.extend(get_text(batch, predictions))
            
            # preds, refs = get_labels(predictions_gathered, labels_gathered)
            # metric.add_batch(
            #     predictions=preds,
            #     references=refs,
            # )  # predictions and preferences are expected to be a nested list of labels, not label_ids

        # eval_metric = metric.compute()
        eval_metric = compute_metrics(pred_text, ref_text)
        accelerator.print(f"epoch {epoch}:", eval_metric)
        predictions = {eval_ids[i]: pred_text[i] for i in range(len(eval_ids))}
        if checkpoint_dir is not None:
            results_file = os.path.join(checkpoint_dir, "results.json" if stage is None else f"{stage}_results.json")
            logger.info(f"Saving results to {results_file}.")
            with open(results_file, "w+") as writer:
                json.dump(eval_metric, writer, indent=4)

            prediction_file = os.path.join(checkpoint_dir, "predictions.json" if stage is None else f"{stage}_predictions.json")
            logger.info(f"Saving predictions to {prediction_file}.")
            with open(prediction_file, "w+") as writer:
                writer.write(json.dumps(predictions, indent=4) + "\n")
        if best_f1_metric is None:
            best_f1_metric = eval_metric['eval_f1']
            best_model_checkpoint = checkpoint_dir
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(checkpoint_dir, save_function=accelerator.save)
        elif best_f1_metric < eval_metric['eval_f1']:
            best_model_checkpoint = checkpoint_dir
            logger.info("New best eval metric found!")
            checkpoints = [dir for dir in os.listdir(args.output_dir) if 'checkpoint-' in dir and dir != checkpoint_dir.split('/')[-1]]
            # if len(checkpoints) > 1: import pdb; pdb.set_trace()
            for checkpoint in checkpoints: 
                logger.info(f"Deleting previous checkpoint: {os.path.join(args.output_dir, checkpoint)}")
                shutil.rmtree(os.path.join(args.output_dir, checkpoint))
            patience_counter = 0
            best_f1_metric = eval_metric['eval_f1']
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(checkpoint_dir, save_function=accelerator.save)
        else: 
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Stopping training as patience of {} evals exhausted".format(args.patience))
                break
    # import pdb; pdb.set_trace()
    if best_model_checkpoint is None and args.num_train_epochs == 0:
        best_model_checkpoint = args.model_name_or_path
        epoch = -1

    config = AutoConfig.from_pretrained(best_model_checkpoint, num_labels=num_labels)
    model = AutoModelForTokenClassification.from_pretrained(
            best_model_checkpoint,
            from_tf=bool(".ckpt" in best_model_checkpoint),
            config=config,
        )
    model.to(device)
    # Final Evaluation
    if args.do_eval:
        model.eval()
        pred_text = []
        ref_text = []
        stage = "eval"
        logger.info("***** Running validation *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            
            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)
            ref_text.extend(get_text(batch, labels))
            pred_text.extend(get_text(batch, predictions))
        eval_metric = compute_metrics(pred_text, ref_text, "eval")
        accelerator.print(f"epoch {epoch}:", eval_metric)
        predictions = {eval_ids[i]: pred_text[i] for i in range(len(eval_ids))}
        if args.output_dir is not None:
            results_file = os.path.join(args.output_dir, "results.json" if stage is None else f"{stage}_results.json")
            logger.info(f"Saving results to {results_file}.")
            with open(results_file, "w+") as writer:
                json.dump(eval_metric, writer, indent=4)

            prediction_file = os.path.join(args.output_dir, "predictions.json" if stage is None else f"{stage}_predictions.json")
            logger.info(f"Saving predictions to {prediction_file}.")
            with open(prediction_file, "w+") as writer:
                writer.write(json.dumps(predictions, indent=4) + "\n")
    
    if args.do_predict:
    # Testing
        model.eval()
        pred_text = []
        ref_text = []
        logger.info("***** Running prediction *****")
        logger.info(f"  Num examples = {len(test_dataset)}")
        stage = "predict"
        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            
            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)
            ref_text.extend(get_text(batch, labels))
            pred_text.extend(get_text(batch, predictions))

        # eval_metric = metric.compute()
        test_metric = compute_metrics(pred_text, ref_text, "predict")
        accelerator.print(f"epoch {epoch}:", test_metric)
        predictions = {test_ids[i]: pred_text[i] for i in range(len(test_ids))}
        if args.output_dir is not None:
            results_file = os.path.join(args.output_dir, "results.json" if stage is None else f"{stage}_results.json")
            logger.info(f"Saving results to {results_file}.")
            with open(results_file, "w+") as writer:
                json.dump(test_metric, writer, indent=4)

            prediction_file = os.path.join(args.output_dir, "predictions.json" if stage is None else f"{stage}_predictions.json")
            logger.info(f"Saving predictions to {prediction_file}.")
            with open(prediction_file, "w+") as writer:
                writer.write(json.dumps(predictions, indent=4) + "\n")


    if args.output_dir is not None and epoch >= 0:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)


if __name__ == "__main__":
    main()
