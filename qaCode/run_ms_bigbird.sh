#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python run_ner_no_trainer.py \
  --model_name_or_path Results/bigbird-multispan-mod \
  --tokenizer_name google/bigbird-roberta-base \
  --train_file /home/arprasad/AllData/MultiSpanDataset/final-AMI-ms-train.json\
  --validation_file /home/arprasad/AllData/MultiSpanDataset/final-AMI-ms-dev.json \
  --test_file /home/arprasad/AllData/MultiSpanDataset/final-AMI-ms-test.json \
  --do_eval \
  --do_predict \
  --question_column_name question \
  --text_column_name context \
  --label_column_name labels \
  --label_all_tokens \
  --per_device_train_batch_size 5 \
  --learning_rate 5e-5 \
  --num_train_epochs 15 \
  --max_length 1536 \
  --pad_to_max_length \
  --output_dir Results/finetuned/bigbird-multispan-silver-ft/
