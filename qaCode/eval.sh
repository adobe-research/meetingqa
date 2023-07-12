#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python run_qa.py \
  --model_name_or_path deepset/roberta-large-squad2 \
  --train_file /home/arprasad/AllData/Dataset/train-med.json \
  --validation_file /home/arprasad/AllData/Dataset/dev-med.json \
  --test_file /home/arprasad/AllData/Dataset/final-AMI-test.json \
  --do_predict \
  --version_2_with_negative \
  --per_device_eval_batch_size 64 \
  --load_best_model_at_end True \
  --save_strategy no \
  --max_seq_length 512 \
  --doc_stride 128 \
  --max_answer_length 400 \
  --output_dir Results/zeroshot/roberta-large-squadv2-zero/
  # --do_eval \
  # Results/longformer-squadv2-med/
