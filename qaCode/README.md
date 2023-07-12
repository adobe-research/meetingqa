# Experiment Setup and Code

## Single-Span Models
For span-based extractive models, the base code is present in `run_qa.py` which uses `trainer_qa.py` and `utils_qa.py`. We train different Roberta (or Deberta), Longformer, Bigbird models via `run_qa.sh`, `run_longformer.sh`, `run_bigbird.sh` respectively. Appropritate training hyperparameters as well as the data files are set using the `.sh` files. 

For generative models, the base code is present in `run_seq2seq_qa.py` which uses `trainer_seq2seq_qa.py`. We can train different LongT5, and UnifiedQA models using `run_longT5.sh`, `run_unified.sh` (set training files and appropritate hyperparameters). Since these are generative models we cannot guarentee faithfullness and hence we leave them underexplored. 

## Multi-Span Models
For multi-span models, we use the token classification head, where tokens in the context are marked `I_ANSWER` if it is part of the answer span and `O` otherwise. The base code for implementing this is present in `run_ner_no_trainer.py`.  We train different Roberta (or Deberta), Longformer, Bigbird models via `run_multispan.sh`, `run_ms_longformer.sh`, and `run_ms_bigbird.sh` respectively by setting appropriate hyperparameters and data files. We did not explore T5 or UnifiedQA models in this setup.

## Detect Answerable/Unanswerable
For a simpler task of detecting whether a question (given a context) is answerable or not, we use models for sequence classification task. The base code is present in `run_classification.py` and can be trained by setting the hyperparameters in `run_classify.sh`.

## Instruction-tuned Models
We explore Instruction-tuned (FLAN-T5) models in a zero-shot manner in multiple settings, the code for which is present in `run_instruct_models.py` along with helper code in `instruct_utils.py`. This code directly outputs `.json` files containing predictions in appropriately set `dest_path`.

## Evaluation
Once trained, models can be evaluated to generate a file containing all predictions using `eval.sh` and `eval_multispan.sh` respectively. Note: please be careful to set the hyperparameters appropriately to match the training. This will create a file: `[output_dir]/predict_predictions.json` containing a dictionary with "id" (identifier obtained from the data file) as key and the prediction as value. This predictions file can be used to compute fine-grained scores using `custom_evaluate.py` by running the command (for AMI): 
```
python custom_evaluate.py --test-file ../AllData/Dataset/final-AMI-test-meta.json --test-predictions [output dir]/predict_predictions.json
```
Note that additional command-line argument `--pred-ref-type` is used when the reference and the predictions are in the same format (as opposed to the aformentioned format for predictions).

**Environments:** Run these experiments using the *qa* conda enviromnemnt.
