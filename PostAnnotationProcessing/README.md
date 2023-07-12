# Silver Data Annotation and Post Annotation Processing

## Silver Data Annotation

* Step 1: We use the structure and nice properties of MediaSum Interviews to automatically identify questions and annotate rough answers (silver data annotation). This is implemented in `generate_synthetic_qa.py` which takes in interview files from `../ProcessedTranscripts/MediaSumInterviews` and stores the silver annotated files in `../AllData/Annotated`.

* Step 2: Next, we extract location-based snippets from the (annotated) meeting transcripts and apply data augmentation using `create_snippets.py` for single-span models which are stored in `AllData/Snippets` by default and can be changed accordingly (using `dest_path`). This file also contains a bunch of helper functions used by other code files in this folder. Different augmentations and their relative usage can be controlled by setting the appropritate variables. Input arguments include `--answer-type` which can be set to `cont`, `concat`, and `multispans` corresponding to finding a single continuous answerspan based on the annotated start and end points, concatenating sentences listed in the annotated answer span, and returning a list of spans in the snippets respectively. Its counterpart for multi-span models (with multi-span compatible augmentation) is `create_multi_span_snippets.py`.

* Step 3: The above step generates a large number of snippets, each corresponding to a question stored separately. These are combined to create a dataset for training a QA model using `create_qa_dataset.py` compatible with SQuAD v2.0 format returning a '.json' file. For multispan models which are based on token-classification, the outputs of this file are used as inputs in `create_multi_span_dataset.py` which contains tokens for question and context along with `I_ANSWER/O` labels for if the token is inside or outside the answer span respectively.

## Human Annotated Data

When human annotated data is available, we combine aforementioned step 2 and 3 using the file `create_human_testset.py`. The outputs from this file can be used as is for training or evaluating single-span models. The input to this file comes from `../Processed/Transcripts` and the output dir is set using `dest_path`. For multi-span models, like in step 3, we the generated `.json` dataset files are input to `create_multi_span_dataset.py`. Note: For the purposes of curating and cleaning the dataset we use `process_all_teleus.ipynb` in the beginning (one-time to generate data splits and meta informaton).

## Miscellaneous

To evaluate trained models on behance or Blink data create dataset files using `convert_behance_dataset.py` and `create_blink_eval.py` respectively. To train models for the alternate task of classifying questions as answerable or unanswerable, use the dataset files for single span models as input to `create_answerable_dataset.py` which prepares the dataset for sequence classification task (0: unanswerable; 1: answerable).

**Environments:** For all these steps or tasks (sections) use the *datasets* environment.




