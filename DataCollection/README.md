# Data Collection (Pre-Annotation)
This repository contains scripts that process transcript data from various sources such as AMI/ICSI and YouTube. Note: ICSI is processed exactly same as AMI with appropriate options selected for source and destination files. 

## AMI Processing
* Step 1: Download the [AMI dataset](http://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip) (manual transcriptions) which contains XML files.
* Step 2: Parse and process XML files using `process_transcripts_xml_data.py`. It contains two variables `words_path` which contains the source XML files corresponding to the transcribed words (XML files) and stores the parsed transcript in the location indicated by `dest_path` (typically in `../ProcessedTranscripts`).
* Step 3: AMI data contains several disfluencies, which can be removed using the `disfluency_detector` system available [here](https://git.corp.adobe.com/deilamsa/disfluency-detector). Follow the setup instructions in the repository and then replace `src/` folder with the supplied `disfluency-detector/src` from this repository. This primarily modifies `src/test.py` and stores the disfluency removed punctuated transcripts in `../ProcessedTranscripts`.
* Step 4: Prepare (disfluency removed) files to be sent for human annotation using `prepare_annotation.py` which creates `.json` files. For a more reader-friendly `.tsv` format also use `create_conversation.py`.

**Environments:** *datasets*. Refer to `../requirements/README.md`.

## YouTube Processing

* Step 1: Crawl desired videos from YouTube using `execute_download.sh` (uses `download_YT_videos.py`) which requires a file containing video-ids (can be obtained using select keywords via `keywords.py`). This downloads transcripts and the corresponding audio files. Alternatively, we will need transcripts with timing information which can be downloaded using `dowload_YT_with_time.py`. All these files/folders stores in `../YouTube`.
* Step 2: Obtain speaker information using open-source [diarization package](https://github.com/pyannote/pyannote-audio) by running `process_diarization.py`. 
* Step 3: Speaker information (speaker-time assignemnts) is combined with timing information in transcripts, in order to assign which sentence was uttered by which speaker. This is done using `align_times.py`. 
* Step 4: Most of the downloaded transcripts from YouTube are not punctuated. In order to punctuate them use `repunctuate.py` which uses an open-source repunctuation package [rpunct](https://github.com/Felflare/rpunct). Finally, align the punctuated sentence with the speaker information using `execute_repunctuate.sh` which works in batches using `align_punctuate.py`. The final files are present in `../YouTube/processed`. 
* Step 5: Convert the transcripts into Blink-like format using `convert_YT_to_blink.py` which stores the outputs in `ProcessedTranscripts/YouTube`.

**Environments:** *download*, *pyannote*, *rpunct*, *datasets* correspondingly for different steps. Refer to `../requirements/README.md`.

## MediaSum Processing
* Step 1: Download the dataset from the [paper repository](https://github.com/zcgzcgzcg1/MediaSum).
* Step 2: Convert the dataset to Blink-like format using `process_mediaSum_data.py`, files stored in `../ProcessedTranscripts/MediaSumInterviews`.

**Environments:** *datasets*. Refer to `../requirements/README.md`.

