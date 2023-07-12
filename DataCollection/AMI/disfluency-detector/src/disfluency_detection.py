import time
import os
import re
import torch
import torch.optim.lr_scheduler

from parse_nk import use_cuda
from parse_nk import NKChartParser

DISFLUENCY_WORD_LIST = ['i', 'think', 'we', 'mean', 'like', 'so', 'uh', 'well', 'you', 'know', 'oh', 'yeah', 'hmm', 'um', 'uhm', 'huh']

def process_sentences(raw_sentences):
    '''
    Processes the sentences to make them ready for parser.

        Parameters
        ----------
            raw_sentences; a list of sentences.

        Returns
        -------
            sentences_processed: a list of sentences with punctuation, etc removed.
    '''
    sentences_processed = []
    for sent in raw_sentences:
        # We can't just remove all the punctuations, 
        # because that will remove ' in words like I'll.
        # We will have to remove them case by case.
        cur_sent = sent.replace("_", " ")
        cur_sent = re.sub("[ ]{2,}", " ", cur_sent)
        cur_sent = cur_sent.replace(".", "")
        cur_sent = cur_sent.replace(",", "")
        cur_sent = cur_sent.replace(";", "")
        cur_sent = cur_sent.replace("?", "")
        cur_sent = cur_sent.replace("!", "")
        cur_sent = cur_sent.replace(":", "")
        cur_sent = cur_sent.replace("\"", "")
        cur_sent = cur_sent.replace("'re", " 're")
        cur_sent = cur_sent.replace("'ve", " 've")
        cur_sent = cur_sent.replace("n't", " n't")
        cur_sent = cur_sent.replace("'ll", " 'll")
        cur_sent = cur_sent.replace("'d", " 'd")
        cur_sent = cur_sent.replace("'m", " 'm")
        cur_sent = cur_sent.replace("'s", " 's")
        cur_sent = cur_sent.strip()
        cur_sent = cur_sent.lower()

        sentences_processed.append(cur_sent)

    return sentences_processed

def is_disfluent_word(token):
    if token in DISFLUENCY_WORD_LIST:
        return True
    return False

# Creates fluent tags
def fluent(tokens):
    '''
    Returns the token with fluent label "_" attached to it.

        Parameters
        ----------
            tokens

        Returns
        -------
            disfluency labels: tokens + labels
    '''
    leaves_tags = [t.replace(")","")+" _" for t in tokens if ")" in t]      
    return " ".join(leaves_tags)

# Creates disfluent tags
def disfluent(tokens):
    '''
    Returns the token with disfluent label "E" attached to it.

        Parameters
        ----------
            tokens

        Returns
        -------
            disfluency labels: tokens + labels
    '''
    # remove first and last brackets
    tokens, tokens[-1] = tokens[1:], tokens[-1][:-1]
    open_bracket, close_bracket, pointer = 0, 0, 0      
    df_region = False
    is_prn_intj = False
    tags = []
    while pointer < len(tokens):
        open_bracket += tokens[pointer].count("(")                
        close_bracket += tokens[pointer].count(")")
        if "(INTJ" in tokens[pointer] or "(PRN" in tokens[pointer] or "(EDITED" in tokens[pointer]:  
            open_bracket, close_bracket = 1, 0             
            df_region = True
            if "(INTJ" in tokens[pointer] or "(PRN" in tokens[pointer]:
                is_prn_intj = True
                
        elif ")" in tokens[pointer]:
            # label = "E" if df_region else "_"  
            if (df_region and not is_prn_intj) or (df_region and is_disfluent_word(tokens[pointer].strip(')'))):
                label = "E"
            else:
                label = "_"
            tags.append(
                (tokens[pointer].replace(")", ""), label)
                )                 
        if all(
            (close_bracket,
            open_bracket == close_bracket)
            ):
            open_bracket, close_bracket = 0, 0
            df_region = False   
            is_prn_intj = False         

        pointer += 1
    return " ".join(list(map(lambda t: " ".join(t), tags)))

def torch_load(load_path):
    if use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)

# This funstion calls the parser (from the saved checkpoint) and then disfluency tagger.
def run_parse(model_path_base, sentences):
    '''
    Returns the parsed tree and.

        Parameters:
            model_path_base: the path to the folder where the models and vocabulary are located
            sentences: a list of sentences

        Returns
        -------
            parsed trees
            disfluency labels
    '''

    # TO DO: Ideally this should not be hard coded and should be passed as a parameter.
    batch_size = 100
    # These are hard coded in the saved model. Please do not change!
    SAVED_CHECK_POINT = 'swbd_fisher_bert_Edev.0.9078.pt'
    saved_checkpoint_path = os.path.join(model_path_base, SAVED_CHECK_POINT)
    BERT_MODEL_NAME = 'bert-base-uncased.tar.gz'

    print("Loading model from {}...".format(model_path_base))
    assert saved_checkpoint_path.endswith(".pt"), "Only pytorch savefiles supported"

    info = torch_load(saved_checkpoint_path)
    assert 'hparams' in info['spec'], "Older savefiles not supported"

    # monkey patch the path to the model (really bad solution! but the path has been hard coded into the saved checkpoint and there is no other way to fix it)
    info['spec']['hparams']['bert_model'] = os.path.join(model_path_base, BERT_MODEL_NAME)

    parser = NKChartParser.from_spec(info['spec'], info['state_dict'])

    # print("Parsing sentences...")
    sentences = [sentence.split() for sentence in sentences]

    # Tags are not available when parsing from raw text, so use a dummy tag
    if 'UNK' in parser.tag_vocab.indices:
        dummy_tag = 'UNK'
    else:
        dummy_tag = parser.tag_vocab.value(0)

    start_time = time.time()

    all_predicted = []
    for start_index in range(0, len(sentences), batch_size):
        subbatch_sentences = sentences[start_index:start_index+batch_size]

        subbatch_sentences = [[(dummy_tag, word) for word in sentence] for sentence in subbatch_sentences]
        
        predicted, _ = parser.parse_batch(subbatch_sentences)
        del _
        all_predicted.extend([p.convert() for p in predicted])
        
        

    # print("Sentences were parsed in %s seconds." % (time.time() - start_time))

    parse_trees, df_labels = [], []
    for tree in all_predicted:          
        linear_tree = tree.linearize()
        parse_trees.append(linear_tree)

        tokens = linear_tree.split()
        # disfluencies are dominated by EDITED nodes in parse trees
        if "INTJ" not in linear_tree and "PRN" not in linear_tree: 
            df_labels.append(fluent(tokens))
        else:
            df_labels.append(disfluent(tokens))

    # output_path = 'data/parsed_sentences.txt'
    # with open(output_path, 'w') as output_file:
    #     for tree in all_predicted:
    #         output_file.write("{}\n".format(tree.linearize()))
    # # print("Output written to:", output_path)

    # result_path = 'data/disluencies.txt'
    # with open(result_path, 'w') as result_file:
    #     for line in df_labels:
    #         result_file.write(line)
    #         result_file.write('\n')
    #     # print("Result written to:", result_path)

    return parse_trees, df_labels
