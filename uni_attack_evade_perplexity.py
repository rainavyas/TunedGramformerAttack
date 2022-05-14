'''
Perform concatenation adversarial attack on 
GEC system, with aim of finding universal adversarial phrase
that minimises average number of edits between original and 
predicted gec sentence and ensures attack phrase keeps perplexity below detection threshold.
'''
import sys
import os
import argparse
import torch
from gec_tools import get_sentences, correct, count_edits
from Seq2seq import Seq2seq
from eval_uni_attack import set_seeds
import json
from datetime import date
from uni_attack import get_avg
from perplexity import perplexity
from statistics import mean
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def is_perp_less_than_thresh(sentences, attack_phrase, thresh):
    '''
        Return True if the average dataset perplexity is less than threshold
    '''
    perp_tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')
    perp_model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    perps = []
    for sent in sentences:
        sent = sent + ' ' + attack_phrase
        try:
            perp = perplexity(sent, perp_tokenizer, perp_model)
            perps.append(perp)
        except:
            continue
    avg_perp = mean(perps)
    if avg_perp < thresh:
        return True
    return False

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='Path to input data')
    commandLineParser.add_argument('MODEL', type=str, help='Path to Gramformer model')
    commandLineParser.add_argument('VOCAB', type=str, help='ASR vocab file')
    commandLineParser.add_argument('LOG', type=str, help='Specify txt file to log iteratively better words')
    commandLineParser.add_argument('--prev_attack', type=str, default='', help='greedy universal attack phrase')
    commandLineParser.add_argument('--num_points', type=int, default=1000, help='Number of training data points to consider')
    commandLineParser.add_argument('--search_size', type=int, default=400, help='Number of words to check')
    commandLineParser.add_argument('--start', type=int, default=0, help='Vocab batch number')
    commandLineParser.add_argument('--perp_thresh', type=float, default=0, help='Perplexity Detector threshold')
    commandLineParser.add_argument('--seed', type=int, default=1, help='reproducibility')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/uni_attack_evade_perplexity.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    set_seeds(args.seed)

    # Load Model
    model = Seq2seq()
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.eval()

    # Load input sentences
    _, sentences = get_sentences(args.IN, num=args.num_points)

    # Get list of words to try
    with open(args.VOCAB, 'r') as f:
        test_words = json.loads(f.read())
    test_words = [str(word).lower() for word in test_words]

    # Keep only selected batch of words
    start_index = args.start*args.search_size
    test_words = test_words[start_index:start_index+args.search_size]

    # Add blank word at beginning of list
    # test_words = ['']+test_words

    # Initialise empty log file
    with open(args.LOG, 'w') as f:
        f.write("Logged on "+ str(date.today()))

    best = ('none', 1000)
    for word in test_words:
        attack_phrase = args.prev_attack + ' ' + word + '.'
        if not is_perp_less_than_thresh(sentences, attack_phrase, args.perp_thresh):
            continue
        edits_avg = get_avg(model, sentences, attack_phrase)
        # print(word, edits_avg) # temp debug

        if edits_avg < best[1]:
            best = (word, edits_avg)
            # Write to log
            with open(args.LOG, 'a') as f:
                out = '\n'+best[0]+" "+str(best[1])
                f.write(out)
