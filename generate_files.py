'''
This is specifically for CoNLL-14.

Input: input sentences text file with structure:
    id1 sentence1

Output: 
    - sentences.inc
    - sentences.pred
'''

import argparse
import os
import sys
import torch
from Seq2seq import Seq2seq
from gec_tools import get_sentences, correct

from spacy.lang.en import English
nlp = English()

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('INPUT_PATH', type=str, help='path to input file')
    commandLineParser.add_argument('MODEL_PATH', type=str, help='Gramformer model file path')
    commandLineParser.add_argument('OUT', type=str, help='Directory to save output files')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/generate_files.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load Model
    model = Seq2seq()
    model.load_state_dict(torch.load(args.MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    # Load input sentences
    _, sentences = get_sentences(args.INPUT_PATH, remove_end_space=False)

    corrections = []
    for i,s in enumerate(sentences):
        print(f'On {i}/{len(sentences)}')
        try:
            correction = correct(model, s)
            corrections.append(correction)
        except:
            corrections.append('')
    
    # tokenize corrections
    tok_corrs = []
    tokenizer = nlp.tokenizer
    for l in corrections:
        tokens = tokenizer(l)
        token_strs = [tokens[i].text for i in range(len(tokens))]
        tok_corrs.append(' '.join(token_strs))
    corrections = tok_corrs[:]
    
    # save files
    with open(f'{args.OUT}/sentences.inc', 'w') as f:
        sentences = [s+'\n' for s in sentences]
        f.writelines(sentences)
    with open(f'{args.OUT}/sentences.pred', 'w') as f:
        corrections = [c+'\n' for c in corrections]
        f.writelines(corrections)   
        
