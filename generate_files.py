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
    model.load_state_dict(torch.load(args.MODEL, map_location=torch.device('cpu')))
    model.eval()

    # Load input sentences
    _, sentences = get_sentences(args.INPUT_PATH)

    corrections = []
    for s in sentences:
        correction = correct(model, s)
        corrections.append(correction)
    
    # save files
    with open(f'{args.OUT}/sentences.inc', 'a') as f:
        sentences = [s+'\n' for s in sentences]
        f.writelines(sentences)
    with open(f'{args.OUT}/sentences.pred', 'a') as f:
        corrections = [c+'\n' for c in corrections]
        f.writelines(corrections)   
        
