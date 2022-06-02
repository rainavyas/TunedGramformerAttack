'''
Specifically for BEA-19

Tokenizes the prediction file and generate submission file
'''
import argparse
import os
import sys
from spacy.lang.en import English
nlp = English()

if __name__ == '__main__':

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('IN', type=str, help='path to input .pred file')
    commandLineParser.add_argument('OUT', type=str, help='Directory to save output file')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/prepare_submission.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')
    
    # Load the .pred file
    with open(args.IN, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip('\n') for l in lines]

    # Tokenize each sentences
    tokenized_sents = []
    tokenizer = nlp.tokenizer
    for l in lines:
        tokens = tokenizer(l)
        token_strs = [tokens[i].text for i in range(len(tokens))]
        tokenized_sents.append(' '.join(token_strs))
    
    # Save
    tokenized_sents = [t+'\n' for t in tokenized_sents]
    with open(f'{args.OUT}/submission.txt', 'a') as f:
        f.writelines(tokenized_sents)

