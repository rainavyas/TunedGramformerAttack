'''
At input expect a SOURCE, REF and PRED file:

text1
text2
.
.
.
Use the output of align_preds.py to ensure text is aligned

Output the errant edit type distribution; for each edit type give the following:

Ref count, Pred Total Count, Pred Correct Count, Pred Inserted Count 

'''
import sys
import os
import argparse
from gec_tools import return_edits
from collections import defaultdict

def update_edit_types(ref_edits, pred_edits, ref_count, pred_total, pred_correct, pred_insert, pred_del):
    '''
    Collect all edit type information
    '''
    ref_edit_strs = [e.o_str+' -> '+e.c_str for e in ref_edits]
    pred_edit_strs = [e.o_str+' -> '+e.c_str for e in pred_edits]

    for e in ref_edits:
        ref_count[e.type] += 1

        curr_str = e.o_str+' -> '+e.c_str
        if curr_str not in pred_edit_strs:
            pred_del[e.type] += 1
    
    for e in pred_edits:
        pred_total[e.type] += 1
    
        curr_str = e.o_str+' -> '+e.c_str
        if curr_str in ref_edit_strs:
            pred_correct[e.type] += 1
        else:
            pred_insert[e.type] += 1


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('SOURCE', type=str, help='Path to source data')
    commandLineParser.add_argument('REF', type=str, help='Path to correct reference data')
    commandLineParser.add_argument('PRED', type=str, help='Path to prediction data')
    commandLineParser.add_argument('OUT', type=str, help='Path to save edit type information')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_error_dist.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n') 

    # Get sentences
    with open(args.SOURCE, 'r') as f:
        source_texts = f.readlines(f)
    with open(args.REF, 'r') as f:
        ref_texts = f.readlines(f)
    with open(args.PRED, 'r') as f:
        pred_texts = f.readlines(f)


    # Get the edit types dicts
    ref_count = defaultdict(int)
    pred_total = defaultdict(int)
    pred_correct = defaultdict(int)
    pred_insert = defaultdict(int)
    pred_del = defaultdict(int)

    for i, (s, r, p) in enumerate(zip(source_texts, ref_texts, pred_texts)):
        print(f'On {i}/{len(source_texts)}')
        ref_edits = return_edits(s, r)
        pred_edits = return_edits(s, p)
        update_edit_types(ref_edits, pred_edits, ref_count, pred_total, pred_correct, pred_insert, pred_del)

    # Save edit type distribution to file
    texts = ['Type Ref-Count Pred-Total Pred-Correct Pred-Insert Pred-Delete']
    for edit_type in sorted(list(ref_count.keys())):
        texts.append(f'\n{edit_type} {ref_count[edit_type]} {pred_total[edit_type]} {pred_correct[edit_type]} {pred_insert[edit_type]}  {pred_del[edit_type]}')
    with open(args.OUT, 'w') as f:
            f.writelines(texts)