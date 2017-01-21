#!/usr/bin/python
import re
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
import sys
import getopt
import numpy as np
import scipy as sp
from pprint import pprint
from collections import Counter

np.set_printoptions(threshold=np.inf)

def build_LM(in_file):
    """
    build language models for each label
    each line in in_file contains a label and a string separated by a space
    """
    print('building language models...')
    # This is an empty method
    # Pls implement your code in below
    with open(in_file) as f:
        raw_data = f.read().splitlines()

    # Extract labels
    raw_labels = []
    for i, v in enumerate(raw_data):
        line_split = v.split(' ', 1)
        raw_labels.append(line_split[0])
        raw_data[i] = line_split[1]

    raw_labels = np.array(raw_labels)
    lm_labels = list(set(raw_labels))
    
    # Generate 4-grams
    raw_data = [list(ngrams(line, 4, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')) for line in raw_data]
    raw_data = np.array(raw_data)

    # Populate sorted vocab
    vocab = set()
    for line in raw_data:
        vocab.update(set(line))

    vocab = list(vocab)
    vocab.sort()
    reverse_vocab = {v:i for i,v in enumerate(vocab)}

    # Transform data into count vectors
    data = np.zeros((len(lm_labels), len(vocab)))
    for i, label in enumerate(lm_labels):
        indices = np.where(raw_labels == label)
        
        for line in raw_data[indices]:
            c = Counter(line)
            for ngram in c:
                data[i,reverse_vocab[ngram]] += c[ngram]

        # Add-one smoothing
        for j, val in enumerate(data[i]):
            if val == 0:
                data[i,j] += 1

    del raw_data, raw_labels

    # Convert data into probability vectors
    totals = [sum(line) for line in data]
    lm_data = []
    for i, line in enumerate(data):
        lm_data.append({vocab[vi] : count / totals[i] for vi, count in enumerate(line)})
    
    return {k:v for k,v in zip(lm_labels, lm_data)}
            
    
def test_LM(in_file, out_file, LM):
    """
    test the language models on new strings
    each line of in_file contains a string
    you should print the most probable label for each string into out_file
    """
    print("testing language models...")
    # This is an empty method
    # Pls implement your code in below
    with open(in_file) as f:
        raw_test_data = f.read().splitlines()


def usage():
    print("usage: " + sys.argv[0] + " -b input-file-for-building-LM -t input-file-for-testing-LM -o output-file")

input_file_b = input_file_t = output_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], 'b:t:o:')
except getopt.GetoptError as err:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == '-b':
        input_file_b = a
    elif o == '-t':
        input_file_t = a
    elif o == '-o':
        output_file = a
    else:
        assert False, "unhandled option"
if input_file_b == None or input_file_t == None or output_file == None:
    usage()
    sys.exit(2)

LM = build_LM(input_file_b)
print(LM)
test_LM(input_file_t, output_file, LM)
