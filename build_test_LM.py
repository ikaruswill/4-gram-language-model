#!/usr/bin/python
import re
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
from nltk.util import ngrams
import sys
import getopt

def build_LM(in_file):
    """
    build language models for each label
    each line in in_file contains a label and a string separated by a space
    """
    print('building language models...')
    # This is an empty method
    # Pls implement your code in below
    with open(in_file) as f:
        data = f.read().splitlines()

    # Extract labels
    labels = []
    for i in range(len(data)):
        line_split = data[i].split()
        labels.append(line_split[0])
        data[i] = ' '.join(line_split[1:])

    # Generate 4-grams
    data = [list(ngrams(line, 4, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')) for line in data]

    # Populate vocab
    vocab = set()
    for line in data:
        vocab.update(set(line))


    
def test_LM(in_file, out_file, LM):
    """
    test the language models on new strings
    each line of in_file contains a string
    you should print the most probable label for each string into out_file
    """
    print("testing language models...")
    # This is an empty method
    # Pls implement your code in below

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
# test_LM(input_file_t, output_file, LM)
