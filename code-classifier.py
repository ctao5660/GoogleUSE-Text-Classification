import tensorflow as tf
from tensorflow import keras
import collections
import json
import keyword
import math
import StringIO
import token
import tokenize


import numpy as np
from sklearn.feature_extraction import text as sklearn_text
from sklearn import model_selection
from sklearn import svm

VOCABULARY_THRESHOLD = 5
TOKEN_NAME_VAR = u'V'

TOKEN_NAME_UNK = u'UNK'

T_MIN = 3
T_MAX = 11

K_MIN = 3

def load_data():
    with open('CodeClassifier.json') as f:
        training_data = json.load(f)
        data = collections.OrderedDict()
        count = 0
        for answer_group in training_data:
            for answer in answer_group['answers']:
                data[count] = ({
                    'source': answer['code'],
                    'class': answer_group['answer_group_index']
                })
                count+=1
        return data

def get_tokens(program):
    for token_id, token_name, _, _, _ in tokenize.generate_tokens(
        StringIO.StringIO(program).readline):
        yield (token_id, token_name)

def _is_token_ignorable(token_id, token_name):   
    return (token_id == tokenize.NL or token_id == tokenize.COMMENT
            or token_name.strip() == '')

def tokenize_for_cv(program):
    tokenized_program = []
    for token_id, token_name in get_tokens(program):
        if _is_token_ignorable(token_id, token_name):
            continue
        elif token_id == token.NAME:
            # If token_id is tokens.NAME then only add if it is a python
            # keyword. Treat all variables and methods similar.
            if token_name in keyword.kwlist:
                tokenized_program.append(token_name)
            else:
                tokenized_program.append(TOKEN_NAME_VAR)
        else:
            tokenized_program.append(token_name)

    return tokenized_program

def map_tokens_to_ids(training_data, threshold):
    # All unique tokens and number of time they occur in dataset. A token
    # can be a keyword, an identifier, a constant, an operator.
    vocabulary = collections.defaultdict(int)

    for pid in training_data:
        program = training_data[pid]['source']
        for token_id, token_name in get_tokens(program):
            if _is_token_ignorable(token_id, token_name):
                continue
            # If token_id is tokens.NAME then only add if it is a python
            # keyword.
            elif token_id == token.NAME:
                if token_name in keyword.kwlist:
                    vocabulary[token_name] = vocabulary[token_name] + 1
            else:
                vocabulary[token_name] = vocabulary[token_name] + 1

    # Consider only those tokens which occur for more than threshold times
    # in entire dataset.
    valid_tokens = [
        unicode(k) for k, v in vocabulary.iteritems() if v > threshold]
    token_to_id = dict(zip(valid_tokens, xrange(0, len(valid_tokens))))

    # Add 'UNK' in token_to_id. This will be used to replace any token
    # occurring in program which is not in valid_token.
    token_to_id[TOKEN_NAME_UNK] = len(token_to_id)

    # Add 'V' in token_to_id. This token will be used to replace all
    # variables and methods in program.
    token_to_id[TOKEN_NAME_VAR] = len(token_to_id)
    return token_to_id


def tokenize_data(training_data, threshold=VOCABULARY_THRESHOLD):
    token_to_id = map_tokens_to_ids(training_data, threshold)

    # Tokenize all programs in dataset.
    for program_id in training_data:
        program = training_data[program_id]['source']
        tokenized_program = []
        for token_id, token_name in get_tokens(program):
            if _is_token_ignorable(token_id, token_name):
                continue
            elif token_id == token.NAME and token_name not in keyword.kwlist:
                # If token_id is tokens.NAME and it is not a python keyword
                # then it is a variable or method.
                # Treat all methods and variables same.
                tokenized_program.append(TOKEN_NAME_VAR)
            else:
                # Add token only if it present in token_to_id. Otherwise replace
                # the token with TOKEN_NAME_UNK.
                if token_name in token_to_id:
                    tokenized_program.append(token_name)
                else:
                    tokenized_program.append(TOKEN_NAME_UNK)

        training_data[program_id]['tokens'] = tokenized_program

    return training_data, token_to_id


        
def model():
    model = keras.Sequential()

class CodeClassifier():

    def __init__(self):
        self.model = model()
        self.data = load_data()
    
    def train(self):
        data, token_to_id = tokenize_data(self.data)
        count_vector = sklearn_text.CountVectorizer(
            tokenizer=tokenize_for_cv, min_df=5)
        # TODO: Vectorize Data, Build model
test = CodeClassifier()
test.train()
