#!/usr/local/bin/python3
import tensorflow_hub as hub
import pandas as pd
from sklearn import preprocessing
import keras 
import numpy as np
import yaml
import random
url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
embed = hub.Module(url)
def load_data():
    x, y = [], []
    with open('string_classifier_test.yaml', 'r') as f:
        doc = yaml.load(f)
        interactions = doc['states']['Home']['interaction']

        for answer_group in interactions['answer_groups'][1:]:
            label = answer_group['outcome']['feedback'][0]
            for rule in answer_group['rule_specs']:
                if 'inputs' in rule and 'training_data' in rule['inputs']:
                    for answer in rule['inputs']['training_data']:
                        x.append(answer)
                        y.append(label)
    combined = list(zip(x,y))
    random.shuffle(combined)
    shuffledX, shuffledY = zip(*combined)
    splitX = int(len(shuffledX) * .8)
    splitY = int(len(shuffledY) * .8)
    return shuffledX[:splitX], shuffledY[:splitY], shuffledX[splitX:], shuffledY[splitY:]

x_train, y_train, x_test, y_test = load_data()
print(x_train)
    


