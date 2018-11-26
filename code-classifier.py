import tensorflow as tf
from tensorflow import keras

import numpy as np
import json
def load_data():
    with open('CodeClassifier.json') as f:
        training_data = json.load(f)
        data = collections.OrderedDict()
        count = 0
        for answer_group in training_data:
            for answer in answer_group['answers']:
                data[count] = {
                    'source': answer['code'],
                    'class': answer_group['answer_group_index']
                }
                count += 1
