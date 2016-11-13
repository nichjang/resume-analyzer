import numpy as np
import re
import itertools
import nltk
from string import digits
from collections import Counter


def clean_str(string):
    """
    Regex used to seperate compound words
    Original from : https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py

    Input: string
    Output: string without compound words
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(strong_file, weak_file):
    """
    Loads examples from files. Splits data into strong and weak classifcations.
    
    Input: files of strong and weak words
    Output: list of words and classifications
    """
    # Load data from files
    strong_examples = list(open(strong_file, "r").readlines())
    strong_examples = [s.strip() for s in strong_examples]
    weak_examples = list(open(weak_file, "r").readlines())
    weak_examples = [s.strip() for s in weak_examples]
    # Split by words
    split_text_s = strong_examples
    split_text_s = [nltk.pos_tag(nltk.word_tokenize(clean_str(x))) for x in split_text_s]
    split_text_w = weak_examples
    split_text_w = [nltk.pos_tag(nltk.word_tokenize(clean_str(x))) for x in split_text_w]
    split_text_combined = split_text_s + split_text_w
    updated_list = []
    for item in split_text_combined:
        sub_list = []
        for subitem in item:
            sub_list.append(subitem[1])
        sub_string = " ".join(sub_list)
        updated_list.append(sub_string)    
    # Generate labels
    strong_labels = [[0, 1] for _ in split_text_s]
    weak_labels = [[1, 0] for _ in split_text_w]
    y = np.concatenate([strong_labels, weak_labels], 0)
    return [updated_list, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size) #min of batch size or data size
            yield shuffled_data[start_index:end_index] #yield until out of range
