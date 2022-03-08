import json
import random
import os
import logging
import pickle
import string
import re
from pathlib import Path
from collections import Counter, OrderedDict, defaultdict as ddict
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from util import read_squad

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet 

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
              'ours', 'ourselves', 'you', 'your', 'yours', 
              'yourself', 'yourselves', 'he', 'him', 'his', 
              'himself', 'she', 'her', 'hers', 'herself', 
              'it', 'its', 'itself', 'they', 'them', 'their', 
              'theirs', 'themselves', 'what', 'which', 'who', 
              'whom', 'this', 'that', 'these', 'those', 'am', 
              'is', 'are', 'was', 'were', 'be', 'been', 'being', 
              'have', 'has', 'had', 'having', 'do', 'does', 'did',
              'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
              'because', 'as', 'until', 'while', 'of', 'at', 
              'by', 'for', 'with', 'about', 'against', 'between',
              'into', 'through', 'during', 'before', 'after', 
              'above', 'below', 'to', 'from', 'up', 'down', 'in',
              'out', 'on', 'off', 'over', 'under', 'again', 
              'further', 'then', 'once', 'here', 'there', 'when', 
              'where', 'why', 'how', 'all', 'any', 'both', 'each', 
              'few', 'more', 'most', 'other', 'some', 'such', 'no', 
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
              'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', '']

def synonym_replacement(words, n):
    new_words = words.copy()
    
    # Skip the word if it is in the stop words or capitalized
    random_word_list = list(set([word for word in words if word not in stop_words and word[0].isupper() == False]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            #print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def augment_dataset(dataset_dict, dataset_name):

    aug_qs = []
    for i in range(len(dataset_dict['question'])):
        new_pair = []

        question = dataset_dict['question'][i]
        context  = dataset_dict['context'][i]    
        id_      = dataset_dict['id'][i]
        answer   = dataset_dict['answer'][i]

        question = question.split(' ')
        output = synonym_replacement(question, 3)
        new_question = ' '.join(output)
        
        dataset_dict['question'].append(new_question)
        dataset_dict['context'].append(context)
        print(id_)
        dataset_dict['id'].append(id_ + '_augment')
        dataset_dict['answer'].append(answer)
        print(id_)
        print(dataset_name)

    return dataset_dict






