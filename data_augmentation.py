import pandas as pd
import numpy as np
import copy
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from random import randint
import nltk.data
import string
import random
CHANGES = dict(zip([" game ", " was "], [" lame ", " sos "]))


def augment_dataset_dict_depr(dataset_dict, changes = None):
	changes=CHANGES
	original_df = pd.DataFrame({x: dataset_dict[x] for x in dataset_dict})
	df= copy.deepcopy(original_df)
	df['context'] = df.context.str.strip().replace(changes, regex=True)
	new_dataset_dict = pd.concat([original_df, df[[i for i in dataset_dict.keys()]]]).to_dict(orient='list')
	#TODO : Correct for start end shifts, do not modify answers
	#df['start_char'] = df.answer.apply(lambda x: x['answer_start'][0])
	#df['end_char'] = df['start_char'] + df.answer.apply(lambda x: len(x['text'][0]))
	#df['final_answer'] = [A[B:C] for A, B, C in zip(df.context, df['start_char'], df['end_char'])]
	#df['question'] = df.question.str.strip().replace(changes, regex=True)
	##df['new_context'] = df.context.str.strip().replace(changes,regex=True)
	##df['new_answer'] = [A[B:C] for A, B, C in zip(df['new_context'], df['start_char'],df['end_char'])]
	#new_dataset_dict = pd.concat([original_df , df[[i for i in dataset_dict.keys() if i != 'label']]]).to_dict(orient ='list')
	#new_dataset_dict['label'] = dataset_dict['label']
	return new_dataset_dict




def augment_dataset_dict(dataset_dict, p_sr=0.5, p_rd=0, N=1):
	res=[]

	original_df = pd.DataFrame({x: dataset_dict[x] for x in dataset_dict})
	res.append(original_df)
	#print(original_df.context[42][:100])
	for _ in range(N):
		dataset_dict_copy = copy.deepcopy(dataset_dict)
		df = pd.DataFrame({x: dataset_dict_copy[x] for x in dataset_dict_copy})
		#d=dataset_dict.copy()
		# df['context'] = df.context.str.strip().replace(changes, regex=True)
		df['start_char'] = df.answer.apply(lambda x: x['answer_start'][0])
		df['end_char'] = df['start_char'] + df.answer.apply(lambda x: len(x['text'][0]))
		#df['final_answer0'] = [A[B:C] for A, B, C in zip(df.context, df['start_char'], df['end_char'])]

		df['context_before'] = [A[:C] for A, C in zip(df.context, df['start_char'])]
		df['context_after'] = [A[C:] for A, C in zip(df.context, df['end_char'])]
		df['context_answer'] = [A[B:C] for A, B, C in zip(df.context, df['start_char'], df['end_char'])]

		df['context_before'] = df['context_before'].apply(lambda t: transform_context(t, p_sr=p_sr, p_rd=p_rd))
		df['context_after'] = df['context_after'].apply(lambda t: transform_context(t, p_sr=p_sr, p_rd=p_rd))

		df['new_context'] = df['context_before'] + df['context_answer'] + " " +df['context_after']
		df['new_start_char'] = df['context_before'].str.len()

		a = df.answer.apply(pd.Series)
		a['n'] = df['new_start_char']
		a['answer_start'] = [replace(A, B) for A, B in zip(a.answer_start, a.n)]
		new_answer = a[['answer_start', 'text']].to_dict('records')
		df['new_answer']=new_answer

		df=df[['new_context', 'question', 'new_answer', 'label', 'id']].rename(columns={'new_context':'context','new_answer':'answer'})
		res.append(df[[i for i in dataset_dict.keys()]])
		#print(df.context[42][:100])
	#print(original_df.context[32][original_df.answer[32]['answer_start'][0]:])
	#print(df.context[32][df.answer[32]['answer_start'][0]:])

	#print(original_df.context[42][:100])
	#print(df.context[42][:100])

	new_dataset_dict = pd.concat(res)

	assert(len(original_df)*(N+1)==len(new_dataset_dict))
	return new_dataset_dict.to_dict(orient='list')


def replace(l, el):
    l[0]=el
    return l
# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import random
from random import shuffle
#random.seed(1)

PONCT='!"#$%&\')*+,-./:;<=>?@[\\]^_`{|}~'
#stop words list
stop_words = ['i','I', 'me', 'my', 'myself', 'we', 'our',
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
			'very', 's', 't', 'can', 'will', 'just', 'don',
			'should', 'now','may','us', '']

#cleaning up text
import re
def get_only_chars(line):

	clean_line = ""

	line = line.replace("’", "")
	line = line.replace("'", "")
	line = line.replace("-", " ") #replace hyphens with spaces
	line = line.replace("\t", " ")
	line = line.replace("\n", " ")
	line = line.lower()

	for char in line:
		if char in 'qwertyuiopasdfghjklzxcvbnm ':
			clean_line += char
		else:
			clean_line += ' '

	clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
	if clean_line[0] == ' ':
		clean_line = clean_line[1:]
	return clean_line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

#for the first time you use wordnet
#import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet




def synonym_replacement(words, n):
	new_words = words.copy()
	random_word_list = list(set([word for word in words if (word not in stop_words) and (word[0].islower()) ]))
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

	#this is stupid but we need it, trust me
	sentence = ' '.join(new_words)
	#new_words = sentence.split(' ')

	return sentence

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

def augment_context(context,start_char, len_answer,  p=0.5, p_del=0, N=1):
	end_char = start_char+ len_answer
	context_before = context[:start_char]
	the_answer = context[start_char:end_char]
	context_after = context[end_char:]
	context_before_augment = transform_context(context_before,p=p, p_del=p_del)
	context_after_augment = transform_context(context_after, p=p, p_del=p_del)

	new_start_char = len(context_before_augment)
	new_context = context_before_augment+the_answer+context_after_augment
	return new_context, new_start_char


def transform_context(text, p_sr=0.5,p_rd=0.2):

	# Load a text file if required
	output = ""
	counter = 0
	# Load the pretrained neural net
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	# Tokenize the text
	#tokenized = tokenizer.tokenize(text)

	# Get the list of words from the entire text
	words = word_tokenize(text)

	# Identify the parts of speech
	tagged = nltk.pos_tag(words)

	for i in range(0, len(words)):
		if random.random() < p_rd:
			continue

		#if words[i]  in stop_words:
		#	output = output + " " + words[i]
			#continue
		replacements = []

		# Only replace nouns with nouns, vowels with vowels etc.
		for syn in wordnet.synsets(words[i]):

			# Do not attempt to replace proper nouns or determiners
			if tagged[i][1] == 'NNP' or tagged[i][1] == 'DT':
				break
			# The tokenizer returns strings like NNP, VBP etc
			# but the wordnet synonyms has tags like .n.
			# So we extract the first character from NNP ie n
			# then we check if the dictionary word has a .n. or not
			word_type = tagged[i][1][0].lower()
			if syn.name().find("." + word_type + "."):
				# extract the word only

				r = syn.name()[0:syn.name().find(".")]
				if (r != words[i]) & ("_" not in r):
					replacements.append(r)

		if (len(replacements) > 0) and (random.random()<p_sr) and (words[i] not in stop_words):
			# Choose a random replacement
			replacement = replacements[randint(0, len(replacements) - 1)]
			#print(words[i] + " replaced by " + replacement)
			#counter += len(replacement) - len(words[i])
			output = output + " " + replacement

		else:
			# If no replacement could be found, then just use the
			# original word
			if (words[i] in PONCT) or (len(output)==0):
				output = output + words[i]
			else:
				output = output + " " + words[i]
	return	output


## TODO  Separer context avant et apres rep

if __name__ == '__main__':
	#word = ['interesting', 'boring']
	#print(synonym_replacement(word, 10))
	from debug_german import get_dataset2

	#print(transform_context("back turn return home fast lazy", p=0.5, p_del=0.1))
	dataset_dict = get_dataset2(datasets='duorc,race', data_dir='datasets/oodomain_train', split_name="train", debug=-1)
	augment_dataset_dict(dataset_dict, p_sr=0.5, p_rd=0, N=2)
