import pandas as pd
import numpy as np
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from random import randint
import nltk.data
import string
import random
CHANGES = dict(zip([" game ", " was "], [" lame ", " sos "]))


def augment_dataset_dict(dataset_dict, changes = None):
	changes=CHANGES
	original_df = pd.DataFrame({x: dataset_dict[x] for x in dataset_dict})
	df= original_df.copy()
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





# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import random
from random import shuffle
random.seed(1)

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

def transform_context(text, p=0.5):

	# Load a text file if required
	output = ""
	counter = 0

	# Load the pretrained neural net
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	# Tokenize the text
	tokenized = tokenizer.tokenize(text)

	# Get the list of words from the entire text
	words = word_tokenize(text)

	# Identify the parts of speech
	tagged = nltk.pos_tag(words)

	for i in range(0, len(words)):
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
				if r != words[i]:
					replacements.append(r)

		if (len(replacements) > 0) and (random.random()>p) and (words[i] not in stop_words):
			# Choose a random replacement
			replacement = replacements[randint(0, len(replacements) - 1)]
			#print(words[i] + " replaced by " + replacement)
			#counter += len(replacement) - len(words[i])
			output = output + " " + replacement

		else:
			# If no replacement could be found, then just use the
			# original word
			if (words[i] in string.punctuation) or (len(output)==0):
				output = output + words[i]
			else:
				output = output + " " + words[i]
	return	output


## TODO  Separer context avant et apres rep

if __name__ == '__main__':
	#word = ['interesting', 'boring']
	#print(synonym_replacement(word, 10))
	from debug_german import get_dataset2

	id_example=249
	dataset_dict = get_dataset2(datasets='duorc,race', data_dir='datasets/oodomain_train', split_name="train", debug=-1)
	original_df = pd.DataFrame({x: dataset_dict[x] for x in dataset_dict})
	df = original_df.copy()
	# df['context'] = df.context.str.strip().replace(changes, regex=True)

	df['start_char'] = df.answer.apply(lambda x: x['answer_start'][0])
	df['end_char'] = df['start_char'] + df.answer.apply(lambda x: len(x['text'][0]))
	df['final_answer'] = [A[B:C] for A, B, C in zip(df.context, df['start_char'], df['end_char'])]

	question = df.loc[id_example, 'question']
	answer = df.loc[id_example, 'answer']
	start_char = df.loc[id_example, 'start_char']
	end_char = df.loc[id_example, 'end_char']

	#answer_word = answer["text"][0]
	#start_char = answer["text"][0]
	context = df.loc[id_example, 'context']

	context_before = context[:start_char]
	the_answer = context[start_char:end_char]
	context_after = context[end_char:]
	context_before_augment = transform_context(context_before)
	print(context_before[:250])
	print(context_before_augment[:250])
	#print(question)
	#print(the_answer)
