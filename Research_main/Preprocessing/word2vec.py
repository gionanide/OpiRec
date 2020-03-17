from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from Preprocessing import preprocessing
import sys

model = KeyedVectors.load_word2vec_format('/media/data/gionanide/OpinionRecommendation/GoogleNews-vectors-negative300.bin', binary=True)

#This function removes all stopwords (words like the, a, for, in, etc.)
def del_stopwords (a):
        #remove the general stopwords
	stop_words = set(stopwords.words('english'))
	filtered = [w for w in a if not w in stop_words]	
	return filtered

def del_stopwords_from_file(a):
        #open the file with the stopwords and read every line of the file in a list, after this clean the review
	stopwords_file = open('/media/data/gionanide/OpinionRecommendation/stopwords.txt','r')	
	stopwords_list = stopwords_file.read().splitlines()
	filtered = [w for w in a if not w in stopwords_list]
	return filtered

#This function removes words that are in fact punctuations like commas or question marks.
def del_punct (a):
	return [i for i in a if not all(j.isdigit() or j in string.punctuation for j in i)]

#lemmatize the words (rocks->rock)
def lemmatize (a):
	lemmatizer = WordNetLemmatizer()
	return [lemmatizer.lemmatize(i) for i in a]
	
#All words to lowercase
def toLowercase (a):
	return [i.lower() for i in a]
	
def words2vec(a):
	vectors = []
	review = np.zeros(300)              
	for word in range(len(a)):
		if a[word] in model:
			vectors.append(model[a[word]]) #model[word] turns the word to a 300d vector 
		else:
		        out_of_dict = '{0} is an out of dictionary word'.format(a[word])
	for word in range(len(vectors)):
		for i in range(300):
			review[i] += vectors[word][i]
	for i in range(300):
                if (len(vectors)==0):
	        
                        continue        
                else:
                        review[i] = review[i]/(1.0*len(vectors))		
	return review
	
#erase all the words that are not in the model
def clean_text_review_natural_text(text_review):
        #just to keep some properties untouched
        temp_text_review = text_review.copy()
        for word in temp_text_review:                
                #because I want to keep the starting and the ending symbol
                if  ( not(word in model) ):
                        #we want to erase all the appearences of this word in the list
                        while (word in text_review):                        
                                #remove the word if it is not belong to the model. Exception if the word does not belong to the model but it is EOS/SOS keep
                                text_review.remove(word)       
        return text_review
	
#erase all the words that are not in the model
def clean_text_review(text_review, tokenizer):
        #just to keep some properties untouched
        temp_text_review = text_review[0]        
        #change the format
        text_review = text_review[0]
        for word in temp_text_review:
                word_id = word
                word = preprocessing.index_to_word_mapping(word, tokenizer)                
                if  not( word in model ):
                        text_review.remove(word_id)
        return [text_review]
		
def word2vec_cleaned(a, index):
        #checking first if the word has an embedding
        if a in model:
                return model[a]        
        else:
                #if it has not I have to found a word with embedding
                print('Word with no embedding:',a,'with index:',index)
                word = input("Enter word with embedding: ")
                while(not(word in model)):
                        word = input("Enter word with embedding: ")
                return model[word]

def word2vec (string, index):
	vec = word2vec_cleaned(string, index)
	return vec

def review2vec (string):
	words = word_tokenize(string)
	words = del_stopwords(words)
	words = del_punct(words)
	words = lemmatize(words)
	words = toLowercase(words)
	words = del_stopwords_from_file(words)
	vec = words2vec(words)
	return vec
