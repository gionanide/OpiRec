#!usr/bin/python
from nltk import ngrams
from collections import Counter


'''

Calculate the n-gram overlap in a list of sentences

'''
def count_grams_duplicate(sentences, n):

        duplicate_score_n = [] #keep the score of the n-grams overlap

        for sentence in sentences:# iterate all sentences
        
                n_grams = ngrams(sentence.split(), n)# split the sentence in n-grams(n-words)

                splitted_grams = []
                for grams in n_grams:
                        splitted_grams.append(grams)#put all the n-grams in a list
                        #print(grams)
                        
                count_dict = Counter(splitted_grams)# count the occurence of each gram in the list
                
                if (len(splitted_grams)>0):
                        duplicate_percentage = (1-len(count_dict)/len(splitted_grams))# otherwise is the unique samples devided by the duplicates
                #else:
                #        duplicate_percentage = 0.0
                        duplicate_score_n.append(duplicate_percentage)
                        
                else:
                        duplicate_score_n.append(0.0)#if there are no such grams the outcome is zero
                        
        return (sum(duplicate_score_n)/len(duplicate_score_n))
