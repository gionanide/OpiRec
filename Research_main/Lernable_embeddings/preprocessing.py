#!usr/bin/python
import os
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from classes import lang
import numpy as np
import pickle
import torch
import tensorflow as tf
import string
import minibatch_object

#This function removes all stopwords (words like the, a, for, in, etc.)
def del_stopwords (a):
    #remove the general stopwords
	stop_words = set(stopwords.words('english'))
	filtered = [w for w in a if not w in stop_words]		
	return filtered
	
def del_stopwords_from_file(a):
    #open the file with the stopwords and read every line of the file in a list, after this clean the review
	stopwords_file = open('./data/stopwords.txt','r')	
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
	
#flatten a list of lists into one list
def flatten(input_list):
    flat_list = []
    for sublist in input_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list
    
#open a file and read it as a dictionary
def open_ids(input_file):
    dictionary = []
    for line in input_file.readlines():
        dictionary.append((line.split(" ----- ")[0],line.split(" ----- ")[1]))
    return dictionary
	
#iterate all the words of the review and process them
def clean_review(reviews,language,lang_mode):
    reviews_update = []
    for review in reviews:
        review_update = []
        for words in review.split(" "):
            words = word_tokenize(words) #tokenize
            words = del_stopwords(words) # delete stopwords from nltk
            words = del_punct(words) # delete punctuation
            words = lemmatize(words) # lemmatize words
            words = toLowercase(words) # convert them to lowercase
            words = del_stopwords_from_file(words) # delete our own defined stopwrods
            if not( len(words)==0 or (words[0]=="sos" or words[0]=="eos")): # if we are processing a gap just continue
                    language.add_word(words[0])
                    review_update.append(words[0])
        if(lang_mode=="input"):
            review_update.insert(0,"sos")
            review_update.append("eos")
        reviews_update.append(review_update) # a list with all users reviews
    return reviews_update
    
#iterate all the words of the review and process them
def convert_to_index(reviews,language):
    reviews_update = []
    for review in reviews:
        review_update = []
        for words in review.split(" "):
            words = word_tokenize(words) #tokenize
            words = del_stopwords(words) # delete stopwords from nltk
            words = del_punct(words) # delete punctuation
            words = lemmatize(words) # lemmatize words
            words = toLowercase(words) # convert them to lowercase
            words = del_stopwords_from_file(words) # delete our own defined stopwrods
            if not(len(words)==0): # if we are processing a gap just continue
                    review_update.append(language.word2index[words[0]])
        reviews_update.append(torch.LongTensor(review_update)) # a list with all users reviews
    return reviews_update
    
# delete unused files
def clean_dataset():
    path = "./REAL_dataset/"
    input_files = os.listdir(path) # read all the files from this directoyry
    #load training and testing ids
    train_dict = open_ids(open("./data/train_ids.txt","r"))
    test_dict = open_ids(open("./data/test_ids.txt","r")) 
    for index, input_file in enumerate(input_files):
        input_ids = (input_file.split("--------")[0],input_file.split("--------")[1][:-4])#split the file's name
        if ( input_ids in train_dict or input_ids in test_dict):
            print("Inside") # if it is inside skip it
            continue
        else:
            os.remove(input_file)# else delete it
            print("Remove ---> ",input_file)
        
#gather all the ratings for a business 
def take_business_ratings(current_user_id,current_bussines_id,business_rating_dict,length):
    product_ratings = []
    #---------------------------------------------------------------------------------------> Keeping the rating for every poduct review                
    #iterate all the reviews for this specific business
    if (current_bussines_id in business_rating_dict):
        for rating_review_in_dict in business_rating_dict[current_bussines_id]:
            #assign the fields
            userId = rating_review_in_dict[0]
            rating = rating_review_in_dict[1]
            #if the current review is the review I want to predict just continue the loop skpping this one
            if (userId == current_user_id):
                #this is the rating we want to predict
                continue
            else:
                #else append the rating
                product_ratings.append(rating)
                if (len(product_ratings)==length):
                    break
    
    return product_ratings


#this functio takes all the input or target files, read the text and make a tokenizer based on this text
def make_vocabularies(mode):

    path = "./REAL_dataset/"
    input_files = os.listdir(path) # read all the files from this directoyry
    
    input_dict = open_ids(open("./data/"+mode+"_ids.txt","r"))
    
    #initialize class Language to keep the dictionary with the words and their indices
    language_input = lang.Lang()
    language_output = lang.Lang()
    
    for index, input_file in enumerate(input_dict):
        sample_file = open(path+input_file[0]+"--------"+input_file[1][:-1]+".txt").read() #open the file
        
        #XXX try to read the dictionary and handle some exceptions
        try:
            sample_file_dict = eval(sample_file)
        except NameError:
            print("Nan value",input_file)
            sys.exit(0)
        except SyntaxError:
            print("unexpected EOF while parsing",input_file)
            sys.exit(0)
            
        print("READ ----> ",input_file,index)
        
        '''                
        Dictionary's fields:
        -----------------------> userId
        -----------------------> businessId
        -----------------------> rating_review
        -----------------------> text_review
        -----------------------> rating_estimation
        -----------------------> role
        -----------------------> user_text_reviews
        -----------------------> business_text_reviews
        -----------------------> neighbourhood_text_reviews
        '''
        
        #load the texts
        user_reviews = sample_file_dict['user_text_reviews']
        product_reviews = sample_file_dict['business_text_reviews']
        neighbourhood_reviews = sample_file_dict['neighbourhood_text_reviews']
        text_review = sample_file_dict['text_review']
        
        #clean them
        user_reviews_update = clean_review(user_reviews,language_input,lang_mode="input")
        product_reviews_update = clean_review(product_reviews,language_input,lang_mode="input")
        neighbourhood_reviews_update = clean_review(neighbourhood_reviews,language_input,lang_mode="input")
        text_review_update = clean_review([text_review],language_output,lang_mode="output")
            
    #the text review is ready for the tokenizer procedure
    language_input_file = open("language_input_"+mode+".pkl","wb")
    pickle.dump(language_input, language_input_file)
    language_input_file.close()
    
    language_output_file = open("language_output_"+mode+".pkl","wb")
    pickle.dump(language_output, language_output_file)
    language_output_file.close()
	    
	    
#this function runs again all the files and encode the text sequences based on the tokenizer previously generated
def encode_sequences(mode):

    path = "./REAL_dataset/"
    input_files = os.listdir(path) # read all the files from this directoyry
    
    #load training and testing ids
    input_dict = open_ids(open("./data/"+mode+"_ids.txt","r")) 
    
    #load tokenizers
    language_input = pickle.load(open("language_input_"+mode+".pkl",'rb'))
    language_output = pickle.load(open("language_output_"+mode+".pkl",'rb'))
    
    print("Input vocabulary: ",len(language_input.word2index))
    print("Ouput vocabulary: ",len(language_output.word2index))
    
    
    try: #read the dictionary which contains every business Id and it's ratings
        business_rating_file = open('./data/businesses_ratings.txt','r').read()
        business_rating_dict = eval(business_rating_file)       
    except:
        print('Error reading the dictionary')
        
    user_reviews_last = []
    product_reviews_last = []
    neighbourhood_reviews_last = []
    product_rating_last = []
    text_reviews_target_last = []
    rating_target_last = []
    for index, input_file in enumerate(input_dict):
        sample_file = open(path+input_file[0]+"--------"+input_file[1][:-1]+".txt").read() #open the file
        
        #XXX try to read the dictionary and handle some exceptions
        try:
            sample_file_dict = eval(sample_file)
        except NameError:
            print("Nan value",input_file)
            sys.exit(0)
        except SyntaxError:
            print("unexpected EOF while parsing",input_file)
            sys.exit(0)
            
        print("READ ----> ",input_file,index)
        
        '''                
        Dictionary's fields:
        -----------------------> userId
        -----------------------> businessId
        -----------------------> rating_review
        -----------------------> text_review
        -----------------------> rating_estimation
        -----------------------> role
        -----------------------> user_text_reviews
        -----------------------> business_text_reviews
        -----------------------> neighbourhood_text_reviews
        '''
        
        #load the texts
        user_reviews = sample_file_dict['user_text_reviews']
        product_reviews = sample_file_dict['business_text_reviews']
        neighbourhood_reviews = sample_file_dict['neighbourhood_text_reviews']
        text_review = sample_file_dict['text_review']
        rating_review = sample_file_dict['rating_review']
        product_ratings = take_business_ratings(sample_file_dict["userId"],sample_file_dict["businessId"],business_rating_dict,len(product_reviews))
        
        
        #clean them
        user_reviews_update = convert_to_index(user_reviews,language_input)
        product_reviews_update = convert_to_index(product_reviews,language_input)
        neighbourhood_reviews_update = convert_to_index(neighbourhood_reviews,language_input)
        text_review_update = convert_to_index([text_review],language_output)[0]
        
        #for user_review_update in user_reviews_update:
        #    print(user_review_update.shape)
        user_reviews_update = torch.nn.utils.rnn.pad_sequence(user_reviews_update, batch_first=False, padding_value=0)
        print(user_reviews_update.shape)
        product_reviews_update = torch.nn.utils.rnn.pad_sequence(product_reviews_update, batch_first=False, padding_value=0)
        print(product_reviews_update.shape)
        neighbourhood_reviews_update = torch.nn.utils.rnn.pad_sequence(neighbourhood_reviews_update, batch_first=False, padding_value=0)
        print(neighbourhood_reviews_update.shape)
        print(text_review_update.shape)
        rating_review = torch.LongTensor([rating_review])
        print(rating_review.shape)
        product_ratings = torch.LongTensor(product_ratings)
        print(product_ratings.shape)
        
        user_reviews_last.append(user_reviews_update)
        product_reviews_last.append(product_reviews_update)
        neighbourhood_reviews_last.append(neighbourhood_reviews_update)
        product_rating_last.append(product_ratings)
        text_reviews_target_last.append(text_review_update)
        rating_target_last.append(rating_review)
        
        '''
        input_sequences = torch.nn.utils.rnn.pad_sequence([neighbourhood_reviews_update,product_reviews_update], batch_first=True, padding_value=0)
        print(input_sequences.shape)
        embedding_layer = torch.nn.Embedding(num_embeddings=len(language_input.word2index),embedding_dim=128,padding_idx=0)
        print(embedding_layer(input_sequences).shape)
        sentence_sum = torch.mean(embedding_layer(input_sequences), axis=1, keepdim=True).squeeze(1) # take the mean axis = the dimension where you want to calcualte the mean
        print(sentence_sum.shape)
        
        fsdfds
        '''
        
    minibatch_development = minibatch_object.Minibatch(user_reviews_last, product_reviews_last, neighbourhood_reviews_last, product_rating_last, text_reviews_target_last, rating_target_last)        
    minibatch_development_file = open("raw_"+mode+"_set.pkl","wb")
    pickle.dump(minibatch_development, minibatch_development_file)
    minibatch_development_file.close()
        

if __name__=="__main__":    
    mode = ["train"]
    for x in mode:
        #make_vocabularies(mode = x)
        encode_sequences(mode = x)
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
	    
        
        
