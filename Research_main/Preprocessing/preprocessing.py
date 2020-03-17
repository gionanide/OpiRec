#!usr/bin/python\c
import os
import re
import string
from unicodedata import normalize
from Preprocessing import word2vec as w2v
import numpy as np
import keras
import tensorflow as tf
from matplotlib import pyplot as plt
import sys


#take as an input the directory of the sample files and initialize the procedure
def handle_input(path):
        samples = os.listdir(path)
        #initialize the size of the directory
        total_size = 0
        #calculat the size of the directory
        for path, dirs, files in os.walk(path):
                #iterate all the files
                for f in files:
                        #we have to walk the whole directory tree and add each file's size
                        fp = os.path.join(path,f)
                        #add size in bytes
                        total_size += os.path.getsize(fp)          
        print('\n\n')
        print('--------------------------- Initialization properties ---------------------------')
        print('Directory',path)                
        print(len(samples),'files in the directory, directory size',(total_size/1000000),'MB')
        print('\n\n')

        return samples
        
#make this function in order to assign the reviews from the dictionary to a numpy array with shape: (embedding_size(300) x number_of_reviews )        
def make_reviews_array(input_list, normalize_reviews):
        #initalize the array, recall that the embedding size is 300
        output_array = np.empty([300,int(normalize_reviews)])
        #iterate the enumerate list, so the index in the column to assign the review
        for index,review in enumerate(input_list):
                #because we want an upper bound, and to make all the samples to have the same shape
                if (index==int(normalize_reviews)):
                
                        break
                #convert the review to embedding
                review_vec = w2v.review2vec(review)
                #add then append it as a row
                output_array[:,index] = review_vec
        
        return output_array
       
#because the neighbourhood model consist of a list of lists of lists, we want to make it just a list of lists(reviews), so we have to flatten the list
def flatten(input_list):
        #iterate the general list
        for reviews_list in input_list:
                #iterate every list which is inside the list
                for review in reviews_list:
                        #return a generator for every element in all the lists, we use yield instead of initializing a new list to store the generated lists
                        yield review
                      
'''
We make this function in order to make tha appropiratte format for our target data, we want to clean it
from special characters to make it lowercase etc.
'''                        
def clean_review(text_review, re_print, table):
        #if you want to erase all the single letters
        #text_review = ' '.join( [w for w in text_review.split() if len(w)>1] )
        #print('before preprocessing: ------> ',text_review,'\n')
        #normalize unicode characters
        text_review = normalize('NFD',text_review).encode('ascii','ignore')
        text_review = text_review.decode('UTF-8')
        text_review = text_review.split()
        #convert to lowercase
        text_review = [word.lower() for word in text_review]        
        #remove puncuations from each token
        text_review = [word.translate(table) for word in text_review]        
        #remove non printable chars from each token
        text_review = [re_print.sub('',w) for w in text_review]        
        #remove tokens with numbers
        text_review = [word for word in text_review if word.isalpha()]
        #here we erese all the words that are not in our vocabulary to avoid predictions out of the vocabulary
        text_review = w2v.clean_text_review_natural_text(text_review)
        #add the begin and stop symbols in the begging and in the end of each sentence
        text_review.insert(0,'SOS')
        text_review.append('EOS')
        length = len(text_review)
        text_review = ' '.join(text_review)        
        return text_review, length
      
'''
Using this function we are checking if a tensor contains nan values
'''        
def check_nan_tensor(tensor):
        #returns 0 if everything is ok, else if there is a Nan values it returns 1
        assessment = (tensor != tensor).any()
        return assessment

'''
delete all the words that their count is lower that a threshold
'''
def erase_rare_words(tokenizer, count_thres):
        low_count_words = []
        #find all the low count words
        for word, count in tokenizer.word_counts.items():
                #find all these words
                if (count < count_thres):
                        #and gather them in a list
                        low_count_words.append(word)
        for w in low_count_words:
                #erase all these words from the tokenizer
                del tokenizer.word_index[w]
                del tokenizer.word_docs[w]
                del tokenizer.word_counts[w]
        print('Erased -bad- words:',len(low_count_words),'keep only:',len(tokenizer.word_index),'words \n')        
        return tokenizer
       
'''
use this function in order to erase every word that still appears and it is not in the Tokenizer because of the previous procedure that we use to erase some 'bad' words
'''
def overall_cleaning(cleaned_sentences, tokenizer):
        cleaned_sentences_new = []
        #iterate every sentence in the archive
        for sentence in cleaned_sentences:
                sentence = sentence.split(' ')
                #make a copy of the list to keep all the words
                temp_tentence = sentence.copy()
                #iterate all the words of the sentences
                for word in temp_tentence:
                        #check if the word is in the tokenizer
                        if ( (not(word in tokenizer.word_index)) and ( (not(word=='SOS')) and (not(word=='EOS')) ) ):
                                #we want to erase all the appearences of this word in the list
                                while (word in sentence):
                                        #remove the word if it is not belong to the model. 
					#Exception if the word does not belong to the model but it is EOS/SOS keep
                                        sentence.remove(word) 
                #append the new sentence
                cleaned_sentences_new.append(sentence)     
        return cleaned_sentences_new
   
'''
delete all the words that their count is higher that a threshold
'''
def erase_most_frequent_words(tokenizer, max_id_to_keep):
        high_count_words = []
        if (max_id_to_keep >= len(tokenizer.word_index)):
                print('You choice is: ',max_id_to_keep,'and the length of the tokenizer is: ',len(tokenizer.word_index))
                max_id_to_keep = int(input('please adjust your choice: '))
        for word_id in range(1,max_id_to_keep+1):
                word_found = index_to_word_mapping(word_id, tokenizer)
                if ( word_found=='eos' or word_found=='sos'):
                        continue
                high_count_words.append(word_found)  
        for index, word_found in enumerate(high_count_words):
                #erase all these words from the tokenizer
                del tokenizer.word_index[word_found]
                del tokenizer.word_docs[word_found]
                del tokenizer.word_counts[word_found]        
        print('Erased -very good- words:',max_id_to_keep)        
        return tokenizer
              
'''
Update tokenizer with the new vocabulary
'''
def update_tokenizer(tokenizer, cleaned_sentences):
        #clean the sentences based on the new tokenizer
        cleaned_sentences = overall_cleaning(cleaned_sentences, tokenizer)
        #initalize an instance of the Tokenizer class
        tokenizer_new = tf.keras.preprocessing.text.Tokenizer()       
        #fit tokenizer again because not we erase some words
        tokenizer_new.fit_on_texts(cleaned_sentences)
        return tokenizer_new, cleaned_sentences
                     
'''
We want to make a dictionary which gives unique
id to every work (Tokenizer object) because we want to keep track of it in the 'classification' procedure.
In the end we want to conclude in a array which contains all the words formated as one hot vectors.
'''                
def format_target(path,samples, padding, cut_bad_words, erase_most_frequent_word, max_id_to_keep, count_thres):
        #use the escape to backslash all the non alphanumeric characters
        re_print = re.compile('[^%s]' % re.escape(string.printable))
        #remove pncuations using mapping
        table = str.maketrans('', '', string.punctuation)
        #make a list with the cleaned sentences
        cleaned_sentences = []
        #keep track of the empty text reviews
        empty_text_reviews = []
        #keep record of the length of every target review, dictionary {index: index_length}
        target_reviews_length_train = {}
        target_reviews_length_test = {}
        #initalize an instance of the Tokenizer class
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        #indeces only for the dictionaries
        count_train=0
        count_test=0
        #iterate all the samples in the specific directory to make the sets
        for index, sample in enumerate(samples):
                #open the file of every sample, and read all the content of the file as string
                sample_file = open(path+'/'+sample,'r').read()
                #because sometimes we handle empty strings
                try:
                        #evaluate the command which is in str format to initialize the dictionary
                        sample_file_dictionary = eval(sample_file)
                except SyntaxError:                        
                        empty_text_reviews.append(index)
                        continue        
                except NameError:
                        empty_text_reviews.append(index)
                        continue
                #we have to change the format of the text review into one hot encoding
                text_review = sample_file_dictionary['text_review']
                text_review, length = clean_review(text_review, re_print, table)
                cleaned_sentences.append(text_review)
                #assign the sample to the corresponding set as a tuple
                if (sample_file_dictionary['role'] == 'training'):
                        #keep track only for the training samples
                        target_reviews_length_train[str(count_train)] = length
                        count_train+=1   
                elif (sample_file_dictionary['role'] == 'testing'):
                        #keep track only for the testing samples
                        target_reviews_length_test[str(count_test)] = length
                        count_test+=1
        #the text review is ready for the tokenizer procedure
        tokenizer.fit_on_texts(cleaned_sentences)
        if (cut_bad_words):
                #erase all the 'bad' words
                tokenizer = erase_rare_words(tokenizer, count_thres)       
        if (erase_most_frequent_word):
                #erase the most frequent words
                tokenizer = erase_most_frequent_words(tokenizer, max_id_to_keep)       
        #keep track of the max_sentence length because we want to use it as padding
        max_review_length = 0      
        #update again, in general update every time you change the tokenizer
        tokenizer, cleaned_sentences = update_tokenizer(tokenizer, cleaned_sentences)
        #new vocabulary output
        output_vocabulary_size = len(tokenizer.word_index) + 1       
        #convert every sentence instead of sequence of words a sequence of integers, each one represent 
        #an id of a word
        sequences = tokenizer.texts_to_sequences(cleaned_sentences)
        if(padding):
                #check the max length review for the padding 
                for sentence in sequences:
                        if (len(sentence)>max_review_length):
                                max_review_length = len(sentence)
                #now we have to apply padding to our sequences, to reach the max review
                sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_review_length, padding='post')
                print(sequences)        
        else:
                max_review_length='No padding'
        eos = tokenizer.word_index['eos']
        sos = tokenizer.word_index['sos']
        print('--------------------------- Preprocessing properties ---------------------------')
        #print('target text reviews:',sequences.shape)
        print('Output vocabulary size:',output_vocabulary_size)
        print('max review length:',max_review_length)
        print('EOS tokenizer',eos)
        print('SOS tokenizer',sos)
        print('\n\n')

        return sequences, output_vocabulary_size, max_review_length, empty_text_reviews, eos, sos, tokenizer, target_reviews_length_train, target_reviews_length_test
   
#make the reverse mapping to go from indexing to the real word 
def index_to_word_mapping(requested_index, tokenizer):
        #iterate all the elements in the ['word'] = index, mapping
        for word, index in tokenizer.word_index.items():
                #if the requested index is in the dictionary, return the corresponding word
                if (requested_index == index):
                        return word               
        #if the requested index is not in the dictionary, return None
        return None
     
'''    
prepare training and testing packages as follows
Training:
        input: tuple( --- all the reviews of the user --- all the reviews of the product --- all the reviews of the neighbourhood)
        output: tuple( --- text review --- rating review)
'''
def make_training_testing(path, samples, target_reviews, empty_text_reviews, normalize_user_reviews, normalize_product_reviews, normalize_neighbourhood_reviews, output_vocabulary_size, one_hot, empty_flag):

        #initialize training and testing lists
        user_training_samples_review = []
        user_testing_samples_review = []
        #initialize training and testing lists
        product_training_samples_review = []
        product_testing_samples_review = []
        product_training_samples_rating = []
        product_testing_samples_rating = []
        #initialize training and testing lists
        neighbourhood_training_samples_review = []
        neighbourhood_testing_samples_review = []
        #initialize the ground truths
        training_ground_truth_review = []
        testing_ground_truth_review = []
        #initialize the ground truths
        training_ground_truth_rating = []
        testing_ground_truth_rating = []
        try:
                #read the dictionary which contains every business Id and it's ratings
                business_rating_file = open('/media/data/gionanide/OpinionRecommendation/Proceedings/businesses_ratings.txt','r').read()
                business_rating_dict = eval(business_rating_file)        
        except:
                print('Error reading the dictionary')
        count=0
        count_to_stop=0
        review_index=-1
        empty_flag = False
        #iterate all the samples in the specific directory to make the sets
        for index, sample in enumerate(samples):                
                #open the file of every sample, and read all the content of the file as string
                sample_file = open(path+'/'+sample,'r').read()
                try:
                        #evaluate the command which is in str format to initialize the dictionary
                        sample_file_dictionary = eval(sample_file)
                except NameError:                        
                        empty_flag = True
                        #do not update anything and just continue
                        continue       
                except SyntaxError:                        
                        empty_flag = True
                        #do not update anything and just continue
                        continue        
                #if the index is in the list with the reviews that do not contain a test review, skip it
                if (index in empty_text_reviews):                        
                        empty_flag = True
                        #do not update anything and just continue
                        continue  
                else:
                        #else if the sample is ok update propersly
                        review_index+=1
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
                #------> assign the lists of the reviews of each dictionary
                #print(sample_file_dictionary)
                user_model = sample_file_dictionary['user_text_reviews']
                #print('User reviews:',len(user_model))
                product_model = sample_file_dictionary['business_text_reviews']
                #print('Product reviews:',len(product_model))
                neighbourhood_model = sample_file_dictionary['neighbourhood_text_reviews']
                #print('Neighbourhood reviews:',len(neighbourhood_model))
                rating_review = sample_file_dictionary['rating_review']
                #--> initialize the corresponding matrices, review embedding is being represented with a 300 vector  
                #if we gave as a normalization term None value it means that we are taking all the users and product reviews
                if (normalize_user_reviews==None):
                        normalize_user_reviews = len(user_model)
                elif(normalize_user_reviews>len(user_model)):
                        normalize_user_reviews = len(user_model)    
                user_model_array = make_reviews_array(user_model, normalize_user_reviews)
                if (normalize_product_reviews==None):
                        normalize_product_reviews = len(product_model)
                elif(normalize_product_reviews>len(product_model)):
                        normalize_product_reviews = len(product_model)
                product_ratings = []
                #-----------> Keeping the rating for every poduct review                
                #iterate all the reviews for this specific business
                if ( sample_file_dictionary['businessId'] in business_rating_dict ):
                        for rating_review_in_dict in business_rating_dict[sample_file_dictionary['businessId']]:
                                #assign the fields
                                userId = rating_review_in_dict[0]
                                rating = rating_review_in_dict[1]
                                #if the current review is the review I want to predict just continue the loop skpping this one
                                if (userId == sample_file_dictionary['userId']):
                                        #this is the rating we want to predict
                                        continue
                                else:
                                        #else append the rating
                                        product_ratings.append(rating)                            
                else:
                        review_index-=1
                        #because I do not have the rating for this business
                        continue
                #take all the ratings from this business, keeping only in the range of normalize_product_reviews
                product_ratings_normalized = product_ratings[:normalize_product_reviews].copy()
                product_model_array = make_reviews_array(product_model, normalize_product_reviews)
                #first I have to flatten the list of lists of lists to list of lists
                neighbourhood_model = list(flatten(neighbourhood_model))
                neighbourhood_model_array = make_reviews_array(neighbourhood_model, normalize_neighbourhood_reviews)
                #split the training samples for every submodel -----------------------------------------> Training
                user_model_array = (user_model_array).transpose()
                product_model_array = (product_model_array).transpose()
                neighbourhood_model_array = (neighbourhood_model_array).transpose()
                #split testing samples for every submodel -------------------------------------------------------> Testing
                review = np.array(target_reviews[review_index])
                #assign the sample to the corresponding set as a tuple
                if (sample_file_dictionary['role'] == 'training'):
                        #feed the NN
                        user_training_samples_review.append(user_model_array)
                        product_training_samples_review.append(product_model_array)
                        product_training_samples_rating.append(product_ratings_normalized)
                        neighbourhood_training_samples_review.append(neighbourhood_model_array)
                        #validate the NN
                        training_ground_truth_review.append(review)
                        training_ground_truth_rating.append(rating_review)                        
                elif (sample_file_dictionary['role'] == 'testing'):
                        #feed the NN
                        user_testing_samples_review.append(user_model_array)
                        product_testing_samples_review.append(product_model_array)
                        product_testing_samples_rating.append(product_ratings_normalized)
                        neighbourhood_testing_samples_review.append(neighbourhood_model_array)
                        #NN willing result
                        testing_ground_truth_review.append(review)
                        testing_ground_truth_rating.append(rating_review)
                #just to keep track where we are
                count+=1
                #print('---------------------- Sample No. '+str(count)+' / '+str(len(samples))+' ----------------------', end='\r')
                #sys.stdout.flush()
                #print(str(count),end=)
                #print('Role:',sample_file_dictionary['role'])
                #print('User example:',user_model_array.shape)
                #print('Product example:',product_model_array.shape)
                #print('Neighbourhood example:',neighbourhood_model_array.shape)
                #print('Target review number:',review_index)
                #print('Target review example:',target_reviews[review_index].shape)
                #print()
                #print(training_samples,training_ground_truth_review)
                count_to_stop+=1
                #if(count_to_stop==2):
                #enable the break command only for DEBUGGING PROCEDURE
                #break
        #initialize training and testing lists
        user_training_samples_review = np.array(user_training_samples_review)
        user_testing_samples_review = np.array(user_testing_samples_review)
        #initialize training and testing lists
        product_training_samples_review = np.array(product_training_samples_review)
        product_testing_samples_review = np.array(product_testing_samples_review)
        #initialize training and testing lists
        neighbourhood_training_samples_review = np.array(neighbourhood_training_samples_review)
        neighbourhood_testing_samples_review = np.array(neighbourhood_testing_samples_review)
        training_ground_truth_review = np.array(training_ground_truth_review)
        #one hot encoding
        testing_ground_truth_review = np.array(testing_ground_truth_review)
        #one hot encoding
        if(one_hot):
                training_ground_truth_review = keras.utils.to_categorical(training_ground_truth_review, num_classes=output_vocabulary_size)        
                testing_ground_truth_review = keras.utils.to_categorical(testing_ground_truth_review, num_classes=output_vocabulary_size) 
        return user_training_samples_review,user_testing_samples_review,product_training_samples_review,product_testing_samples_review,neighbourhood_training_samples_review,neighbourhood_testing_samples_review,training_ground_truth_review,testing_ground_truth_review, empty_flag, training_ground_truth_rating, testing_ground_truth_rating, product_training_samples_rating, product_testing_samples_rating
               
        
'''
Now we are making different format: if I do not want the teacher foarcing procedure I do not want to use them, but in the teacher forcing procedure I have to format the deocoder's input as well. To do this I have to remove the EOS symbol from it's inputs and to remove SOS symbol from it's output.
index{sos:1}
index{eos:2}
------ input ------
- [1,6,8,17,2,0,0,0,0,0]
----- output ------
- [6,8,17,2,0,0,0,0,0,0]
'''
#remove Start Of Sequence symbol 
def removeSOS(array,set_size):
        #make an array of zeros same length as our input array
        zeros = np.zeros(set_size)
        #reshape it
        zeros = np.reshape(zeros, (zeros.shape[0], 1))
        #take all the elements except from the first one (Start Of Sequence)
        array = array[:,1:]
        #concat a column of zeros to the end the keep the same padding
        array = np.concatenate((array,zeros), axis=1).astype(int)
        #return array
        return array
        
'''     
----- input ------
- [1,6,8,17,2,0,0,0,0,0]
------ output ------
- [1,6,8,17,0,0,0,0,0,0]
'''
#remoce End Of Sequence sumbol       
def removeEOS(array):
        #where the array value is 2 replace is it with 0
        array[array==22] = 0
        return array
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
