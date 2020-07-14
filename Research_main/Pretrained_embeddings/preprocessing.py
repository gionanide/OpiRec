#!usr/bin/python
import numpy as np
import sys
from Preprocessing import preprocessing
import minibatch_object
import pickle
import itertools


#sort a list based on another list
def sort_based_on_list(base_list, list_to_sort):

        #sort all the lists based on the index of the target
        sorted_list = [list_to_sort_element for _,list_to_sort_element in sorted(zip(base_list,list_to_sort), key=lambda pair: pair[0])]
        
        return sorted_list
        
        
#given a list take only the order of indices
def make_order(input_list):
        
        keep_order = []
        for review in input_list:
        
                keep_order.append(review[1])
                
        return keep_order


#main function to keep the dataset sorted and save it as an object
def make_sorted_dataset():
       
        #file to read the sampes
        path = '/media/data/gionanide/OpinionRecommendation/Preprocessing/REAL_dataset' # 47036 samples
        #path = '/media/data/gionanide/OpinionRecommendation/Preprocessing/Feeding_samples_90-10/' # -----------------> 53565 samples
        #path = '/media/data/gionanide/OpinionRecommendation/data/Feeding_samples_90-10_medium/' # -------------> 2089 samples
        #path = '/media/data/gionanide/OpinionRecommendation/data/Feeding_smaples_90-10_small/' # ---------------> 512 samples
        #path = '/media/data/gionanide/OpinionRecommendation/data/Feeding_smaples_90-10_oneshot/' # ------------> 1 sample
        
        train_file = open("./train_ids.txt","r")
        test_file = open("./test_ids.txt","r")
        train_dict = []
        test_dict = []
        for line in train_file.readlines():
                train_dict.append((line.split(" ----- ")[0],line.split(" ----- ")[1][:-1]))
        for line in test_file.readlines():
                test_dict.append((line.split(" ----- ")[0],line.split(" ----- ")[1][:-1]))
        
        #return a list with all the samples, as paths
        samples = preprocessing.handle_input(path)
        print('samples',len(samples))
        
        #train_ids = open("train_ids.txt","w+")
        #test_ids = open("test_ids.txt","w+")
        dev_ids = open("dev_ids.txt","w+")
        
        #if we want padding
        padding=False
        
        #if we want to erase the less common words
        cut_bad_words=True
        
        #define if you want to crop your vocabulary
        cut_bad_words = True
        
        #define the threshlod 
        count_thres = 125
        
        #erase the most frequent words
        erase_most_frequent_word = True
        
        #define the threshlod 
        max_id_to_keep = 1
        
        #max length to keep
        max_review = 300
                
        #make the appropriatte format for the target
        targets, output_vocabulary_size, max_review_length, empty_text_reviews, eos, sos, tokenizer, target_reviews_length_train, target_reviews_length_test = preprocessing.format_target(path, samples, padding, cut_bad_words, erase_most_frequent_word, max_id_to_keep, count_thres)
        
        #print(targets)
        #print(len(targets[0]))
        print('targets',len(targets))
        
        
        #make the appropriate format to feed the neural network, training testing samples
        
        #make this as True if you want to encode your target to one-hot encoding istead of a sequence of words IDs
        one_hot=False
        
        #just for the initialization
        empty_flag=False
        
        #define the number of maximum reviews for every sample, if None we are taking all the existing reviews
        normalize_user_reviews = 40
        normalize_product_reviews = 100
        normalize_neighbourhood_reviews = 40
        
        
        targets_overall = []
        
        for index, target in enumerate(targets):
        
                targets_overall.append((target, index, len(target)))
        

        #initialize training and testing lists
        user_training_samples_review = []
        user_testing_samples_review = []
        user_development_samples_review = []
        
        #initialize training and testing lists
        product_training_samples_review = []
        product_testing_samples_review = []
        product_development_samples_review = []
        product_training_samples_rating = []
        product_testing_samples_rating = []
        product_development_samples_rating = []
        
        #initialize training and testing lists
        neighbourhood_training_samples_review = []
        neighbourhood_testing_samples_review = []
        neighbourhood_development_samples_review = []
        
        #initialize the ground truths
        training_ground_truth_review = []
        testing_ground_truth_review = []
        development_ground_truth_review = []
        
        #initialize the ground truths
        training_ground_truth_rating = []
        testing_ground_truth_rating = []
        development_ground_truth_rating = []
        
        
        try:
                
                #read the dictionary which contains every business Id and it's ratings
                business_rating_file = open('/media/data/gionanide/OpinionRecommendation/Proceedings/businesses_ratings.txt','r').read()
                business_rating_dict = eval(business_rating_file)
                        
        except:
        
                print('Error reading the dictionary')

        count=0
        
        count_to_stop=0
        
        empty_flag = False
        
        #print('review index',review_index)
        review_index = -1
        
        
        count_training = 5000
        count_testing = 3000
        count_development = 0
        

        #iterate all the samples in the specific directory to make the sets
        for index, sample in enumerate(samples):
        
                #print(sample)
                
                #open the file of every sample, and read all the content of the file as string
                sample_file = open(path+'/'+sample,'r').read()
                
                
                try:
                
                        #evaluate the command which is in str format to initialize the dictionary
                        sample_file_dictionary = eval(sample_file)
                        
                except NameError:
                
                        print('\n Nan value \n')
                        
                        empty_flag = True
                        
                        #do not update anything and just continue
                        continue
                        
                except SyntaxError:
                
                        print('\n unexpected EOF while parsing \n')
                        
                        empty_flag = True
                        
                        #do not update anything and just continue
                        continue
                        
                        
                        
                #if the index is in the list with the reviews that do not contain a test review, skip it
                if (index in empty_text_reviews):
                
                        print('\n Empty review \n')
                        
                        empty_flag = True

                        #do not update anything and just continue
                        continue
                        
                else:
                
                        #else if the sample is ok update properly
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
                
                #--------------------------------------------------------------------------------> assign the lists of the reviews of each dictionary
                #print(sample_file_dictionary)
                
                
                user_model = sample_file_dictionary['user_text_reviews']
                #print('User reviews:',len(user_model))
                
                product_model = sample_file_dictionary['business_text_reviews']
                #print('Product reviews:',len(product_model))
                
                neighbourhood_model = sample_file_dictionary['neighbourhood_text_reviews']
                #print('Neighbourhood reviews:',len(neighbourhood_model))
                
                
                rating_review = sample_file_dictionary['rating_review']

                
                
                #-------------------------------------------------------------------------------> initialize the corresponding matrices, review embedding is being represented with a 300 vector  
                
                #if we gave as a normalization term None value it means that we are taking all the users and product reviews
                if (normalize_user_reviews==None):
                        
                        normalize_user_reviews = len(user_model)
                        
                elif(normalize_user_reviews>len(user_model)):
                
                        #print(len(user_model))
                        #continue
                        normalize_user_reviews = len(user_model)
                
                              
                user_model_array = preprocessing.make_reviews_array(user_model, normalize_user_reviews)
                
                
                if (normalize_product_reviews==None):
                        
                        normalize_product_reviews = len(product_model)
                        
                elif(normalize_product_reviews>len(product_model)):
                
                        #print(len(product_model))
                        #continue
                        normalize_product_reviews = len(product_model)
                        
                if (normalize_neighbourhood_reviews==None):
                        
                        normalize_neighbourhood_reviews = len(neighbourhood_model)
                        
                elif(normalize_neighbourhood_reviews>len(neighbourhood_model)):
                
                        #print(len(product_model))
                        #continue
                        normalize_neighbourhood_reviews = len(neighbourhood_model)
                        
                        
                if ( (normalize_user_reviews<39) or (normalize_product_reviews<99) or (normalize_neighbourhood_reviews<39) ):
                        print(index,normalize_user_reviews,normalize_product_reviews,normalize_neighbourhood_reviews,'continue')
                        continue #skip this sample
                
                
                product_ratings = []
                #---------------------------------------------------------------------------------------> Keeping the rating for every poduct review                
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
                
                        #review_index-=1
                        
                        #because I do not have the rating for this business
                        continue

                
                #take all the ratings from this business, keeping only in the range of normalize_product_reviews
                product_ratings_normalized = product_ratings[:normalize_product_reviews].copy()
                
                
                
                product_model_array = preprocessing.make_reviews_array(product_model, normalize_product_reviews)
                
                
                #first I have to flatten the list of lists of lists to list of lists
                neighbourhood_model = list(preprocessing.flatten(neighbourhood_model))
              
                neighbourhood_model_array = preprocessing.make_reviews_array(neighbourhood_model, normalize_neighbourhood_reviews)
                
                
                
                #split the training samples for every submodel -----------------------------------------> Training
                user_model_array = (user_model_array).transpose()
                
                product_model_array = (product_model_array).transpose()
                
                neighbourhood_model_array = (neighbourhood_model_array).transpose()
                
                #print(train_user.shape)
                
                #we need to reshape our format as follows: || samples -- time steps -- features || as timesteps we consider the number of reviews
                #user_model_array = np.reshape(user_model_array,(1,user_model_array.shape[0],300))
                
                #product_model_array = np.reshape(product_model_array,(1,product_model_array.shape[0],300))
                
                #neighbourhood_model_array = np.reshape(neighbourhood_model_array,(1,neighbourhood_model_array.shape[0],300))
                
                
                #split testing samples for every submodel -------------------------------------------------------> Testing
                review = np.array(targets_overall[review_index][0])
                if (review.size>max_review):
                        continue
                
                #review = np.reshape(review, (review.shape[0], 1))            
                
                #assign the sample to the corresponding set as a tuple
                if (sample_file_dictionary['role'] == 'training'):
                
                        if (count_training>3000):
                                continue
                                
                        #feed the NN
                        user_training_samples_review.append(user_model_array)
                        
                        product_training_samples_review.append(product_model_array)
                        product_training_samples_rating.append(product_ratings_normalized)
                        
                        neighbourhood_training_samples_review.append(neighbourhood_model_array)
                        
                        #print(review)
                        #print('DEBUGGIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIING')
                        
                        #validate the NN
                        training_ground_truth_review.append((review,targets_overall[review_index][2]))
                        
                        training_ground_truth_rating.append(rating_review)
                        
                        #train_ids.write(str(sample_file_dictionary['userId'])+" ----- "+str(sample_file_dictionary['businessId']))
                        #train_ids.write("\n")
                        
                        count_training+=1
                        
                        
                elif (sample_file_dictionary['role'] == 'testing'):
                
                        if (count_testing<1000):
                
                                #feed the NN
                                user_testing_samples_review.append(user_model_array)
                                
                                product_testing_samples_review.append(product_model_array)
                                product_testing_samples_rating.append(product_ratings_normalized)
                                
                                neighbourhood_testing_samples_review.append(neighbourhood_model_array)
                                
                                #NN willing result
                                testing_ground_truth_review.append((review,targets_overall[review_index][2]))
                                
                                testing_ground_truth_rating.append(rating_review)
                                
                                count_testing+=1
                                
                                #test_ids.write(str(sample_file_dictionary['userId'])+" ----- "+str(sample_file_dictionary['businessId']))
                                #test_ids.write("\n")
                        
                        elif ( (count_development<755) and (count_testing>1000) ):
                        
                                if not( ((sample_file_dictionary['userId'],sample_file_dictionary['businessId']) in test_dict) or ((sample_file_dictionary['userId'],sample_file_dictionary['businessId']) in train_dict) ):
                        
                                        #feed the NN
                                        user_development_samples_review.append(user_model_array)
                                        
                                        product_development_samples_review.append(product_model_array)
                                        product_development_samples_rating.append(product_ratings_normalized)
                                        
                                        neighbourhood_development_samples_review.append(neighbourhood_model_array)
                                        
                                        #NN willing result
                                        development_ground_truth_review.append((review,targets_overall[review_index][2]))
                                        
                                        development_ground_truth_rating.append(rating_review)
                                        
                                        count_development+=1
                                        
                                        dev_ids.write(str(sample_file_dictionary['userId'])+" ----- "+str(sample_file_dictionary['businessId']))
                                        dev_ids.write("\n")
                        
                        else:
                                continue
                
                
                #just to keep track where we are
                count+=1
                
                #if ( (count_training>3000) and (count_testing>1755) ):
                #        break
                
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
                #count_to_stop+=1
                #if(count_to_stop==100):
                #enable the break command only for DEBUGGING PROCEDURE
                        #break
                print(sample_file_dictionary['userId'],'   ',sample_file_dictionary['businessId'],'   ',sample_file_dictionary['role'])
                #print('---------------------- Sample No. '+str(user_model_array.shape)+str(product_model_array.shape)+str(neighbourhood_model_array.shape)+str(count)+' / '+str(len(samples))+' ----------------------', end='\r')
                #sys.stdout.flush()
                
        
        #initialize training and testing lists
        user_training_samples_review = np.array(user_training_samples_review)
        user_testing_samples_review = np.array(user_testing_samples_review)
        user_development_samples_review = np.array(user_development_samples_review)
        
        #initialize training and testing lists
        product_training_samples_review = np.array(product_training_samples_review)
        product_testing_samples_review = np.array(product_testing_samples_review)
        product_development_samples_review = np.array(product_development_samples_review)
        
        #initialize training and testing lists
        neighbourhood_training_samples_review = np.array(neighbourhood_training_samples_review)
        neighbourhood_testing_samples_review = np.array(neighbourhood_testing_samples_review)
        neighbourhood_development_samples_review = np.array(neighbourhood_development_samples_review)
        
        
        #print('\n')
        training_ground_truth_review = np.array(training_ground_truth_review)
        #one hot encoding
        
        testing_ground_truth_review = np.array(testing_ground_truth_review)
        #one hot encoding
        
        development_ground_truth_review = np.array(development_ground_truth_review)
        #one hot encoding
        
        if(one_hot):
        
                training_ground_truth_review = keras.utils.to_categorical(training_ground_truth_review, num_classes=output_vocabulary_size)
                
                #print(training_ground_truth_review.shape)
        
                testing_ground_truth_review = keras.utils.to_categorical(testing_ground_truth_review, num_classes=output_vocabulary_size) 
                
                #print(testing_ground_truth_review.shape)
        
 
        #--------------------------------------------------------------------------------------------------------------------------> Training
        #save the sorted order        
        keep_order_training = make_order(training_ground_truth_review)
        
        #sort all the lists based on the index of the target
        user_training_samples_review = sort_based_on_list(keep_order_training, user_training_samples_review)
        neighbourhood_training_samples_review = sort_based_on_list(keep_order_training, neighbourhood_training_samples_review)
        product_training_samples_review = sort_based_on_list(keep_order_training, product_training_samples_review)
        product_training_samples_rating = sort_based_on_list(keep_order_training, product_training_samples_rating)
        training_ground_truth_review = sort_based_on_list(keep_order_training, training_ground_truth_review)
        training_ground_truth_rating = sort_based_on_list(keep_order_training, training_ground_truth_rating)
        
        
        minibatch_training = minibatch_object.Minibatch(user_training_samples_review, product_training_samples_review, neighbourhood_training_samples_review, product_training_samples_rating, training_ground_truth_review, training_ground_truth_rating)
        
        
        minibatch_training_file = open("training_LAST.pkl","wb")
        pickle.dump(minibatch_training, minibatch_training_file)
        minibatch_training_file.close()
        

        #--------------------------------------------------------------------------------------------------------------------------> Testing
        #save the sorted order        
        keep_order_testing = make_order(testing_ground_truth_review)
        
        
        user_testing_samples_review = sort_based_on_list(keep_order_testing, user_testing_samples_review)
        neighbourhood_testing_samples_review = sort_based_on_list(keep_order_testing, neighbourhood_testing_samples_review)
        product_testing_samples_review = sort_based_on_list(keep_order_testing, product_testing_samples_review)
        product_testing_samples_rating = sort_based_on_list(keep_order_testing, product_testing_samples_rating)
        testing_ground_truth_review = sort_based_on_list(keep_order_testing, testing_ground_truth_review)
        testing_ground_truth_rating = sort_based_on_list(keep_order_testing, testing_ground_truth_rating)
        
        
        minibatch_testing = minibatch_object.Minibatch(user_testing_samples_review, product_testing_samples_review, neighbourhood_testing_samples_review, product_testing_samples_rating, testing_ground_truth_review, testing_ground_truth_rating)
        
        minibatch_testing_file = open("testing_LAST.pkl","wb")
        pickle.dump(minibatch_testing, minibatch_testing_file)
        minibatch_testing_file.close()
        
        #--------------------------------------------------------------------------------------------------------------------------> Development
        #save the sorted order        
        keep_order_development = make_order(development_ground_truth_review)
        
        
        user_development_samples_review = sort_based_on_list(keep_order_development, user_development_samples_review)
        neighbourhood_development_samples_review = sort_based_on_list(keep_order_development, neighbourhood_development_samples_review)
        product_development_samples_review = sort_based_on_list(keep_order_development, product_development_samples_review)
        product_development_samples_rating = sort_based_on_list(keep_order_development, product_development_samples_rating)
        development_ground_truth_review = sort_based_on_list(keep_order_development, development_ground_truth_review)
        development_ground_truth_rating = sort_based_on_list(keep_order_development, development_ground_truth_rating)
        
        
        minibatch_development = minibatch_object.Minibatch(user_development_samples_review, product_development_samples_review, neighbourhood_development_samples_review, product_development_samples_rating, development_ground_truth_review, development_ground_truth_rating)
        
        minibatch_development_file = open("development_LAST.pkl","wb")
        pickle.dump(minibatch_development, minibatch_development_file)
        minibatch_development_file.close()
        
        
        tokenizer_file = open("tokenizer_LAST.pkl","wb")
        pickle.dump(tokenizer, tokenizer_file)
        tokenizer_file.close()
        
        return output_vocabulary_size
        
        
        
make_sorted_dataset() 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
