#!usr/bin/python
from __future__ import division
from scipy.spatial import distance
import pandas as pd
import numpy as np
import operator
from time import perf_counter as pc
import csv
import NN_feeding_sample as feeding_sample
import json
import sys
import word2vec as w2v


'''
read the file with all the reviews as a dataframe with the following (names) columns.
'''
def readReviewFile(path):
	#define the columns
	names = ['user_star','user_id','business_id','review_text']
	#read the file as dataframe
	data = pd.read_csv(path, names=names, engine='python', sep='###', index_col=False)
	#return an array (dtaframe) with number_of_reviews rows and our columns(names)
	return data

'''
We use this function to insert a txt file of the form x,y and conclude in a dictionary of the form dict[x]=y
'''
def read_txt_file_as_dict(path, reverse):
        #----------------------------------------------->  read the mappings as dictionaries
        with open(path, mode='r') as infile:
                reader = csv.reader(infile, delimiter=',')
                #dictionary with the following format
                #{businessId(key): column_number(value)}, example {naqJ8iKmZ1m9YWOyvgODZQ: 0}
                new_dict = {rows[0]:rows[1] for rows in reader}
                if (reverse):
                        #the inverse mapping, example {0: YWBuX2RBbwhYK1jGdZttJA}
                        new_dict = dict(map(reversed, new_dict.items()))        
        return new_dict
	
'''
We use this function to make a txt file which contains all the ratings for every business
'''
def make_business_rating_dict():
        #read the dictionary which contains every business Id and it's ratings
        mapping_business_dict = read_txt_file_as_dict('./mapping_business.txt',reverse=False)
        business_reviews_dict = read_txt_file_as_dict('./counter_business_review.txt',reverse=False)
        #iterate all the reviews
        path = './all_reviews.txt'
        #read the file as a dataframe
        data = readReviewFile(path)    
        #initialize the dictionary
        dict_business_ratings = {}
        #iterate all dataset
        for review in range(len(data)):
                #assign the fields
                userId = data.iloc[review]['user_id'].strip()
                businessId = data.iloc[review]['business_id'].strip()
                rating = data.iloc[review]['user_star']     
                #because some reviews do not exist, we are taking a 'Nan' value, so we use try-except to handle it
                try:
                        text_review = data.iloc[review]['review_text'].strip()        
                #handle nan value
                except AttributeError:
                        print('Nan value, no text review')
                        continue
                #before we append a rating we have to check if the user is valid(which means is has reviews higher than a threshold we predefined)
                if ( (int(business_reviews_dict[mapping_business_dict[businessId]]) >= 50) ):
                        #and then check if the specific business is already in the dictionary or I have to initialize it
                        if(businessId in dict_business_ratings):
                                #if a business already in the dictionary just append the rating to its list of ratings
                                dict_business_ratings[businessId].append((userId,rating))
                        else:
                                #if a business is not in the dictionary just initialize it
                                dict_business_ratings[businessId] = [(userId,rating)]       
        outputfile =  open('./businesses_ratings.txt','w+')        
        #write it to a file using json
        outputfile.write(json.dumps(dict_business_ratings))

'''
This function is implements the following procedures:
-- takes all the reviews and split the into two dictionaries as denoted below, dict_users, dict_businesses.
-- make two files (.txt) in order to map the userId with the corresponding row, and businessId with the corresponding column.
-- make the user (reviewer) item (business) matrix, to feed the matrix factorization procedure. It saves this matrix in order not to construct it again, for saving time.
-- besides the array construction, we made another format to feed the algorithms taken from Surprise library (userId, businessId, rating) as a dataframe instead of a matrix.
'''
def make_user_product_array(data):
	#I have to make two dictionaries
	# one for the users as follows, key: user_id {value: (business_id,rating to this business_id)}
	# and one for the businesses, key: business_id {value: new mapping (column)}
	#with this format for every business it's column number, it is tha value of it's real id, the same for the users
	dict_businesses = {}
	dict_users = {}
	for review in range(len(data)):
		user =  data.iloc[review]['user_id'].strip() # add .strip() to remove leading and ending spaces
		business =  data.iloc[review]['business_id'].strip()
		rating = data.iloc[review]['user_star']
		#review = data.iloc[review]['review_text'].strip()
		#we just append for every user, based on its id his rating and bid who rate
		#range of users dictionary [0, len(dict_users)-1]
		if user in dict_users:
			#if the user already in the dicionary we just append his ratings
			dict_users[user].append((business,rating,review))
		else:
			#if the user is not in the dictionary, we initialize him
			dict_users[user] = [(business,rating,review)]
		#add the business in the dictionary just to make the index
		if not(business in dict_businesses):
			# range of the dictionary[0,len(dict_businesses)-1]
			#only one initialization
			dict_businesses[business] = len(dict_businesses)
		#in this point we have one dictionary with all the users(unique records) and all their reviews
		#and another one dictionary with all the businesses and one index point at them, which is their 
		#position in the matrix that we are going to initialize
	#now I can initialize a matrix with dimensions (len(dict_users) which has all the users(unique) , len(dict_businesses) which has all the businesses(unique))	
	#additionaly I initialize the matrix with zeros in order to replace only the existing values and leave the unobserved(missing ratings) as zeros
	user_business_matrix = np.zeros((len(dict_users), len(dict_businesses)))
        #----------------------------------------------->  Making an array
	#we iterating all the users in order to fill the matrix
	for index,key_user in enumerate(dict_users):
		for review in dict_users[key_user]:
			# review[0] is the business id and review[1] is the rating that the user we iterate review the business id
			# recall that dict_businesses[review[0]] show us in which column to put the rating review[1]
			# we iterated the enumerate dictionary in order to give ID to users, which is their position in the dictionary
			#user_business_matrix review[0], review[1]
	                print ('row: ',index,'column: ',dict_businesses[review[0]],'rating: ',review[1])
	                user_business_matrix[index][dict_businesses[review[0]]] = review[1]
			#----------------------------------------------->  Making the corresponding dataframe
			#each row: userid, businessid, rating
			#row = str(index)+','+str(dict_businesses[review[0]])+','+str(review[1])
			#f.write(row)
			#f.write('\n')				
	#save the user-business-matrix to a file, in order to have them ready and not initialize it every time
	#np.savetxt('user-business-matrix.txt', user_business_matrix, fmt='%.0f', delimiter=',')   # X is an array
	#count the non zero elements in the matrix
	non_zero_count = np.count_nonzero(user_business_matrix)
	#count all the cells in the matrix
	matrix_cells = user_business_matrix.shape[0]*user_business_matrix.shape[1]
	#calculate the percentage of observations
	nonzero_percentage = (non_zero_count / matrix_cells)*100
	print ('From the ',matrix_cells,'cells only the',non_zero_count,'are not empty: ',nonzero_percentage,'% \n')

	return user_business_matrix

'''
We made this function to calculate the cosine similarity between two vectors:
-- 'users': corresponds to the adjusted cosine similarity with respect to the rows, we subtract from each row its mean. We do this because every users has his own profile, maybe a user rates all the movies with high rating, or another one is skeptic and is not rating too high. Because of this we subtract from every user his average rating in order to appear some latent similarities.
-- 'businesses': we do the same for the businesses. Some businesses are tend to be rated with a higher rating than other. We want to subtract this trend in order to reveal similarities.
-- 'full': we adjust with respect to both users and businesses.
'''
def cosine_dist(matrix1, neigbours, adjust):
        print('\n Calculate similarities \n')
        #in every user subtrack his mean
        if (adjust == 'users'):
                print('users adjustment')
                #axis=1 take the mean of every row
                users_means = np.mean(matrix1, axis=1)
                print(users_means.shape)
                #make an array with the same size as the users array, which have every column duplicated by the users_means column that we previous calculated
                users_means_array = np.array([users_means,]*matrix1.shape[1]).transpose()
                #print(users_means_array)
                print(users_means_array.shape)
                #we subtract from every users his mean rating
                matrix1 = matrix1 - users_means_array
        #in every bussines subtrack its mean
        elif (adjust == 'businesses'):
                print('businesses adjustment')
                #axis=0 take the mean of every column
                businesses_means = np.mean(matrix1, axis=0)
                print(businesses_means.shape)
                #make an array with the same size as the users array, which have every column duplicated by the users_means column that we previous calculated
                businesses_means_array = np.array([businesses_means,]*matrix1.shape[0])
                #print(businesses_means_array)
                print(businesses_means_array.shape)
                #we subtract from every users his mean rating
                matrix1 = matrix1 - businesses_means_array
        #full adjustment
        elif (adjust == 'full'):
                print('full adjustment')
                #axis=1 take the mean of every row
                users_means = np.mean(matrix1, axis=1)
                #axis=0 take the mean of every column
                businesses_means = np.mean(matrix1, axis=0)
                #make the array for users, span by column number
                users_means_array = np.array([users_means,]*matrix1.shape[1]).transpose()
                #make the array for businesses, span by row number
                businesses_means_array = np.array([businesses_means,]*matrix1.shape[0])
                #subtract from rows (users) and from columns (businesses)
                matrix1 = matrix1 - users_means_array - businesses_means_array        
        else:
                print('Do not want an adjustment')
        distance_matrix = np.zeros((matrix1.shape[0],matrix1.shape[0]))
        #we have to apply vector similarity among rows of the matrices
        for row in range(len(matrix1)):
                #apply one to the main diagonal to denote that it is the similarity of 
		#every row with itself, and this can mess up our job where we are seeking for the minimum distance
                distance_matrix[row,row] = 0
                #calculate the distance between the upper diagonal
                for every_row in range(row+1,len(matrix1)):
                        distance_matrix[row,every_row] = distance.cosine(matrix1[row,:], matrix1[every_row,:])
                        distance_matrix[every_row,row] = distance_matrix[row,every_row]
        #take only the k neigbours that are similar to every row
        min_positions = []
        for row in range(len(distance_matrix)):
                min_positions.append(np.argsort(distance_matrix[row,:], kind='quicksort')[:neigbours])
        '''
        #----------------------------------------------->  
	#Memory problems using this approach
        distance_matrix = np.genfromtxt('./distance_matrix_90-10.txt', delimiter=',')
        for row in range(len(distance_matrix)):
                min_positions.append(np.argsort(distance_matrix[row,:], kind='quicksort')[:neigbours])
        '''
  
        '''     
        #----------------------------------------------->  
	#With this method we read every time a line from the array        
        #because of memory problems we do not read all the matrix, instead we read row by row
        distance_matrix = open('./distance_matrix_80-20.txt','r')
        #open the file to write
        min_positions_file = open('./user_neigbours_80-20.txt','w+')
        #initialize the list to save the neigbours of the current user
        min_positions = []
        #iterate every line of the file
        for line in distance_matrix:
                #convert string into a numpy array
                line = np.fromstring(line, dtype=float, sep=',')
                #sort the distances, and keep the indeces
                min_positions.append(np.argsort(line, kind='quicksort')[:neigbours])
                #convert the numpy array to list
                min_positions_list = min_positions[0].tolist()
                print(len(min_positions_list))
                min_positions_list = ', '.join(str(element) for element in min_positions_list)
                print(min_positions_list)
                min_positions_file.write(min_positions_list)
                min_positions_file.write('\n')
                min_positions = []
                #break
        '''        
        print('Distance matrix',distance_matrix)
        print('Min positions for the nodes',min_positions)

        return distance_matrix #, min_positions
                
'''
#----------------> open a file to write every user and the number of his reviews
'''
def UsersRatingCounts(R):
        #open the dataframe witch contains all the reviews that have been made
        file_counts = open('./users_review_counter.txt','w+')
        #for every unique user count his reviews
        counts = R['userId'].value_counts()
        print('counts = ',counts)
        #sort the counts based on userId and convert it to dataframe
        counts = counts.sort_index(ascending=True).to_frame()
        counts.to_csv(file_counts, sep=',')
        
        return counts
       
'''
#----------------> open a file to write every business and the number of its reviews (how many review every business)
'''
def BusinessRatingCounts(R):
        #open the dataframe witch contains all the reviews that have been made
        file_counts = open('./business_review_counter.txt','w+')
        #for every unique user count his reviews
        counts = R['businessId'].value_counts()
        print('counts = ',counts)
        #sort the counts based on userId and convert it to dataframe
        counts = counts.sort_index(ascending=True).to_frame()
        counts.to_csv(file_counts, sep=',')
        
        return counts        
       
'''
#---------------------------> make a file with the mapping LargeUserId - smallUserId (index)
'''
def makeUserMapping():
        f_users = open('mapping_user.txt','w+')
        #we iterate all the users in order to fill the matrix
        for index, key_user in enumerate(dict_users):
                #iterate each review seperately
                for review in dict_users[key_user]:
                        #-------------------------------------> Users mapping
                        print('user: ',str(index))
                        print('user mapping: ',str(key_user))
                        users_row = str(key_user)+','+str(index)
                        f_users.write(users_row)
                        f_users.write('\n')
	        
'''	       
#---------------------------> make a file with the mapping LargeBusinessId - smallBusinessId (index)
'''    
def makeBusinessMapping():
	f_businesses = open("./mapping_business.txt","w+")
	#------------------------------------------> Businesses mapping
        for business in dict_businesses:
                print('business: ',str(business))
                print('business mapping: ',str(dict_businesses[business]))
                businesses_row = str(business)+','+str(dict_businesses[business])
                f_businesses.write(businesses_row)
                f_businesses.write('\n')
	               
'''
#------------------------------> make this function in order to keep the userIds that used in the training stage of matrix factorization procedure
'''
def userIdTrainingMF(trainset,path):
        f = open(path,'w+')
        f.write('userId,businessId,rating_groundTruth')
        f.write('\n')
        #itarate all the samples that participate in training (tuples)
        for rating in trainset.all_ratings():
                #take each field seperately, and write it to the file, but first convert them from the inner id's to the previos ones that we define
                uid = trainset.to_raw_uid(rating[0])
                iid = trainset.to_raw_iid(rating[1])
                rating_goundTruth = rating[2]
                new_row = str(uid)+','+str(iid)+','+str(rating_goundTruth)
                f.write(new_row)
                f.write('\n')
        f.close()

'''
#------------------------------> make this function in order to keep the userIds that used in the testing stage of matrix factorization procedure
'''
def userIdTestingMF(testset,path):
        f = open(path,'w+')
        f.write('userId,businessId,rating_goundTruth,rating_estimation')
        f.write('\n')        
        #iterate every sample exist in the testset
        for estimation in testset:
                #because the format is userId,businessId,rating assign the values accordingly
                userId = estimation.uid
                businessId = estimation.iid
                rating_goundTruth = estimation.r_ui
                rating_estimation = estimation.est
                new_row = str(userId)+','+str(businessId)+','+str(rating_goundTruth)+','+str(rating_estimation)
                f.write(new_row)
                f.write('\n')               
        f.close()
            
'''
make a function in order to subtract from the dataset the users that have a number of ratings below a given threshold
'''
def subtrackSparing_uid(R, threshold):
        print('Cleaning procedure \n')
        #for every unique user count his reviews
        counts = R['userId'].value_counts()
        #convert Series to dataframe
        counts = counts.to_frame()
        #after this we have subtract the rows of the dataframe that contains users with ratins below our threshold
        #change the index column, because it is the userId's
        counts['userID'] = counts.index
        #change column names, for interpretability
        counts.columns = ['count','userId']
        #we iterate the dataframe with the counts and we subtract from our original dataframe based on out threshold
        for count in range(len(counts)):
                #if the user has not enough reviews to charactirie him
                if(counts['count'].iloc[count] < threshold):
                        #print('Deleting userId: ',counts['userId'].iloc[count],'who has only: ',counts['count'].iloc[count],'reviews \n')
                        R = R[R.userId != counts['userId'].iloc[count]]
                        # we use break when we want to test it in one sample
                        #break
	return R

'''
make a function in order to subtract from the dataset the businesses that have a number of ratings below a given threshold
'''
def subtrackSparing_iid(R, threshold):
        print('Cleaning procedure \n')
        #for every unique business count his reviews
        counts = R['businessId'].value_counts()
        #convert Series to dataframe
        counts = counts.to_frame()
        #after this we have subtract the rows of the dataframe that contains businesses with number of ratings below our threshold
        #change the index column, because it is the businessId's
        counts['businessID'] = counts.index
        #change column names, for interpretability
        counts.columns = ['count','businessId']
        #we iterate the dataframe with the counts and we subtract from our original dataframe based on out threshold
        for count in range(len(counts)):
                #if the business has not enough reviews to charactirie it
                if(counts['count'].iloc[count] < threshold):
                        #print('Deleting businessId: ',counts['businessId'].iloc[count],'who has only: ',counts['count'].iloc[count],'reviews \n')
                        R = R[R.userId != counts['businessId'].iloc[count]]
                        # we use break when we want to test it in one sample
                        #break       
        return R

'''
Build another function which is reading all the dataset and it is assign to every ID, business or user all its/his reviews. Because we want to save computational cost.
'''
def makeBusinessUserRecords():
        path = './all_reviews.txt'
        start_time = pc()
        #read the file as a dataframe
        data = readReviewFile(path)
        end_time = pc()
        overall_time = end_time - start_time
        print('Time reading the file: ',overall_time)
        dict_business_user = {}
        #iterate all dataset
        for review in range(len(data)):
                #assign the fields
                userId = data.iloc[review]['user_id'].strip()
                businessId = data.iloc[review]['business_id'].strip()
                rating = data.iloc[review]['user_star']     
                #because some reviews do not exist, we are taking a 'Nan' value, so we use try-except to handle it
                try:
                        text_review = data.iloc[review]['review_text'].strip()
                #handle nan value
                except AttributeError:
                        print('Nan value, no text review')        
                #if the user is already in the dictionary just append the review
                if (userId in dict_business_user):
                        dict_business_user[userId].append(w2v.review2vec(text_review).tolist())
                #else inialize its instance
                else:
                        dict_business_user[userId] = [w2v.review2vec(text_review).tolist()]        
                #same procedure for businessId
                if(businessId in dict_business_user):
                        dict_business_user[businessId].append(w2v.review2vec(text_review).tolist())        
                else:
                        dict_business_user[businessId] = [w2v.review2vec(text_review).tolist()]      
        outputfile =  open('businesses_users_reviews_embeddings.txt','w+')        
        #write it to a file using json
        outputfile.write(json.dumps(dict_business_user))

'''
we make this function in order to make the format to feed our NN model, the format is as follows
{userId: }
{businessId: }
{rating_review: } --------------> predict userId's rating to businessId
{text_review: } ----------------> predict userId's text review to businessId
{estimation_rating: } ----------> the rating we estimate through the MF procedure
{user_text_reviews: } ----------> all user's text reviews (except the one to the specific business: businessId)
{business_text_review: } -------> all the business' text reviews (except the one from the specific user: userId)
{neighbourhood_text_reviews} ---> all the reviews that have been made from the users that are very similar to the specific user: userId
{role} -------------------------> we define the sample's role, training/testing in matrix factorization procedure, because we want to use the same sets in Neural procedure:
'''
def makeNNfeedFormat():
        path = './all_reviews.txt'
        start_time = pc()
        #read the file as a dataframe
        data = readReviewFile(path)
        end_time = pc()
        overall_time = end_time - start_time
        print('Time reading the file: ',overall_time)
        #------------------------------------------------------> read the mappings as dictionaries
        with open('./mapping_business.txt', mode='r') as infile:
                reader = csv.reader(infile, delimiter=',')
                #dictionary with the following format
                #{businessId(key): column_number(value)}, example {naqJ8iKmZ1m9YWOyvgODZQ: 0}
                dict_business_to_columnNumber = {rows[0]:rows[1] for rows in reader}                
                #inverse mapping, example {0: naqJ8iKmZ1m9YWOyvgODZQ}
                dict_columnNumber_to_business = dict(map(reversed, dict_business_to_columnNumber.items()))
                print('dict_columnNumber_to_business',len(dict_columnNumber_to_business))        
        with open('./mapping_user.txt', mode='r') as infile:
                reader = csv.reader(infile, delimiter=',')
                #dictionary with the following format
                #{userId(key): row_number(value)}, example {YWBuX2RBbwhYK1jGdZttJA: 0}
                dict_user_to_rowNumber = {rows[0]:rows[1] for rows in reader}                
                #the inverse mapping, example {0: YWBuX2RBbwhYK1jGdZttJA}
                dict_rowNumber_to_user = dict(map(reversed, dict_user_to_rowNumber.items()))
                print('dict_rowNumber_to_user',len(dict_rowNumber_to_user))
        with open('./user_neigbours_90-10.txt',mode='r') as infile:
                #because the row_index represent the row_number with is the userId after mapping it with the largeId, 
		#(dict_user dictionary is keeping this mapping)
                row_index = 0
                dict_user_neighbourhood={}
                #we have a file that the row denotes the user and evry row contains the neigbourhood of every user,
		#so we use the previous dictionary to make the mapping
                for line in infile:
                        #we use dict_rowNumber_to_user to map the row_index to userId
                        #example {dict_rowNumber_to_user[0]=YWBuX2RBbwhYK1jGdZttJA : 6342, 21622, 11004, 59890, ...... , 9805, 14942, 39910, 45644}
                        dict_user_neighbourhood[dict_rowNumber_to_user[str(row_index)]] = line[:-2].replace(" ", "").split(',') #last two characters '\n'
                        #go to the next user
                        row_index+=1
        with open('./TRAINING_uid-iid-rgt_90-10.txt',mode='r') as infile:
                reader = csv.reader(infile, delimiter=',')
                #the file is containing: userId,businessId,rating_groundTruth
                #our dictionary (userId, businessId(keys): rating_groundTruth(value))
                #example ( (0,0): 5.0 )
                dict_user_training = {(rows[0],rows[1]): rows[2] for rows in reader}              
                print('dict_user_training',len(dict_user_training))        
        with open('./TESTING_uid-iid-rgt-rest_90-10.txt',mode='r') as infile:
                reader = csv.reader(infile, delimiter=',')
                #the file in containing: userId,businessId,rating_goundTruth,rating_estimation
                #our dictionary (userId, businessId(keys): (rating_goundTruth,rating_estimation(value) )
                #example ( (8081,2848): (2.0,3.598731148340866) )
                dict_user_testing = {(rows[0],rows[1]): rows[3] for rows in reader}
                print('dict_user_testing',len(dict_user_testing))      
        print('Reading dictionaries \n')
        #read the file as dictionary, userId which contains all the reviews that this user made
        users_business_reviews = open('./all_businesses_users_reviews.txt','r').read()        
        dict_users_business_reviews = eval(users_business_reviews)        
        through_out=0
        not_in_neig=0
        #iterate the dataframe
        for review in range(len(data)):                
                sample = feeding_sample.Sample()
                #assign the fields
                userId_interest = data.iloc[review]['user_id'].strip()
                businessId_interest = data.iloc[review]['business_id'].strip()
                rating_interest = data.iloc[review]['user_star']   
                print('Going for',userId_interest,businessId_interest)  
                #because some reviews do not exist, we are taking a 'Nan' value, so we use try-except to handle it
                try:
                        text_review_interest = data.iloc[review]['review_text'].strip()
                #handle nan value
                except AttributeError:
                        print('Nan value, no text review')       
                #-----------------------------------------------> Assign values to the Object
                sample.set_userId(userId_interest)
                sample.set_businessId(businessId_interest)
                sample.set_ratingReview(rating_interest)
                sample.set_textReview(text_review_interest)
                #-----------------------------------------------> Check if the model is for training or testing
                #break
                tuple_to_check = (str(dict_user_to_rowNumber[userId_interest]),str(dict_business_to_columnNumber[businessId_interest]))
                #if the sample is in training set, we just add this to its role
                if (tuple_to_check) in dict_user_training:
                        sample.set_role('training')  
                #if the sample is in testing set, we add this to its role, and we add the estimation of the rating as well
                elif (tuple_to_check) in dict_user_testing:
                        sample.set_role('testing')
		sample.set_estimationRating(dict_user_testing[(dict_user_to_rowNumber[userId_interest],dict_business_to_columnNumber[businessId_interest])])   
                #if the sample is nowhere, something is wrong
                else:
                        #exit the program if the sample in not in the training or testing set
                        print('Sample is nowhere????????????????')
                        sys.exit()
                #we user the dictionary dict_users_business_reviews, which contains for a given user/business id all his/its reviews
                #--------------------------  SAVE COMPUTATIONAL COST
                #first of all we have to find the common review and subtract it from both 'bug' of reviews
                #we initialize three lists that contains the feedings for the three models
                user_model = []
                product_model = []
                user_neighbourhood_model = []
                #user_model_embeddings = []
                #product_model_embeddings = []
                #user_neighbourhood_model_embeddings = []
                #we copy it because we want our initial dictionary to remain untouchable
                user_model = dict_users_business_reviews[userId_interest].copy()
                #user_model_embeddings = dict_users_business_reviews_embeddings[userId_interest].copy()
                product_model = dict_users_business_reviews[businessId_interest].copy()
                #product_model_embeddings = dict_users_business_reviews_embeddings[businessId_interest].copy()
                #----------------------------------------------->  
		#Kick off bab users or products (with  3 or less reviews)
                if(len(user_model)<50 or len(product_model)<50):
                        print('Skip because of small number of reviews, ',userId_interest,businessId_interest)
                        through_out+=1
                        #skip this specific review
                        continue
                #subtrack the current review, which we want to predict
                user_model.remove(text_review_interest)
                product_model.remove(text_review_interest)
                #----------------------------------------------->  Embedding
                #first convert it to word embedding and then subtracted from the dictionary
                #text_review_interest_embedding = w2v.review2vec(text_review_interest).tolist()
                #user_model_embeddings.remove(text_review_interest_embedding)
                #product_model_embeddings.remove(text_review_interest_embedding)
                #----------------------------------------------->  Embedding
                neighbourhood_count = 0
                if(userId_interest in dict_user_neighbourhood):
                        #for every of the first k neighbours of every user, append all his reviews until you reach the number of 40
                        for neighbour in (dict_user_neighbourhood[userId_interest][:40]):
                                user_neighbourhood_model.append(dict_users_business_reviews[dict_rowNumber_to_user[neighbour]]) 
                                #user_neighbourhood_model_embeddings.append(dict_users_business_reviews_embeddings[dict_rowNumber_to_user[neighbour]])
                                neighbourhood_count+=len(dict_users_business_reviews[dict_rowNumber_to_user[neighbour]])    
                                if (len(user_neighbourhood_model)>40):
                                        break                         
                else:
                        not_in_neig+=1
                        print('userID: ',userId_interest,'reviews: ',len(user_model))  
                '''
                #-----------------------------------------------> 
		#We iterate all the dataset every time to find all the reviews for users/businesses, 
                #  ------------------------- COMPUTATIONALY EXPENSIVE ---------------------------
                #we initialize three lists that contains the feedings for the three models
                user_model = []
                product_model = []
                user_neighbourhood_model = []
                user_count=0
                product_count=0
                neighbourhood_count=0
                #we iterate all the reviews again in order to make our format
                for review in range(len(data)):
                        #assign the fields
                        userId = data.iloc[review]['user_id'].strip()
                        businessId = data.iloc[review]['business_id'].strip()
                        rating = data.iloc[review]['user_star']                               
                        #because some reviews do not exist, we are taking a 'Nan' value, so we use try-except to handle it
                        try:
                                text_review = data.iloc[review]['review_text'].strip()
                        #handle nan value
                        except AttributeError:
                                print('Nan value, no text review')
                        #----------------------------------------------->
			#User model
                        #now we check to find all the reviews for the user of interest except the review made for the business of interest
                        if(userId == userId_interest):
                                if(not(businessId == businessId_interest)):
                                        #if this two conditions are true we have all the reviews 
					#(except the one for the businessId_interest) for the user and we can feed the user model
                                        user_model.append(text_review)
                                        #print('user model')
                                        user_count+=1
                        #-----------------------------------------------> 
			#Product model
                        #which means (userId ==[not] userId_interest)
                        else:
                                #and this review is for the business of interest, we feed the product model
                                if(businessId == businessId_interest):
                                        product_model.append(text_review)
                                        #print('product model')
                                        product_count+=1                
                        #-----------------------------------------------> 
			#Neighbourhood model
                        #which means (userId ==[not] userId_interest) and we do not care if (businessId ==[not] / == businessId_interest) 
                        #else:
                                #if the user is not the user of interest, 
				#and the business is not the business of interest, 
				#we check if this user belongs to user's neighbourhood, if it is true we feed the neighbourhood model
                                if(dict_user_to_rowNumber[str(userId)] in dict_user_neighbourhood[userId_interest][:5]):
                                        user_neighbourhood_model.append(text_review)
                                        #print('neighbourhood model')
                                        neighbourhood_count+=1'''
                sample_dict = sample.make_field_dictionary(user_model[:40], product_model[:40], user_neighbourhood_model[:40])
                with  open('Feeding_samples_90-10/'+str(userId_interest)+'--------'+str(businessId_interest)+'.txt','w+') as outputfile:
                #for every sample we make a new file, otherwise maybe we have memory overflow
                        #write it to a file using json
                        outputfile.write(json.dumps(sample_dict))
                        outputfile.write('\n')  
                #print('File is ready: ',str(userId_interest)+'--------'+str(businessId_interest)+'.txt')
                print('\n\n -----------------------------> ',review)
                #break
        print(not_in_neig,through_out)




















