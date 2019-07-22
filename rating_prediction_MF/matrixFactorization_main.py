#!usr/bin/python
import numpy as np
import opinion_recommendation_help as recommenderHelp
import sklearn.decomposition
from sklearn.metrics import mean_squared_error
import scipy.sparse.linalg
import scipy.sparse as sps
import scipy.linalg
import math
from time import perf_counter as pc
import surprise
import pandas as pd
import random
import customMF as customMF_implemented






#------------------------------- in order to have reprodusable experiments
my_seed = 0

np.random.seed(my_seed)



'''
In this function we read the input data concerning the ratings and we construct the users-items matrix
'''
def readData():

        start_time = pc()

        #take the user-business matrix from opinion_recommendation python_script, we call this when we initialize the matrix
        
        '''
        #-----------------------> Initialization of the matrix
        path = 'review.txt'
        
        data = recommenderHelp.readReviewFile(path)
        
        user_business_matrix = recommenderHelp.make_user_product_array(data)
        
        print('\n End constructing the user-businesses matrix \n')
        
        print(user_business_matrix.shape)
        
        print(user_business_matrix)
        '''

        #because we already made the matrix now we only read it from the file, format: numpy user-item matrix
        #R = np.genfromtxt('/media/data/gionanide/OpinionRecommendation/user-business-matrix.txt', delimiter=',')


        print('\n\n')
        #---------------------------------> Quick example
        R = np.array([ [5, 3, 0, 1], [4, 0, 0, 1], [1, 1, 0, 5], [1, 0, 0, 4], [0, 1, 5, 4], ])
        
        print(R)
        print('\n\n')
        
        #R = pd.read_csv('/media/data/gionanide/OpinionRecommendation/Proceedings/example.txt',header=0)


        #------------------------------------------------------------------------------------------------------------------------> read Dataframe
        #read the file as dataframe, which is containg the information formated as follows: userId, bussinessId, rating
        
        
        #------------------------------------> contains a very small percentage of the total reviews, --- use this if you want to test something using a small example
        #R = pd.read_csv('/media/data/gionanide/OpinionRecommendation/Proceedings/part_of_the_reviews_dataframe.txt',header=0)
        
        
        #all the reviews, real set
        #R = pd.read_csv('/media/data/gionanide/OpinionRecommendation/Proceedings/user_business_matrix_dataframe.txt',header=0)
        

        #calculate the time
        end_time = pc()
       
        print('Matrix dimensions: ',R.shape)
        print('Time to read the matrix: ', end_time - start_time,'secs')
        
        
        return R





#because there are many dot products to calculate during the procedure, GPU can be used for this task
#use_gpu = True/False

#Skicit-Learn Non negative matrix factorization
def sklearnNMF(R):

        
        mf = sklearn.decomposition.NMF(n_components=3, init='nndsvd',random_state=0,alpha=0.1)
        
        
        #take only the non zero values, Compressed Sparse Row matrix
        formated_data =  sps.csr_matrix(R, shape=(R.shape[0], R.shape[1]))
        
        
        print(formated_data)
        

        W = mf.fit_transform(formated_data)
        print(W.shape)

        H = mf.components_
        print(H.shape)

        R_approx = np.dot(W,H)
        print(R)
        print(R.shape)
        print(R_approx)
        print(R_approx.shape)

        error = mean_squared_error(R, R_approx)

        print('RMSE: ',error)
        
        
        
        
def customMF(R):

        #define the custom mf
        mf = customMF_implemented.MF(R, K=2, alpha=0.1, beta=0.01, iterations=20, use_gpu=False)
        
        #training procedure
        training_process = mf.train()
        
        print('\n\n')
        print('Users in latent space: ',mf.P.shape)
        print('Items in latent space: ',mf.Q.shape)
        print('\n')
        print(R)
        print('\n')
        print(mf.full_matrix())
        



def sparseSVD(R):
#Scipy SVD for large and sparse arrays


        #convert our array to floatm so sparse svd can handle it
        R = R.astype(np.float32)

        print('\n Our matrix and its dimensions: ')
        print(R,R.shape)

        latent_features = 3
        #SVD decomposition
        u, s, v_t = scipy.sparse.linalg.svds(R, k=latent_features)

        #diagonalize matrix s in order to do the multiplications
        s = np.diag(s)

        print('\n U matrix with only k latent features: ')
        print(u,u.shape)
        print('\n S diagonal matrix with only k scalers of the latent features: ')
        print(s,s.shape)
        print('\n V transpose matrix with only k latent features: ')
        print(v_t,v_t.shape)
        print('\n')


        #multiple first the two matrices
        u_s = np.dot(u,s)
        #and the product with the remaining matrix, reconstructing the matrix R
        rec = np.dot(u_s,v_t)

        print('Our original matrix approximation using only k =',latent_features,' latent features: ')
        print(np.abs(rec), rec.shape)
        print('\n')


        error = math.sqrt(mean_squared_error(R, rec))
        print('RMSE: ',error,'stars')


        '''
        neigbours = 3

        distance_matrix, min_positions = recommender.cosine_dist(u, neigbours=neigbours)

        print('\n The Cosine distance between all the rows of reconstructed matrix: ')
        print(distance_matrix)

        
        print('\n Keeping only the',neigbours,'nearest neigbours: ')
        print(min_positions)
        '''
        
 
 
#SVD++ surprise library scikit learn 
def surpriseSVDpp(R):


        counts = R.rating.value_counts()
                
        
        #print('Dataframe: ')
        #print(R)
        #print('\n')
        
        print('Rating value counts:')
        print(counts)
        print('\n')
        
        
        #define the range of the ratings
        reader = surprise.Reader(rating_scale = (1,5))
                
        #call surprise library reader to make the appropriatte format 
        data = surprise.Dataset.load_from_df(R[['userId', 'businessId', 'rating']], reader)
        
        
        #we have to define minimum and maximum rating, we give static values because we already know the values
        minimum_rating = 1 #data['rating'].min()
        maximum_rating = 5 #data['rating'].max()
        print('Review range: {0} - {1}'.format(minimum_rating,maximum_rating))
        
        
        
        #initialization
        #SVD
        #algo_svd = surprise.SVD(lr_all=0.01, reg_all=0.02, n_factors=5, n_epochs=20, verbose=True)
        
        #SVD++
        algo_svdpp = surprise.SVDpp(lr_all=0.01, reg_all=0.02, n_factors=5, n_epochs=20, verbose=True)

        
        #------------------------> Grid search
        '''print('Grid search \n')
        
        param_grid = {'lr_all': [0.001, 0.01], 'reg_all': [0.01, 0.1]}
        
        gridSearch = surprise.model_selection.GridSearchCV(surprise.SVDpp, param_grid, measures=['rmse', 'mae'], cv=3)
        
        
        algo_svdpp_all = gridSearch.fit(data)
        
        print('Best params: ')
        print(gridSearch.best_params['rmse'])'''
        #-------------------------------> End grid search


        
        #-----------------------------------------------------> Training using all the dataset
        '''print('\n')
        print('Train the model using the training set \n')
        algo_svdpp.fit(data.build_full_trainset())
        
        
        algo_svd.fit(data.build_full_trainset())
        
        
        #test_size = 1, train taking all the set, and teting to all the set as well
        trainset, testset = surprise.model_selection.train_test_split(data, test_size=1)
        
        print(len(testset))
    
        print('SVDpp trained using all the samples: ')
        predictions_svdpp_testset = algo_svdpp.test(testset)
        
        
        print('\n SVDpp on testing set: ')
        accuracy_svdpp = surprise.accuracy.rmse(predictions_svdpp_testset)
        print('\n')
        
        print('SVD trained using all the samples: ')
        predictions_svd_testset = algo_svd.test(testset)
        
        
        print('\n SVD on testing set: ')
        accuracy_svd = surprise.accuracy.rmse(predictions_svd_testset)
        print('\n')
        '''
        
        
        '''#----------------------------------------------------------------------------------------------------------------------------> Cleaning dataframe
        print('Dataframe lenght: ',len(R['userId']),'\n')
        # we throw out the bad samples (samples with a low number of reviews, so it is misleading to characterize them)
        R = recommenderHelp.subtrackSparing_uid(R, threshold=2)
        print('Dataframe lenght after subtraction of users: ',len(R['userId']),'\n')
        
        print('Dataframe lenght: ',len(R['businessId']),'\n')
        # we throw out the bad samples (samples with a low number of reviews, so it is misleading to characterize them)
        R = recommenderHelp.subtrackSparing_iid(R, threshold=2)
        print('Dataframe lenght after subtraction of businesses: ',len(R['businessId']),'\n')
        #----------------------------------------------------------------------------------------------------------------------------> End cleaning dataframe'''
        
        
        #--------------------> Split to training testing
        trainset, testset = surprise.model_selection.train_test_split(data, test_size=0.2)
        
        #print(trainset.all_ratings())
        #print(testset)
        
        
        #-------------------------> Calcuate time of fitting
        start_time_svdpp = pc()
        
        algo_svdpp.fit(trainset)
        
        end_time_svdpp = pc()
        
        overall_svdpp = end_time_svdpp - start_time_svdpp
        print('SVDpp fit time: ',overall_svdpp)
        
        
        #start_time_svd = pc()
        
        #algo_svd.fit(trainset)
        
        #end_time_svd = pc()
        
        #overall_svd = end_time_svd - start_time_svd
        
        
        
        #------------------------------> Keep the matrices
        #users in latent space
        latent_users = algo_svdpp.pu
        
        #items in latent space 
        latent_items = algo_svdpp.qi
        
        #implicit preferences
        latent_implicit = algo_svdpp.yj
        
        #the users biases
        user_biases = algo_svdpp.bu
        
        #the items biases
        item_biases = algo_svdpp.bi
        
        
        print('Users in latent space: ',latent_users.shape)
        print('Items in latent space: ',latent_items.shape)
        print('Implicit preferences in latent space: ',latent_implicit.shape)
        print('Users biases: ',user_biases.shape)
        print('Items biases: ',item_biases.shape,'\n')
        
               
               
               
        predictions_svdpp_trainset = algo_svdpp.test(trainset.all_ratings())

        print('SVDpp on training set:')
        accuracy_svdpp = surprise.accuracy.rmse(predictions_svdpp_trainset)
              
        
        predictions_svdpp_testset = algo_svdpp.test(testset)
        #predictions_svd_testset = algo_svd.test(testset)
        
        #'''
        #------------------------------------------------------------------------------------------------------------------------> write the testset in a file
        #recommenderHelp.userIdTestingMF(predictions_svdpp_testset,path='Training-Testing_sets/TESTING_uid-iid-rgt-rest.txt')
        
        #------------------------------------------------------------------------------------------------------------------------> write the trainset in a file
        #recommenderHelp.userIdTrainingMF(trainset,path='Training-Testing_sets/TRAINING_uid-iid-rgt.txt')
        #'''
        
        
        print('Test set size: ',len(predictions_svdpp_testset),'\n')

        print('SVDpp on testing set:')
        accuracy_svdpp = surprise.accuracy.rmse(predictions_svdpp_testset)
        print('\n')
        
        #print('\n SVD on testing set: ')
        #accuracy_svd = surprise.accuracy.rmse(predictions_svd_testset)
        #print('\n')
        
        # Run 5-fold cross-validation and print results
        #surprise.model_selection.cross_validate(algo_svdpp, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
           
           
           
           
        #--------------------------------------------------------------------------------------------------------------------------------------. Saving the model, and evluate the procedure
        '''
        #---------------------------------------------------------> make one prediction before you save the model
        u_i_prediction_loaded = algo_svdpp.predict(uid=0,iid=0)
        print(u_i_prediction_loaded)
        print(u_i_prediction_loaded.est)  
        
       
       
        print('\n')
        print('Saving the model using Pickle')
        print('----------------------------------------')
        
        import pickle

        pickle_file = 'Proceedings/svdpp.pickle'
        
        
        with open(pickle_file, 'wb') as picklafile:
        
                pickle.dump(algo_svdpp, picklafile) 
                print('Succeed saving the model \n')   
                
        
        print('Loading the model using Pickle') 
        print('----------------------------------------')       
                
                
        with open('Proceedings/svdpp.pickle', 'rb') as picklafile:

                svdpp = pickle.load(picklafile) 
                print('Succeed loading the model')
                print('\n')
                
                
                #make the same prediction to compare the outcomes, and check if we result to the same estimation
                u_i_prediction_loaded = svdpp.predict(uid=0,iid=0)
                print(u_i_prediction_loaded)
                print(u_i_prediction_loaded.est)   
        '''
                
        return latent_users, latent_items, latent_implicit, user_biases, item_biases
        





if __name__ == '__main__':
#def main():

        print('Initialization')

        R = readData()
        
        print('Done reading the matrix')  
        
        
                #choose technique
        choice = input('Which version of matrix factorization you want to run \n skicit-learn NMF (1) \n custom NMF (2) \n sparse SVD Scipy(3) \n surprise SVD++(4) \n')
        
        print('------------> answer: ',choice)
        
        
        
        
        
        #------------------------------------------------------> Matrix Factorization
        print('\n Start matrix factorization procedure \n')
        
        #start calculating seconds
        start_time = pc()
        
        #Skicit-Learn Non negative matrix factorization
        if(choice=='1'):      

                sklearnNMF(R)
                
                print('\n Skicit-Learn Non negative matrix factorization \n')
                
        #Custon Implementation Matrix factorization
        elif(choice=='2'):

                customMF(R)
                
                print('\n Custon Implementation Matrix factorization \n')

        #Scipy SVD for large and sparse arrays
        elif(choice=='3'):

                sparseSVD(R)
                
                print('\n Scipy SVD for large and sparse arrays \n')
                     
        #SVD++ surprise library scikit learn 
        elif(choice=='4'):


                latent_users, latent_items, latent_implicit, user_biases, item_biases = surpriseSVDpp(R)
                        
                print('\n SVD++ surprise library scikit learn  \n')
                
                
                
        end_time = pc()

        print('Matrif Factorization procedure time: ', end_time - start_time,'secs')
        
        
        
        
        
        #---------------------------------------------------------------------------------------------------------------------------> Find neighbours
        #start_time = pc()
        
        #cosine distance
        #distance_matrix = recommenderHelp.cosine_dist(latent_users, neigbours=500, adjust='full')
        
        #end_time = pc()

        
        #distance_matrix is a numpy array (dimensions: numberofusers x numberofusers)
        #np.savetxt('min_positions.txt', min_positions, fmt='%.0f', delimiter=',')   # write the distance matrix in a file (saving time not to calculate it every time)

        #print('Time to calculate distance matrix and neighbours: ', end_time - start_time)
        
        #print(distance_matrix)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

