#!usr/bin/python
import keras
import sys
#insert the path to shared file first and then import the scripts
sys.path.insert(0, '/media/data/gionanide/shared_python_scripts')
import gpu_initializations as gpu_init
import preprocessing
from matplotlib import pyplot as plt
import numpy as np
from models import recommender_v2
from models import recommender_v1
from models import recommender_v3
from models import recommender_v4
import evaluation


#reproducability
np.random.seed(7)




if __name__ == '__main__':

        #initialize some properties
        gpu_init.CUDA_init(core='GPU',memory='dynamically',parallel=True)
        
        #file to read the sampes
        #path = '/media/data/gionanide/OpinionRecommendation/Feeding_samples_90-10_big' # -----------------> 53565 samples
        #path = '/media/data/gionanide/OpinionRecommendation/Feeding_samples_90-10_medium/' # -------------> 2089 samples
        path = '/media/data/gionanide/OpinionRecommendation/Feeding_smaples_90-10_small/' # ---------------> 512 samples
        #path = '/media/data/gionanide/OpinionRecommendation/Feeding_smaples_90-10_oneshot/' # ------------> 1 sample
        
        #return a list with all the samples, as paths
        samples = preprocessing.handle_input(path)
        
        #make the appropriatte format for the target
        targets, output_vocabulary_size, max_review_length, empty_text_reviews, eos, sos, tokenizer, target_reviews_length_train, target_reviews_length_test = preprocessing.format_target(path,samples)
        
        #define the number of maximum reviews for every sample
        normalize_reviews = 5
        
        
        '''
        
        Returns:
        
        training_samples / testing_samples ---------------------> tuple (user_reviews, product_reviews, neighbourhood_reviews)
        
        training_ground_truth / testing_ground_truth ---------------------> tuple (rating_review, text_review)
        
        '''
        #make the appropriate format to feed the neural network, training testing samples
        user_training_samples,user_testing_samples,product_training_samples,product_testing_samples,neighbourhood_training_samples,neighbourhood_testing_samples,training_ground_truth,testing_ground_truth = preprocessing.make_training_testing(path, samples, targets, empty_text_reviews, normalize_reviews)
        
        
        print('\n')
        print('---------------------- Model properties ----------------------')
        print('---------------> Training')
        print('Model user input:',user_training_samples.shape)
        print('Model product input:',product_training_samples.shape)
        print('Model neighbourhood input:',neighbourhood_training_samples.shape)
        print('Model output:',training_ground_truth.shape)
        print('---------------> Testing')
        print('Model user input:',user_testing_samples.shape)
        print('Model product input:',product_testing_samples.shape)
        print('Model neighbourhood input:',neighbourhood_testing_samples.shape)
        print('Model output:',testing_ground_truth.shape)
        print('\n')
        
        
        #--------------------------------> model properties
        hidden_size = 500 #number of features to result with (input feature space is 300 from the embedding and with this hyper-parameter we decide to increase or decrease it)
        
        #initialize the model
        model = recommender_v3.lstm_joint_network(user_training_samples, product_training_samples, neighbourhood_training_samples, training_ground_truth, max_review_length, normalize_reviews, hidden_size)


        #pass the model so as to use parallelism with multiple GPUs
        parallel_model = keras.utils.multi_gpu_model(model, gpus=[0, 1])
 
        #properties of the training procedure
        batch_size = 32
        
        epochs = 200
        
        loss = 'mean_squared_error'
        
        decay = 0
        
        learning_rate = 0.0001
        
        #define when you load a pretained model or not
        pretrained = False

        #training phase
        model = evaluation.train_model(parallel_model, user_training_samples, product_training_samples, neighbourhood_training_samples, training_ground_truth, user_testing_samples, product_testing_samples, neighbourhood_testing_samples, testing_ground_truth, batch_size, epochs, decay, learning_rate, loss, pretrained)
        
        
        
        #model.save_weights("recommender_v_playingaround.h5")
        #print("\n\n--------------------------------- Saved model to disk ---------------------------------\n\n")
        
        #load weights into new model
        #parallel_model.load_weights("recommender_v1.h5")
        #parallel_model.load_weights("recommender_v2.h5")
        #parallel_model.load_weights("recommender_v3.h5")
        #print("\n\n--------------------------------- Loaded model from disk ---------------------------------\n\n")
        
        
        
        #----------------------------------------------------- Evaluation procedure ------------------------------------------------
        '''
        
        Having problem with memory with have to evaluate each sample seperately, and not load all the evaluation set in the memory, so we have to predict the output just for one review, decode it express it in natural language and then clear the memory to go to the next.
        
        '''
        
        print('Going for evaluation')
        
        #----------------------- uncomment this if you want to make all the predictions by once
        
        
        role='Training' 
        
        predictions = evaluation.predictions(model, user_training_samples, product_training_samples, neighbourhood_training_samples)
        
        evaluation.make_sentence(predictions, training_ground_truth, target_reviews_length_train, tokenizer, role)
        
        
        
        role='Testing'
        
        predictions = evaluation.predictions(model, user_testing_samples, product_testing_samples, neighbourhood_testing_samples)
        
        evaluation.make_sentence(predictions, testing_ground_truth, target_reviews_length_test, tokenizer, role)
        


        #-------------------------------------------------------- train one by one
        
        '''
        role = 'Training'
        
        evaluation.predict_one_by_one(role, user_training_samples, product_training_samples, neighbourhood_training_samples, training_ground_truth, parallel_model,target_reviews_length_train, tokenizer)
        

        
        role = 'Testing'
        
        evaluation.predict_one_by_one(role, user_testing_samples, product_testing_samples, neighbourhood_testing_samples, testing_ground_truth, parallel_model, target_reviews_length_test, tokenizer)
        '''
                
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
