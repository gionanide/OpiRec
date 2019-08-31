#!usr/bin/python
import preprocessing
from models import recommender_v2
from models import recommender_v1
from models import recommender_v3
from models import recommender_v4
import keras
import evaluation
import matplotlib.pyplot as plt


'''

We make this function because we have huge problems with memory handling. In this specific function we are loading only the samples we need in order to train our model. For example if the batch size is 5, we are loading 5 samples from the dataset, apply the appropriatte format and then train the model with only this five samples. We continue the procedure until we iterate all the samples from our dataset. To make a clarification the number of the samples that we are loading from our file is going to be our batch size but with a global meaning. An advantage of this procedure is that if we use a batch size(global) of 1 we are free of padding.

---- Padding problem ----

       We noticed that if the padding is too big, most of the times bigger that the sequence itself, our model is going to stick with the easy task. In this situation the easy task is for our model to predict only 0s. Let's analyze it further, if we have a sequence of 5 words and the padding size is 2000. If our model predict only zeros is going to be accurace for the 2000 of the 2005, so the error will be very small. That is the easy task for our model.

'''
def alternative_procedure(path, samples):


        #make this as True if you want to encode your target to one-hot encoding istead of a sequence of words IDs
        one_hot = True
        
        #define if you want padding or not
        padding = True
        
        #running on multiple GPUs
        parallel = True
        
        #keep a flag for the initialization
        initialization = True
        
        #define the number of maximum reviews for every sample
        normalize_reviews = 5
        
        #make the appropriatte format for the target
        targets, output_vocabulary_size, max_review_length, empty_text_reviews, eos, sos, tokenizer, target_reviews_length_train, target_reviews_length_test = preprocessing.format_target(path, samples, padding)
        
        
        #initialize some properties to split the dataset
        starting = 0        
        global_batch_size = 5
        
        #define the number of iterations based on the predefined global batch size
        div = round(len(samples) / global_batch_size)
        mod = len(samples) % global_batch_size
        
        print('div: ',div)
        print('mod: ',mod)
        
        #define the epochs of the training
        global_epochs = 2
        
        
        for global_epoch in range(global_epochs):
        
                #make the loss zero as we are starting a new epoch
                loss = []
                
                #assign them values again to initialize them for the next epoch
                starting = 0        
                global_batch_size = 5
                
                print('Going again')
        
                #make one more iteration to take the remaining samples
                for batch in range(div+1):
                
                        #just checking
                        #print(global_batch_size)
                        #print(len(samples))
                        #print(div)
                        #print(mod)
                        #print(starting)
                        print('epoch:',global_epoch+1,'out of:',global_epochs)
                        print('samples:',global_batch_size,'out of:',len(samples),'\n')
                        
                        #if we meet our goal, just take the other samples in the end of our dataset
                        if ( (batch == div) and (starting<len(samples)) ):                
                                
                                #take the remaining samples        
                                batch_samples = samples[starting:len(samples)]  
                                
                        elif( batch < div ):
                        
                                #take only samples as the global batch size defines        
                                batch_samples = samples[starting:global_batch_size] 
                                
                        else:
                        
                                print('End of the first epoch')
                                #skip all the below code and run the next epoch
                                continue
                        
                        
                        #going further
                        starting = global_batch_size                
                        global_batch_size = global_batch_size + 5
                      
                        user_training_samples,user_testing_samples,product_training_samples,product_testing_samples,neighbourhood_training_samples,neighbourhood_testing_samples,training_ground_truth,testing_ground_truth = preprocessing.make_training_testing(path, batch_samples, targets, empty_text_reviews, normalize_reviews, output_vocabulary_size, one_hot)
                        
                        #print(user_training_samples)
                        print('User: ',user_training_samples.shape)
                        #print(product_training_samples)
                        print('Product: ',product_training_samples.shape)
                        #print(neighbourhood_training_samples)
                        print('Neighbourhood: ',neighbourhood_training_samples.shape)
                        #print(training_ground_truth)
                        print('Ground truth: ',training_ground_truth.shape)
                        print('\n')



                        ################################### We have to initialize our model just once ##################################################
                        
                        
                        if(initialization):
                                
                                #--------------------------------> model properties
                                hidden_size = 500 #number of features to result with (input feature space is 300 from the embedding and with this hyper-parameter we decide to increase or decrease it)
                                        
                                        
                                #initialize the model
                                #model = recommender_v2.lstm_joint_network(user_training_samples, product_training_samples, neighbourhood_training_samples, training_ground_truth, max_review_length, normalize_reviews, hidden_size)
                                
                                #model = recommender_v3.lstm_joint_network(user_training_samples, product_training_samples, neighbourhood_training_samples, training_ground_truth, max_review_length, normalize_reviews, hidden_size)
                                
                                model = recommender_v4.lstm_joint_network(user_training_samples, product_training_samples, neighbourhood_training_samples, training_ground_truth, max_review_length, normalize_reviews, hidden_size, output_vocabulary_size)
                                
                                
                                if (parallel):
                        
                                        #pass the model so as to use parallelism with multiple GPUs
                                        parallel_model = keras.utils.multi_gpu_model(model, gpus=[0, 1])
                                        
                                else:
                                
                                        parallel_model = model
                                
                                
                                #loss = 'sparse_categorical_crossentropy'
                                #loss = 'mean_squared_error' #use this if we want to train to all the samples by once, because the target vector is not going to be to big
                                loss = 'categorical_crossentropy' #if we are going to validate one by one the samples and train loading in the memory only the samples to reach the batch size
                                
                                #define the learning rate
                                learning_rate = 0.0001
                                
                                #define if you want to decrease the learning rate as you are approaching the target
                                decay = 0
                                
                                #compile the model
                                algorithm = keras.optimizers.Adam(lr=learning_rate)
                                parallel_model.compile(optimizer=algorithm, loss=loss)
                                
                                
                                #change the flag
                                initialization=False
                                
                                

                        #properties of the training procedure
                        batch_size = 2
                        
                        #because I want to see the samples just one time, epoch here is not an epoch
                        epochs = 1                                             
                        
                        #define when you load a pretained model or not
                        pretrained = False
                        
                        
                        #----------------------------------------------------- Training procedure ------------------------------------------------
                        
                        #in case that I have a testing sample
                        if (user_training_samples.size == 0):
                        
                                #skip this iteration
                                continue
                        
                        
                        #training phase
                        parallel_model, loss = evaluation.train_model_alternative(parallel_model, user_training_samples, product_training_samples, neighbourhood_training_samples, training_ground_truth, batch_size, epochs, decay, learning_rate, loss, pretrained)
                        
                        print(loss[0])
                        #when I keep the history of the training the memory is overflowing
                        loss.append(loss[0])
                        
                        
                        
                #check some insights
                plt.figure(1)
                plt.plot(loss)
                plt.legend(['train loss'])
                plt.xlabel('cross entropy loss')
                plt.ylabel('epoch')
                plt.show(block=False)
                plt.pause(5)
                plt.close()
                
        
        #after all the epochs are finished we save the model        
        parallel_model.save_weights("recommender_new.h5")
        print("\n\n--------------------------------- Saved model to disk ---------------------------------\n\n")        
        
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                                                                                                                 

