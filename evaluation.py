#!usr/bin/python
import keras
import matplotlib.pyplot as plt
import preprocessing
import numpy as np
from rouge import Rouge


#evaluation procedure
def train_model(model, user_training_samples, product_training_samples, neighbourhood_training_samples, training_ground_truth, user_testing_samples, product_testing_samples, neighbourhood_testing_samples, testing_ground_truth, batch_size, epochs, decay, learning_rate, loss, pretrained):


        if not(pretrained):

         	#---------------------------------------------------------------------------------------------------------------------> compile the model
                algorithm = keras.optimizers.Adam(lr=learning_rate)
                model.compile(optimizer=algorithm, loss=loss)
                
                
                #---------------------------------------------------------------------------------------------------------------------> fit network
                history = model.fit([user_training_samples, product_training_samples, neighbourhood_training_samples], training_ground_truth, validation_data=([user_testing_samples, product_testing_samples, neighbourhood_testing_samples], testing_ground_truth), batch_size=batch_size, epochs=epochs, verbose=1)
                

         #---------------------------------------------------------------------------------------------------------------------> Visualizing the results
                #check some insights
                plt.figure(1)
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.legend(['train loss','test loss'])
                plt.xlabel('mean squared error')
                plt.ylabel('epoch')
                plt.show()

        
        return model
        


'''

        
use this function to decode one hot encoding to words Id

input: [0 0 0 1 .... 0 0 0 ]
output: [4]

'''
def decode_one_hot(prediction):

        temp_predictions = []
                
        #iterate this individual predictions to decode it to natural language, iterate every 'word' to find the word with the maximum probability and to assign the according ID
        #print(prediction[0].shape)
        for element in prediction:
        
                #print('element in prediction:',element.shape)
                #find the maximum ID
                index_of_max_element = np.where(element == np.amax(element))
                
                #take it as an integer
                word_id = index_of_max_element[0][0]
                
                #append it to the list
                temp_predictions.append(word_id)
              
        #convert it to numpy array to feed the make sentence function
        temp_predictions = np.array(temp_predictions)
        #print(temp_predictions.shape)
        
        return temp_predictions        

        
      
'''

we want this function to make predictions for the model, input the test set and the training set, and we want to make the sentence predictions.

'''       
def predictions(model, user, product, neighbourhood):


        predictions = model.predict([user, product, neighbourhood])

        
        #print('\n\n')
        #print('predictions shape: ',predictions.shape)
        #print(predictions)
        
        return predictions
        
        
#iterate all the samples and predict one by one
def predict_one_by_one(role, user_samples, product_samples, neighbourhood_samples, ground_truths, model, target_reviews_length_train, tokenizer):

        count_predictions = 0 #keep track of the predictions
        
        #iterate every sample seperately to make the prediction, because of the aforementioned reasons (memory usage)
        for sample in range(user_samples.shape[0]):                
        
                #reshape all the prediction inputs because our model wants 3 dimensions inputs(batch, reviews, features))
                user_sample = np.reshape(user_samples[sample], (1, user_samples[sample].shape[0], user_samples[sample].shape[1]))
                #print(user_training_samples.shape)
                
                product_sample = np.reshape(product_samples[sample], (1, product_samples[sample].shape[0], product_samples[sample].shape[1]))
                #print(product_training_samples.shape)
                
                neighbourhood_sample = np.reshape(neighbourhood_samples[sample], (1, neighbourhood_samples[sample].shape[0], neighbourhood_samples[sample].shape[1]))
                #print(neighbourhood_training_samples.shape)
        
                #individual prediction
                prediction = predictions(model, user_sample, product_sample, neighbourhood_sample)
                #print(prediction.shape)
                #print('individual prediction shape:',prediction.shape)
                
                
                #decode one hot encoding to words ID
                temp_predictions = decode_one_hot(prediction[0])
                
                ground_truth = decode_one_hot(ground_truths[sample])
                
                #print(temp_predictions.shape)
                #print(ground_truth.shape)
                
                
                
                make_sentence(temp_predictions, ground_truth, target_reviews_length_train, tokenizer, role, count_predictions)
                
                count_predictions+=1
        
        

      
#convert prediction into sentence
def make_sentence(predictions, ground_truth, target_reviews_length, tokenizer, role):


        print('\n\n\n\n\n\n --------------------',role,'--------------------')


        #initialize some properties
        count_predictions = 0 #keep track of the predictions
        target_str = ''
        predicted_str = ''
        
        rouge = Rouge()
        
        for prediction in predictions:
        
                print('\n\n\n Sentence \n\n\n')
        
                
                count_words = 0 #keep track of the words
                print('\n ---------- Predicted ---------- \n')
                for word in prediction:
                
                        word = preprocessing.index_to_word_mapping(np.round(word), tokenizer)

                        predicted_str = predicted_str+' '+str(word)

                        print(word,end =" ")


                        #we are have two condition, the first one assumes that the model has learned when to stop 'speaking' and generates the word 'eos'
                        #LIMITATIONS
                        '''
                        if( word=='eos' ):
                        
                                print('   |----- end of sentence -----|   ')
                                break
                        '''

                        #otherwise for more readable results we stop when we met the lenght of the ground truth
                        
                        if ( (word=='eos') or (count_words == target_reviews_length[str(count_predictions)]) ):
                        
                                print('   |----- end of sentence -----|   ')
                                break
                        
                                
                                
                        count_words+=1
                                
                print('\n ---------- Target ---------- \n')
                for word in ground_truth[count_predictions]:
                
                
                        word = preprocessing.index_to_word_mapping(np.round(word), tokenizer)
                        
                        target_str = target_str+' '+str(word)
                        
                        print(word,end =" ")

                        
                        if ( (word=='eos') ):
                        
                                print('   |----- end of sentence -----|   ')
                                break
                           
                                
                #print(target_str)
                #print(predicted_str)
                score = rouge.get_scores(predicted_str, target_str)
                
                print('\n ---------- Rouge score ---------- \n')
                print(score)

                #initialize again for the next prediction
                target_str = ''
                predicted_str = ''
                count_predictions+=1
                
                if(count_predictions==2):
                
                        break
                        
                        
                        
        print('\n\n\n')
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
