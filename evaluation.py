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

we want this function to make predictions for the model, input the test set and the training set, and we want to make the sentence predictions.

'''       
def predictions(model, user, product, neighbourhood):


        predictions = model.predict([user, product, neighbourhood])

        
        #print('\n\n')
        #print('predictions shape: ',predictions.shape)
        #print(predictions)
        
        return predictions
        

      
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
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
