#!usr/bin/python
from Preprocessing import preprocessing
import numpy as np
from rouge import Rouge
               
'''       
use this function to decode one hot encoding to words Id
input: [0 0 0 1 .... 0 0 0 ]
output: [3]
'''
def decode_one_hot(prediction, mode, beam_deapth):
        temp_predictions = []
        if (mode == 'greedy_search'):
                #iterate this individual predictions to decode it to natural language, 
		#iterate every 'word' to find the word with the maximum probability and to assign the according ID
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
        elif(mode == 'beam_search'):
                #here we apply beam search to decode the sequence
                print('beam_search')    
        return temp_predictions       
    
#convert prediction into sentence
def make_sentence_one_by_one(prediction, ground_truth, target_reviews_length, tokenizer, role, count_predictions, rating_prediction, ground_truth_rating, print_sentence,max_review_length):

        if (print_sentence):
                print('\n\n\n\n\n\n --------------------',role,'--------------------')

        #initialize some properties
        target_str = ''
        predicted_str = ''
        rouge = Rouge()
        if (print_sentence):
                print('\n\n\n Sentence \n\n\n')
        count_words = 0 #keep track of the words
        if (print_sentence):
                print('\n ---------- Predicted ---------- \n')
                print('\n------ Rating prediction:',rating_prediction,'Ground truth rating:',ground_truth_rating,'------\n')
        for index, index_word in enumerate(prediction):
                word = preprocessing.index_to_word_mapping(index_word, tokenizer)
                #we are have two condition, the first one assumes that the model has learned when to stop 'speaking' and generates the word 'eos'
                #LIMITATIONS
                if (index==len(prediction)-1):
                        word='eos'
                predicted_str = predicted_str+' '+str(word)     
                if (print_sentence):   
                        print(word,end =" ")        
                count_words+=1
                if( (word=='eos') or (count_words==max_review_length) ):
                
                        if (print_sentence):
                                print('   |----- end of sentence -----|   ')
                        break
                #otherwise for more readable results we stop when we met the lenght of the ground truth
                '''
                if ( (word=='eos') or (count_words == target_reviews_length[str(count_predictions)]) ):
                
                        print('   |----- end of sentence -----|   ')
                        break
                '''
        if (print_sentence):
                print('\n')
                print('\n ---------- Target ---------- \n\n')
        for word in ground_truth:
                word = preprocessing.index_to_word_mapping(word, tokenizer)
                target_str = target_str+' '+str(word)
                if (print_sentence):       
                        print(word,end =" ")
                if ( (word=='eos') ):
                
                        if (print_sentence):
                                print('   |----- end of sentence -----|   ')
                        break
        score = rouge.get_scores(predicted_str, target_str)

        if (print_sentence):
                print('\n ---------- Rouge score ---------- \n')
                print(score)                         
        if (print_sentence):        
                print('\n\n\n')
        #return only the rouge-1 f score
        return predicted_str, target_str, score[0]['rouge-1']['f']
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
