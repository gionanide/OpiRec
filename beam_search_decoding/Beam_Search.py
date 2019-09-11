#!usr/bin/python
import queue
import word2vec
import preprocessing
import torch


def beam_search(priority_queue, values, indices, user_lstm_output, user_h, neighbourhood_lstm_output, neighbourhood_h, product_lstm_output, product_h, decoder_hidden, recommender_decoder, tokenizer, target_word, beam_deapth, device, temp_queue, decoder_count):
        
        decoder_inputs = []
        
        #if the priority queue is empty just fill it 
        if (priority_queue.empty()):
        
        
                #iterate all the results
                for iteration in range(len(indices[0])):
                
                        prob = values[0][iteration].item()
                        predicted_index = indices[0][iteration].item()
                        
                        #insert them in priority queue
                        priority_queue.put((prob, [predicted_index]))
                        
                        
                        #print('prob: ',prob,'index: ',predicted_index)
                        
                        
                        #make the appropriate format in order to feed the previous word to the decoder
                        decoder_input = preprocessing.index_to_word_mapping(predicted_index, tokenizer)
                        #print(decoder_inputs)
                        decoder_input = word2vec.word2vec(decoder_input)
                        decoder_input = torch.tensor(decoder_input, dtype=torch.torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                        
                        #now I have an array with all the k decoder inputs to feed it
                        decoder_inputs.append(decoder_input)
                        
                        
                        
                        
        #if the queue if not empty we have to replace some elements        
        else:
        
        
                #print('\n\n Queue not empty going for the temporary  queue \n\n')
                
                
                #-------------------------------------------------------------------------------------> This means that we are in the first iteration of the next word

                #if the priority queue is empty just fill it 
                if (temp_queue.empty()):
                
                
                        #take the path just once
                        path = priority_queue.queue[decoder_count][1]
                        #print('Path until now: ',path)
                
                
                        #iterate all the results
                        for iteration in range(len(indices[0])):
                        
                                #keep it temporary, and make a copy otherwise they will be linked
                                temp_path = path.copy()
                        
                                prob = values[0][iteration].item()*priority_queue.queue[decoder_count][0]
                                temp_path.append(indices[0][iteration].item())
                                
                                #insert them in priority queue, multiply the probabilities and add the next element in the path
                                temp_queue.put((prob, temp_path))
                                
                                
                                #print('prob: ',prob,'path: ',path)
                                
                                
                                
                                
                else:
                
                        #print('\n Temporary queue: ',temp_queue.queue,'\n')
                        
                        min_prob, min_prob_path = min(temp_queue.queue)
                        
                        #print('minimum probability in the queue: ',min_prob,'path: ',min_prob_path)
                        #print(temp_queue.queue)
                        
                        
                        #take the path just once
                        path = priority_queue.queue[decoder_count][1]
                        #print('Path until now: ',path)
                        
                                                
                        #iterate all the results
                        for iteration in range(len(indices[0])):
                        
                                #keep it temporary, and make a copy otherwise they will be linked
                                temp_path = path.copy()
                        
                                #take the joing probability of the path that is formulating
                                prob = values[0][iteration].item()*priority_queue.queue[decoder_count][0]
                                temp_path.append(indices[0][iteration].item())
                                
                                
                                if (prob > min_prob):
                                
                                        #remove the element with the lowest probability
                                        temp_queue.get()
                                
                                        #insert them in priority queue
                                        temp_queue.put((prob, temp_path))
                                        
                                        #find the new min the queue
                                        min_prob, min_prob_path = min(temp_queue.queue)
                        
                                        #print('minimum probability in the queue: ',min_prob,'path: ',min_prob_path)
                                
                                
                                #print('prob: ',prob,'index: ',predicted_index)
                                
        
        return decoder_inputs, priority_queue, temp_queue
