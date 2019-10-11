#decode the predictions using beam search
def make_prediction_beam_search(path, recommender_encoder, recommender_decoder, samples, output_vocabulary_size, targets, empty_text_reviews, normalize_user_reviews, normalize_product_reviews, normalize_neighbourhood_reviews, one_hot, padding, tokenizer, target_reviews_length, role, teacher_forcing, max_id_to_keep, beam_depth):
          
          
          
          
        #because we want to run on GPUs
        device = torch.device('cuda:0')
        
        
        #initialize the flag, to catch some empty reviews
        empty_flag = False
        
        #initialize a queue for every new sample
        global_priority_queue = queue.PriorityQueue(beam_depth)

        #iterate all the samples, one by one
        for index, sample in enumerate(samples):
        
                #keep record of the predictions
                predict_sentences = [ [] for lists in range(beam_depth)]
        
                #for maching the format, here if we want we can define a batch size
                sample = [sample]
                target = [targets[index]]
                #print(sample)
                #print(target)
                
                
                #erase all the words that are not in the pretrained model, which provide us with the embeddings
                target = word2vec.clean_text_review(target, tokenizer)
        
                user_training_samples,user_testing_samples,product_training_samples,product_testing_samples,neighbourhood_training_samples,neighbourhood_testing_samples,training_ground_truth,testing_ground_truth,empty_flag = preprocessing.make_training_testing(path, sample, target, empty_text_reviews, normalize_user_reviews, normalize_product_reviews, normalize_neighbourhood_reviews, output_vocabulary_size, one_hot, empty_flag)
                
                
                if (role=='Training'):
                
                
                        #in case that I have a training sample, otherwise we feed our model
                        if (user_training_samples.size == 0):
                        
                                #skip this iteration
                                continue
                
                        #removing the SOS symbol from the ground truth
                        ground_truth = np.delete(training_ground_truth[0], 0)
                        
                        
                        #reshape the resulting array to keep the format
                        ground_truth = np.reshape(ground_truth, (1, ground_truth.shape[0]))
                        
                        
                        #convert them to torch tensor and put the on GPU, use permute to change the axes of the tensor
                        user_inputs = torch.tensor(user_training_samples, dtype=torch.float32, device=device).permute(1,0,2)
                        product_inputs = torch.tensor(product_training_samples, dtype=torch.torch.float32, device=device).permute(1,0,2)
                        neighbourhood_inputs = torch.tensor(neighbourhood_training_samples, dtype=torch.torch.float32, device=device).permute(1,0,2)
                        
                        
                        
                
                elif(role=='Testing'):
                
                        #in case that I have a training sample, otherwise we feed our model
                        if (user_testing_samples.size == 0):
                        
                                #skip this iteration
                                continue
                
                
                        print(testing_ground_truth)
                        
                        #removing the SOS symbol from the ground truth
                        ground_truth = np.delete(testing_ground_truth[0], 0)
                        
                        #reshape the resulting array to keep the format
                        ground_truth = np.reshape(ground_truth, (1, ground_truth.shape[0]))
                        
                        
                        #convert them to torch tensor and put the on GPU, use permute to change the axes of the tensor
                        user_inputs = torch.tensor(user_testing_samples, dtype=torch.float32, device=device).permute(1,0,2)
                        product_inputs = torch.tensor(product_testing_samples, dtype=torch.torch.float32, device=device).permute(1,0,2)
                        neighbourhood_inputs = torch.tensor(neighbourhood_testing_samples, dtype=torch.torch.float32, device=device).permute(1,0,2)
                        
                        
                        
                        

                #feed the model
                user_lstm_output, user_h, neighbourhood_lstm_output, neighbourhood_h, product_lstm_output, product_h = recommender_encoder(user_inputs, product_inputs, neighbourhood_inputs)
                
                
                #assign the first hidden state of the decoder
                decoder_hidden = product_h
                
                
                if (teacher_forcing):
                        
                                #make the appropriate format in order to feed the previous word to the decoder -- the first word to feed decoder is SOS
                                decoder_inputs = torch.FloatTensor(torch.Size((1, 1, 300)))
                                #but their word embeddings can be arbitrary, small random numbers will do fine, because this vector would be equally far from all "normal" vectors.
                                torch.randn(torch.Size((1, 1, 300)), out=decoder_inputs)
                                decoder_inputs = decoder_inputs.to(device)
                                #print('Decoder initialization with SOS token',decoder_inputs)
                                #print('Decoder initialization with SOS token shape',decoder_inputs.shape)  
                                               
                
                #print('\n\n ---------- End of Encoder ---------- \n\n')
                
                
                decoder_inputs = [decoder_inputs]
                
                
                #print(testing_ground_truth)
                
                for target_word_count, target_word in enumerate(ground_truth[0]):
                
                        #print('target_word_count: ',target_word_count)
                
                
                        #make a temporary word every time we are iterating a word to keep only the best paths and then add them in the global queue
                        temp_queue = queue.PriorityQueue(beam_depth)
                        
                        #print(temp_queue.empty())
                
                
                
                        #iterate every possible path
                        for decoder_count, decoder_input in enumerate(decoder_inputs):
                        
                        
                                #print('decoder count: ',decoder_count)
                
                        
                                #print('Generating a word \n')
                                #print(target_word)
                        
                                activations, decoder_hidden = recommender_decoder(user_lstm_output, user_h, neighbourhood_lstm_output, neighbourhood_h, product_lstm_output, product_h, decoder_hidden, decoder_input)
                                

                                #iterating the ground thruth
                                target_word = torch.tensor(target_word, dtype=torch.long, device=device).unsqueeze(0)
                                #print('Target:',target_word.shape)
                                
                                activations = activations.squeeze(0)
                                #print('Prediction:',activations.shape)
                                
                                #print('Target word:',target_word.shape)
                                #print(activations)
                                #print('Predicted word:',activations.shape)
                                
                                #find the position of the max element, which represents the word ID, returns (1, 1, 3)
                                values, indices = torch.topk(activations, k = beam_depth, dim=1)

                                #give all the first k indices with their values
                                decoder_inputs, priority_queue, temp_queue = Beam_Search.beam_search(global_priority_queue, values, indices, user_lstm_output, user_h, neighbourhood_lstm_output, neighbourhood_h, product_lstm_output, product_h, decoder_hidden, recommender_decoder, tokenizer, target_word, beam_depth, device, temp_queue, decoder_count)
                                
                                
                                #predict_sentence.append(indices[0].item())

                                #print(beam_deapth,': values ',values,'and indices: ',indices)
                                
                              
                        #we are applying the following procedure only after we have made the two prerequisit initialization steps
                        if (target_word_count>0):
                        
                        
                                #print('\n\n Going for the update step \n\n')
                              
                                #clean the decoder
                                decoder_inputs = []
                                        
                                #after I finish iterating all the possible outputs for the specific word, I am going to the next one. I have to Iterate the queue and take the last index of every path to feed the Decoder
                                for length in range(len(temp_queue.queue)):
                                
                                        #take the last element of every path because this will be the next word to feed the decoder
                                        last_path_index = temp_queue.queue[length][1][-1]
                                        
                                        #make the appropriate format in order to feed the previous word to the decoder
                                        decoder_input = preprocessing.index_to_word_mapping(last_path_index, tokenizer)
                                        #print(decoder_inputs)
                                        decoder_input = word2vec.word2vec(decoder_input)
                                        decoder_input = torch.tensor(decoder_input, dtype=torch.torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                                        
                                        decoder_inputs.append(decoder_input)
                                        
                                        
                                #before we start again the whole procedure we have to replace the temp queue with the global one, we are going one step forward
                                global_priority_queue = temp_queue
                                #print('New global queue: ',global_priority_queue.queue)

                #define the ground thruth
                ground_truth = ground_truth[0]
                
                #keep a record
                count_predictions = index

                
                #iterate all the K paths generated from Beam search
                while not priority_queue.empty():

                        #take the sentece with the lowest probability
                        sentence_properties = priority_queue.get()
                
                        #split the path and the probability
                        sentence_prob = sentence_properties[0]
                        predict_sentence = sentence_properties[1]

                        #make the appropriate formats
                        prediction = predict_sentence


                        evaluation.make_sentence_one_by_one(prediction, ground_truth, target_reviews_length, tokenizer, role, count_predictions)
                        
                        print('Sentece probability:',sentence_prob)
                        
                break

                        
                        
                #print('END OF SENTENCE')
