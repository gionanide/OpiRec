#!usr/bin/python
from pytorch_models import recommender_v1_encoder
from pytorch_models import recommender_v1_decoder
import preprocessing
import torch
import evaluation
import sys
import word2vec


def training(path, samples, output_vocabulary_size, targets, empty_text_reviews, normalize_user_reviews, normalize_product_reviews, normalize_neighbourhood_reviews, one_hot, padding, hidden_units, epochs, tokenizer):

        #--------------------------------------------------------> Pytorch properties

        #reproducibility
        torch.manual_seed(7)

        #because we want to run on GPUs
        device = torch.device('cuda:0')
        
        #set all the tensors type
        #torch.set_default_tensor_type('torch.cuda.LongTensor')
        
        #set default tensors
        #torch.set_default_tensor_type('torch.cuda.LongTensor')

        #define some properties, hidden units and the classification space
        output_space = output_vocabulary_size
        parallel = False
        
        #initialize the network, encoder
        recommender_encoder = recommender_v1_encoder.Recommender_v1_encoder(hidden_units)
        
        #initialize the network decoder
        recommender_decoder = recommender_v1_decoder.Recommender_v1_decoder(hidden_units, output_space)
        
        #check for paralellism
        if (parallel):
        
                #return the parallel model
                recommender_encoder = torch.nn.DataParallel(recommender_encoder)
                recommender_decoder = torch.nn.DataParallel(recommender_decoder)
                
                
        #define the loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        #and the optimizer properties
        learning_rate_enc = 0.001
        recommender_encoder_optimizer = torch.optim.Adam(recommender_encoder.parameters(), lr=learning_rate_enc)
        
        learning_rate_dec = 0.001
        recommender_decoder_optimizer = torch.optim.Adam(recommender_decoder.parameters(), lr=learning_rate_dec)
                        
        #going on GPU
        recommender_encoder = recommender_encoder.to(device)   
        recommender_decoder = recommender_decoder.to(device)
        
        #training properties
        #epochs = 100
        batch_size = 1
        
        #keep track of the batch 
        batch_counter = 0
        
        #initialize the flag, to catch some empty reviews
        empty_flag = False
        
        

        #start training
        for epoch in range(epochs):
        
                #keep track of the loss function
                overall_loss = []
                
                #gradient loss
                gradient_loss = 0
                
                #keep recorder tof the review index due to some problems with the missing entries
                review_index = 0
                
                #iterate all the samples, one by one
                for index, sample in enumerate(samples):
                
                        #keep track of sample loss
                        sample_loss = 0
                        
                        #print('index',index,'review_index',review_index)
                        
                        
                        #for maching the format, here if we want we can define a batch size
                        sample = [sample]
                        
                        #check if I have a signal of an Error
                        if (empty_flag):
                        
                                #if I faced an empty reivew I have to change my index to look back
                                review_index = review_index - 1
                                
                                print('HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')

                                
                                
                        target = [targets[review_index]]
                        
                        #------------------------------------------------------------------------------> Prepare format for teacher forcing
                        #print(target)
                        #for word in target[0]:
                        #        word = preprocessing.index_to_word_mapping(word, tokenizer)
                        #        print(word)
                        #        word_vector = word2vec.word2vec(word)
                        #        print(word_vector)
                                
                        #break
                        
                       
                        user_training_samples,user_testing_samples,product_training_samples,product_testing_samples,neighbourhood_training_samples,neighbourhood_testing_samples,training_ground_truth,testing_ground_truth, empty_flag = preprocessing.make_training_testing(path, sample, target, empty_text_reviews, normalize_user_reviews, normalize_product_reviews, normalize_neighbourhood_reviews, output_space, one_hot, empty_flag)
                        
                        
                        #just checking
                        #print('\n')
                        #print('---------------------- Model properties ----------------------')
                        #print('---------------> Training')
                        #print('Model user input:',user_training_samples.shape)
                        #print('Model user input:',user_training_samples)
                        #print('Model product input:',product_training_samples.shape)
                        #print('Model product input:',product_training_samples)
                        #print('Model neighbourhood input:',neighbourhood_training_samples.shape)
                        #print('Model neighbourhood input:',neighbourhood_training_samples)
                        #print('Model output:',training_ground_truth.shape)
                        #print('Model output:',training_ground_truth)
                        #print('---------------> Testing')
                        #print('Model user input:',user_testing_samples.shape)
                        #print('Model product input:',product_testing_samples.shape)
                        #print('Model neighbourhood input:',neighbourhood_testing_samples.shape)
                        #print('Model output:',testing_ground_truth.shape)
                        #print('\n')
                        
                        
                        #in case that I have a testing sample, otherwise we feed our model
                        if ( (user_training_samples.size == 0) or (product_training_samples.size == 0) or (neighbourhood_training_samples.size == 0) ):
                        
                                #update index before continue
                                review_index = review_index + 1
                        
                                #skip this iteration
                                continue
                                


                        #convert them to torch tensor and put the on GPU, use permute to change the axes of the tensor
                        user_inputs = torch.tensor(user_training_samples, dtype=torch.float32, device=device).permute(1,0,2)
                        product_inputs = torch.tensor(product_training_samples, dtype=torch.torch.float32, device=device).permute(1,0,2)
                        neighbourhood_inputs = torch.tensor(neighbourhood_training_samples, dtype=torch.torch.float32, device=device).permute(1,0,2)
                        #print(user_inputs)
                        #print(product_inputs)
                        #print(neighbourhood_inputs)


                        #feed the model
                        # ----------------------------------------------------------- ENCODER ---------------------------------------------------------
                        user_lstm_output, user_h, neighbourhood_lstm_output, neighbourhood_h, product_lstm_output, product_h = recommender_encoder(user_inputs, product_inputs, neighbourhood_inputs)
                        
                        
                        #assign the first hidden state of the decoder
                        decoder_hidden = product_h
                                                       
                        
                        #print('\n\n ---------- End of Encoder ---------- \n\n')
                        
                        for target_word in training_ground_truth[0]:
                        
                                
                        
                                #print('Generating a word \n')
                                #print(target_word)
                        
                                # ------------------------------------------------------------------- DECODER -----------------------------------------------------------
                                activations, decoder_hidden = recommender_decoder(user_lstm_output, user_h, neighbourhood_lstm_output, neighbourhood_h, product_lstm_output, product_h, decoder_hidden)
                                
                                #iterating the ground thruth
                                target_word = torch.tensor(target_word, dtype=torch.long, device=device).unsqueeze(0)
                                #print('Target:',target_word.shape)
                                
                                activations = activations.squeeze(0)
                                #print('Prediction:',activations.shape)
                                
                                #print(target_word)
                                #print('Target word:',target_word.shape)
                                #print(activations)
                                #print('Predicted word:',activations.shape)
                        
                        
                                #calculate the loss
                                sample_loss += criterion(activations, target_word)
                                gradient_loss += criterion(activations, target_word)
                                
                                #print('sample loss:',sample_loss)
                                
                                
                        loss_current = sample_loss.item()/len(training_ground_truth[0])
                        
                        
                        #print('loss',loss_current)
                        overall_loss.append(loss_current)
                        
                        #because I want to update the weights every 5 samples
                        batch_counter += 1

                        
                        #check when to make the gradient to zero
                        if(batch_counter == 5):
                                
                                #print('\n')
                                #print('Backpropagation')
                                #print('\n\n')
                                
                                #print('batch:',batch_counter,'loss:',gradient_loss.item())
                                
                                #backpropagation
                                gradient_loss.backward()
                                
                                #weight optimization
                                recommender_encoder_optimizer.step()
                                recommender_decoder_optimizer.step()
                                
                                #gradient to zero
                                recommender_encoder_optimizer.zero_grad()
                                recommender_decoder_optimizer.zero_grad()
                                
                                gradient_loss = 0
                                batch_counter = 0
                                
                                
                        #update index
                        review_index = review_index + 1
      
                        #print('\n')
                        print('--------------- epoch:',epoch+1,'out of:',epochs,'samples:',index+1,'out of:',len(samples),'---------------',end='\r')
                        sys.stdout.flush()
                        #print('loss:',sample_loss.item())
                        #print('\n')
                        

                print(overall_loss)
                print(len(overall_loss))
                print('\n')
                print('epoch:',epoch,'loss:',sum(overall_loss)/len(overall_loss))
                print('\n')
                #break
        
        
        return recommender_encoder, recommender_decoder

        
'''

Make this function in order to test our model to unseen data, we follow the training procedure with some variations.

'''        
def make_prediction(path, recommender_encoder, recommender_decoder, samples, output_vocabulary_size, targets, empty_text_reviews, normalize_user_reviews, normalize_product_reviews, normalize_neighbourhood_reviews, one_hot, padding, tokenizer, target_reviews_length, role):

        
        #because we want to run on GPUs
        device = torch.device('cuda:0')
        
        
        #initialize the flag, to catch some empty reviews
        empty_flag = False
        

        #iterate all the samples, one by one
        for index, sample in enumerate(samples):
        
        
                #keep record of the predictions
                predict_sentence = []
        
                #for maching the format, here if we want we can define a batch size
                sample = [sample]
                target = [targets[index]]
                #print(sample)
                #print(target)
        
                user_training_samples,user_testing_samples,product_training_samples,product_testing_samples,neighbourhood_training_samples,neighbourhood_testing_samples,training_ground_truth,testing_ground_truth,empty_flag = preprocessing.make_training_testing(path, sample, target, empty_text_reviews, normalize_user_reviews, normalize_product_reviews, normalize_neighbourhood_reviews, output_vocabulary_size, one_hot, empty_flag)
                

                #in case that I have a training sample, otherwise we feed our model
                if (user_testing_samples.size == 0):
                
                        #skip this iteration
                        continue




                #convert them to torch tensor and put the on GPU, use permute to change the axes of the tensor
                user_inputs = torch.tensor(user_testing_samples, dtype=torch.float32, device=device).permute(1,0,2)
                product_inputs = torch.tensor(product_testing_samples, dtype=torch.torch.float32, device=device).permute(1,0,2)
                neighbourhood_inputs = torch.tensor(neighbourhood_testing_samples, dtype=torch.torch.float32, device=device).permute(1,0,2)


                #feed the model
                user_lstm_output, user_h, neighbourhood_lstm_output, neighbourhood_h, product_lstm_output, product_h = recommender_encoder(user_inputs, product_inputs, neighbourhood_inputs)
                
                
                #assign the first hidden state of the decoder
                decoder_hidden = product_h
                                               
                
                #print('\n\n ---------- End of Encoder ---------- \n\n')
                
                
                #print(testing_ground_truth)
                
                for target_word in testing_ground_truth[0]:
                
                        
                
                        #print('Generating a word \n')
                        #print(target_word)
                
                        activations, decoder_hidden = recommender_decoder(user_lstm_output, user_h, neighbourhood_lstm_output, neighbourhood_h, product_lstm_output, product_h, decoder_hidden)
                        
                        #iterating the ground thruth
                        target_word = torch.tensor(target_word, dtype=torch.long, device=device).unsqueeze(0)
                        #print('Target:',target_word.shape)
                        
                        activations = activations.squeeze(0)
                        #print('Prediction:',activations.shape)
                        
                        #print('Target word:',target_word.shape)
                        #print(activations)
                        #print('Predicted word:',activations.shape)
                        
                        #find the position of the max element, which represents the word ID
                        index = torch.argmax(activations).item()

                        #print('index:',index)
                        
                        predict_sentence.append(index)
                     
                    
                        
                prediction = predict_sentence
                ground_truth = testing_ground_truth[0]
                print(len(predict_sentence))
                print(len(testing_ground_truth[0]))
                
                count_predictions = index
                                
                
                evaluation.make_sentence_one_by_one(prediction, ground_truth, target_reviews_length, tokenizer, role, count_predictions)

                        
                        
                #print('END OF SENTENCE')


