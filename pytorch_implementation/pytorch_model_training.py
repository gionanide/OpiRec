#!usr/bin/python
from pytorch_models import recommender_v1_encoder
from pytorch_models import recommender_v1_decoder
from pytorch_models import recommender_v2_decoder
import preprocessing
import torch
import evaluation
import sys
import word2vec
import numpy as np
import math
import random



def training(path, samples, output_vocabulary_size, targets, empty_text_reviews, normalize_user_reviews, normalize_product_reviews, normalize_neighbourhood_reviews, one_hot, padding, hidden_units, epochs, tokenizer, teacher_forcing):

        #--------------------------------------------------------> Pytorch properties

        #reproducibility
        torch.manual_seed(7)
        
        #define the decay function in order to control the trade off between feeding the model with it's output or the real output
        scheduled_sampling = lambda k,i : k/(k + math.exp(i/k))

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
        #recommender_decoder = recommender_v1_decoder.Recommender_v1_decoder(hidden_units, output_space)
        
        #initialize the network decoder --------------------------------------------------------------------------------------> Using teacher forcing, and DMN 
        recommender_decoder = recommender_v2_decoder.Recommender_v2_decoder(hidden_units, output_space)
        
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
        
        #define the function to handle the percentage of teacher forcing -- Initialization
        teacher_forcing_ratio = scheduled_sampling(5,0)
        #print(teacher_forcing_ratio)
        
        #keep track of the times that the model get as an input the actual word or it's prediction
        #counter_true = 0
        #counter_predicted = 0
        

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
                
                        '''
                        #use this if you want to test just for a small set
                        if (index==30):
                                
                                break
                                #print(index)
                        '''
                        
                
                        #keep track of sample loss
                        sample_loss = 0
                        
                        #print('index',index,'review_index',review_index)
                        
                        
                        #for maching the format, here if we want we can define a batch size
                        sample = [sample]
                        
                        #check if I have a signal of an Error
                        if (empty_flag):
                        
                                #if I faced an empty reivew I have to change my index to look back
                                review_index = review_index - 1

                                
                                
                        target = [targets[review_index]]

                        
                        #print(target.type) this is a list
                        #print(target[0].type) this is a list as well 

                        #erase all the words that are not in the pretrained model, which provide us with the embeddings
                        target = word2vec.clean_text_review(target, tokenizer)
                        
                       
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
       
                                
                        #take the decoder inputs
                        decoder_inputs = training_ground_truth
                        '''
                        #------------------------------------------------------------------------------> Prepare format for teacher forcing
                        print(target)
                        for word in target[0]:
                                word = preprocessing.index_to_word_mapping(word, tokenizer)
                                print(word)
                                word_vector = word2vec.word2vec(word)
                                #print(word_vector.shape)
                        '''
                        
                        
                        #removing the SOS symbol from the ground truth
                        training_ground_truth = np.delete(training_ground_truth[0], 0)
                        
                        #reshape the resulting array to keep the format
                        training_ground_truth = np.reshape(training_ground_truth, (1, training_ground_truth.shape[0]))
                                
                                
                                


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
                        
                        
                        if (teacher_forcing):
                        
                                #make the appropriate format in order to feed the previous word to the decoder -- the first word to feed decoder is SOS
                                decoder_inputs = torch.FloatTensor(torch.Size((1, 1, 300)))
                                #but their word embeddings can be arbitrary, small random numbers will do fine, because this vector would be equally far from all "normal" vectors.
                                torch.randn(torch.Size((1, 1, 300)), out=decoder_inputs)
                                decoder_inputs = torch.tensor(decoder_inputs, dtype=torch.torch.float32, device=device)
                                #print('Decoder initialization with SOS token',decoder_inputs)
                                #print('Decoder initialization with SOS token shape',decoder_inputs.shape)                                                       
                        
                        #print('\n\n ---------- End of Encoder ---------- \n\n')
                        
                        for target_word in training_ground_truth[0]:
                        
                                
                        
                                #print('Generating a word \n')
                                #print(target_word)

                                # ------------------------------------------------------------------- DECODER -----------------------------------------------------------
                                activations, decoder_hidden = recommender_decoder(user_lstm_output, user_h, neighbourhood_lstm_output, neighbourhood_h, product_lstm_output, product_h, decoder_hidden, decoder_inputs)
                                
                                
                                if (teacher_forcing):
                                
                                        #generate a random number between 0-1, the bigger the ratio the more probable the number to be smaller that it
                                        random_number = random.uniform(0, 1)
                                        #print('\n Random number:',random_number,'\n')
                                        
                                        if ( random_number < teacher_forcing_ratio ):
                                        
                                                #print('True target')
                                                #counter_true+=1

                                                #because the teacher forcing ratio in close to one, it is more probable that the number wil be lower that the ratio
                                                decoder_inputs = target_word
                                                
                                        else:
                                        
                                                #print('Predicted target')
                                                #counter_predicted+=1
                                        
                                                #otherwise feed the model with it's previous output, this becomes more probable as tha training goes on
                                                decoder_inputs = torch.argmax(activations).item()
                                                #print('Decodeeeer inputs',decoder_inputs)
                                        
                                        
                                        #make the appropriate format in order to feed the previous word to the decoder
                                        decoder_inputs = preprocessing.index_to_word_mapping(decoder_inputs, tokenizer)
                                        #print(decoder_inputs)
                                        decoder_inputs = word2vec.word2vec(decoder_inputs)
                                        decoder_inputs = torch.tensor(decoder_inputs, dtype=torch.torch.float32, device=device).unsqueeze(0).unsqueeze(0)


                                #iterating the ground thruth
                                target_word = torch.tensor(target_word, dtype=torch.long, device=device).unsqueeze(0)
                                #print('Target:',target_word)
                                
                                activations = activations.squeeze(0)
                                #print(activations)
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
                        if(batch_counter == 1):
                        
                                #change the rate every time a batch is passing through
                                teacher_forcing_ratio = scheduled_sampling(100,index)
                                #print(teacher_forcing_ratio)
                                
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
                        #print('\n')
                        #print('loss:',sample_loss.item())
                        #print('\n')
                        

                #print(overall_loss)
                #print(len(overall_loss))
                print('\n')
                print('epoch:',epoch+1,'loss:',sum(overall_loss)/len(overall_loss))
                print('\n')
                #break
        
        
        return recommender_encoder, recommender_decoder




#iterate the test samples every time an epoch is ended
def epoch_validation(test_samples, test_samples_ground_truth, recommender_encoder, recommender_decoder, device, teacher_forcing, tokenizer, criterion):
                
                #keep a list with the oveall lost
                overall_loss = []


                for sample_index, sample in enumerate(test_samples):
                
                
                        #initialize the loss
                        loss = 0
                
                
                
                        #split the input
                        user_inputs = sample[0]
                        product_inputs = sample[1]
                        neighbourhood_inputs = sample[2]
                        
                        #convert them to torch tensor and put the on GPU, use permute to change the axes of the tensor
                        user_inputs = torch.tensor(user_inputs, dtype=torch.float32, device=device).permute(1,0,2)
                        product_inputs = torch.tensor(product_inputs, dtype=torch.torch.float32, device=device).permute(1,0,2)
                        neighbourhood_inputs = torch.tensor(neighbourhood_inputs, dtype=torch.torch.float32, device=device).permute(1,0,2)
        
        
        
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
                        
                        
                        #print(testing_ground_truth)
                        
                        for target_word in test_samples_ground_truth[sample_index][0]:
                        
                                
                        
                                #print('Generating a word \n')
                                #print(target_word)
                        
                                activations, decoder_hidden = recommender_decoder(user_lstm_output, user_h, neighbourhood_lstm_output, neighbourhood_h, product_lstm_output, product_h, decoder_hidden, decoder_inputs)
                                

                                #iterating the ground thruth
                                target_word = torch.tensor(target_word, dtype=torch.long, device=device).unsqueeze(0)
                                #print('Target:',target_word.shape)
                                        
                                activations = activations.squeeze(0)
                                #print('Prediction:',activations.shape)
                                
                                #print('Target word:',target_word.shape)
                                #print(activations)
                                #print('Predicted word:',activations.shape)
                                
                                #find the position of the max element, which represents the word ID
                                predicted_index = torch.argmax(activations).item() + 1
                                
                                
                                if (teacher_forcing):
                                
                                        #print(index)
                                        #make the appropriate format in order to feed the previous word to the decoder
                                        decoder_inputs = preprocessing.index_to_word_mapping(predicted_index, tokenizer)
                                        #print(decoder_inputs)
                                        decoder_inputs = word2vec.word2vec(decoder_inputs)
                                        decoder_inputs = torch.tensor(decoder_inputs, dtype=torch.torch.float32, device=device).unsqueeze(0).unsqueeze(0)       
                                
                                #calculate the loss
                                loss += criterion(activations, target_word)
                                
                                
                        loss_current = loss.item()/len(test_samples_ground_truth[sample_index][0])
                
                
                        #print('loss',loss_current)
                        overall_loss.append(loss_current)
                        
                        
                loss_return = sum(overall_loss)/len(overall_loss)
                        
                        
                return loss_return


        
        
        
        
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
                                               
                
                #print('\n\n ---------- End of Encoder ---------- \n\n')
                
                
                #print(testing_ground_truth)
                
                for target_word in ground_truth[0]:
                
                        
                
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
                     
                    
                #make the appropriate formats
                prediction = predict_sentence
                ground_truth = ground_truth[0]
                
                #keep a record
                count_predictions = index

                                
                
                evaluation.make_sentence_one_by_one(prediction, ground_truth, target_reviews_length, tokenizer, role, count_predictions)

                        
                        
                #print('END OF SENTENCE')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
