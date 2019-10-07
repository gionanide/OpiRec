#!usr/bin/python
import preprocessing
from matplotlib import pyplot as plt
import numpy as np
from pytorch_models import seq2seq
import evaluation
import torch
import math
import time as timer
import word2vec
import random
import sys


#reproducability
np.random.seed(7)

#ENBALE CuDNN benchmark
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True




if __name__ == '__main__':

        #file to read the sampes
        #path = '/media/data/gionanide/OpinionRecommendation/Feeding_samples_90-10_big' # -----------------> 53565 samples
        path = '/media/data/gionanide/OpinionRecommendation/Feeding_samples_90-10_medium/' # -------------> 2089 samples
        #path = '/media/data/gionanide/OpinionRecommendation/Feeding_smaples_90-10_small/' # ---------------> 512 samples
        #path = '/media/data/gionanide/OpinionRecommendation/Feeding_smaples_90-10_oneshot/' # ------------> 1 sample
        
        #return a list with all the samples, as paths
        samples = preprocessing.handle_input(path)
        
        #-----------------------------------------------------------------> Preprocessing properties
        
        #make this as True if you want to encode your target to one-hot encoding istead of a sequence of words IDs
        one_hot = False
        
        #define if you want padding or not
        padding = False
        
        #define the number of maximum reviews for every sample, if None we are taking all the existing reviews
        normalize_user_reviews = 49
        normalize_product_reviews = 49
        normalize_neighbourhood_reviews = 5
        
        #define the number of hidden units to use in the LSTM
        hidden_units_encoder = 300
        hidden_units_decoder = 600
        
        #because we want to run on GPUs
        device1 = torch.device('cuda:0')
        
        
       
        
        
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------> Train and save the model
        
        #define the decay function in order to control the trade off between feeding the model with it's output or the real output
        scheduled_sampling = lambda k,i : k/(k + math.exp(i/k))
        
        #define if you want to crop your vocabulary
        cut_bad_words = True
        
        #define the threshlod 
        count_thres = 50
        
        #erase the most frequent words
        erase_most_frequent_word = True
        
        #define the threshlod 
        max_id_to_keep = 30
                
        #make the appropriatte format for the target
        targets, output_vocabulary_size, max_review_length, empty_text_reviews, eos, sos, tokenizer, target_reviews_length_train, target_reviews_length_test = preprocessing.format_target(path, samples, padding, cut_bad_words, erase_most_frequent_word, max_id_to_keep, count_thres)
        
        #define if you want teacher forcing
        teacher_forcing = True
        
        #define training properties
        epochs = 35
        encoder_input_dropout = 0.2
        dropout_lstm_output = 0.4
        dropout_after_linear = 0.6
        bias_initialization_highway_network = -3
        episodes = 3
        
        #calculate GPU time, Wrapper around a CUDA event.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        #because our loss function has three losses we have to assign weights to each one of them
        review_weight = 0.4
        coverage_weight = 0.6
        rating_weight = 1
        
        #start measuring time
        start.record()
        
        #initialize the flag, to catch some empty reviews
        empty_flag = False
        
        #define the function to handle the percentage of teacher forcing -- Initialization
        teacher_forcing_ratio = scheduled_sampling(5,0)
        #print(teacher_forcing_ratio)
        
        #initialize the ratio
        teacher_forcing_ratio_decay = 0
        
        #make the appropriate format in order to feed the previous word to the decoder -- the first word to feed decoder is SOS
        decoder_inputs = torch.FloatTensor(torch.Size((1, 1, 300)))
        #small random numbers will do fine, because this vector would be equally far from all "normal" vectors.
        torch.randn(torch.Size((1, 1, 300)), out=decoder_inputs)
        decoder_inputs = decoder_inputs.to(device1)
        
        #for the review maybe we use Negative Log likelihood error
        criterion_review = torch.nn.NLLLoss().to(device1)
        
        #define the model
        seq2seq_model = seq2seq.Seq2Seq(hidden_units_encoder, encoder_input_dropout, hidden_units_decoder, output_vocabulary_size, dropout_after_linear, dropout_lstm_output, episodes, normalize_product_reviews, review_weight, criterion_review, coverage_weight, teacher_forcing_ratio, teacher_forcing, device1, tokenizer, eos).to(device1)
        
        #set the optimizer
        learning_rate = 0.01
        seq2seq_optimizer = torch.optim.Adam(seq2seq_model.parameters(), lr=learning_rate)
        
        #define the state of the model
        seq2seq_model.train()
        
        #start training the model
        for epoch in range(epochs):
        
                epoch_time = 0
        
                #keep track of the loss function
                overall_loss = []
                
                #keep recorder tof the review index due to some problems with the missing entries
                review_index = 0
                
                #keep testing samples, we are keeping the testing samples to validate our network in each iteration
                test_samples = []
                test_samples_ground_truth = []
                
                
                #iterate all the samples, one by one
                for index, sample in enumerate(samples):
                
                        #start counting
                        time_measure = timer.time()
                
                        
                        #--------------------------------------------------------------------------------> ONLY FOR DEBUGGING REASONS
                        '''
                        if (index==30):
                        
                                break
                        '''
                
                        #make the teacher forcing ratio smaller as the samples are passing by
                        teacher_forcing_ratio_decay+=1                        
                        
                        #for maching the format, here if we want we can define a batch size
                        sample = [sample]
                        
                        #check if I have a signal of an Error
                        if (empty_flag):
                        
                                #if I faced an empty reivew I have to change my index to look back
                                review_index = review_index - 1
        
        
                        target = [targets[review_index]]
                        #erase all the words that are not in the pretrained model, which provide us with the embeddings
                        target = word2vec.clean_text_review(target, tokenizer)


                        user_training_samples,user_testing_samples,product_training_samples,product_testing_samples,neighbourhood_training_samples,neighbourhood_testing_samples,training_ground_truth,testing_ground_truth, empty_flag = preprocessing.make_training_testing(path, sample, target, empty_text_reviews, normalize_user_reviews, normalize_product_reviews, normalize_neighbourhood_reviews, output_vocabulary_size, one_hot, empty_flag)

                        
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
                                
                                if not( (user_testing_samples.size == 0) or (product_testing_samples.size == 0) or (neighbourhood_testing_samples.size == 0) ):
                                
                                        #append the testing samples in the list
                                        test_samples.append((user_testing_samples, product_testing_samples, neighbourhood_testing_samples))
                                        test_samples_ground_truth.append(testing_ground_truth)
                        
                                #skip this iteration
                                continue                       
                        
                        #removing the SOS symbol from the ground truth
                        training_ground_truth = np.delete(training_ground_truth[0], 0)
                        
                        #reshape the resulting array to keep the format
                        training_ground_truth = np.reshape(training_ground_truth, (1, training_ground_truth.shape[0]))

                        #convert them to torch tensor and put the on GPU, use permute to change the axes of the tensor
                        user_inputs = torch.tensor(user_training_samples, dtype=torch.float32, device=device1).permute(1,0,2)
                        product_inputs = torch.tensor(product_training_samples, dtype=torch.torch.float32, device=device1).permute(1,0,2)
                        neighbourhood_inputs = torch.tensor(neighbourhood_training_samples, dtype=torch.torch.float32, device=device1).permute(1,0,2)
                        #print(user_inputs)
                        #print(product_inputs)
                        #print(neighbourhood_inputs)
                        
                        #initialize the coverage vector, every time we meet a new sample this vector is zero
                        coverage_vector = torch.zeros([1, 1, product_inputs.shape[0]], dtype=torch.float).to(device1)

                        #----------------------------------------------------------------------------> Zero grad before the model's forward pass                       
                        #gradient to zero
                        seq2seq_optimizer.zero_grad()


                        #make a prediction for just one sample
                        gradient_loss = seq2seq_model(user_inputs, product_inputs, neighbourhood_inputs, decoder_inputs, training_ground_truth)
                        
                        
                        #change the rate every time a batch is passing through
                        teacher_forcing_ratio = scheduled_sampling(1000,teacher_forcing_ratio_decay)
                        #print('teacher_forcing_ratio: ',teacher_forcing_ratio)
                        
                        #backpropagation
                        gradient_loss.backward()
                        
                        #append the loss
                        overall_loss.append(gradient_loss.item())
                        
                        #weight optimization
                        seq2seq_optimizer.step()
                        
                        #count the time for each epoch
                        epoch_time = epoch_time + (timer.time() - time_measure)
                

                        #print('\n')
                        print('--------------- epoch:',epoch+1,'out of:',epochs,'samples:',index+1,'out of:',len(samples),'loss:',sum(overall_loss)/len(overall_loss),'time:',epoch_time,'secs','---------------',end='\r')
                        sys.stdout.flush()
       
                        #update index
                        review_index = review_index + 1

                        
                        #IF WE WANT TO RUN THE TRAINING FOR JUST ONE SAMPLE, WE USE THIS break COMMAND
                        #break

                #break
                
        #change epoch
        print('\n')
        
        #end point
        end.record()
        
        
        # Waits for everything to finish running
        #torch.cuda.synchronize()

        #Returns the time elapsed in milliseconds after the event was recorded and before the end_event was recorded.
        miliseconds = start.elapsed_time(end)
        seconds = miliseconds/1000
        minutes = seconds/60
        hours = minutes/60
        print('GPU time:',miliseconds,'miliseconds')
        print('GPU time:',seconds,'seconds')
        print('GPU time:',minutes,'minutes')
        print('GPU time:',hours,'hours')
        
        
        #save the model
        torch.save(seq2seq, "/media/data/gionanide/OpinionRecommendation/pytorch_models/seq2seq.txt")
        print("\n\n--------------------------- Saved model to disk ---------------------------\n\n")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
