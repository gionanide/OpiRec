#!usr/bin/python
import torch
import random
import numpy as np
import pickle
import sys
import math
import time as timer
from classes import dataset_loader
import os
from rouge import Rouge
import matplotlib.pyplot as plt
from pytorch_models import encoder
from pytorch_models import decoder
from classes import word2vec
from torchnlp.samplers.shuffle_batch_sampler import ShuffleBatchSampler
from torchnlp.samplers import SortedSampler
from progress.bar import Bar
import argparse
import random




ap = argparse.ArgumentParser()
ap.add_argument("--epochs", "--epochs", required=True, help="define from which epoch you want to load a model")
ap.add_argument("--batch_size", "--batch_size", required=True, help="define the number of samples to feed the model in every step")
ap.add_argument("--hidden_units", "--hidden_units", required=True, help="number of hidden units")
ap.add_argument("--shuffle", "--shuffle", required=True, help="shuffle the dataset before every epoch")
ap.add_argument("--pretrained", "--pretrained", required=True, help="load pretrained model")
ap.add_argument("--model", "--model", required=False, help="which pretrained model to use")
ap.add_argument("--encoder_input_dropout", "--encoder_input_dropout", required=True, help="dropout applied to word vector before lstm")
ap.add_argument("--dropout_lstm_output", "--dropout_lstm_output", required=True, help="dropout applied to the lstm output")
ap.add_argument("--dropout_after_linear", "--dropout_after_linear", required=True, help="dropout applied before the last linear layer, i.e., before classification")
ap.add_argument("--episodes", "--episodes", required=True, help="number of episodes in DMN")
ap.add_argument("--number_of_layers_encoder", "--number_of_layers_encoder", required=True, help="number of layers of the lstm encoder")
ap.add_argument("--number_of_layers_decoder", "--number_of_layers_decoder", required=True, help="number of layers of the lstm decoder")
ap.add_argument("--parallel", "--parallel", required=True, help="parallel and distributed training")
args = vars(ap.parse_args())


#XXX---------------------------------------------> DEBBUGING
import colorama
import torch
import pdb
import traceback
from colorama import Fore, Back, Style
from torch import autograd
colorama.init()
class GuruMeditation(autograd.detect_anomaly):  

        def __init__(self):
                super(GuruMeditation, self).__init__()
                
        def __enter__(self):
                super(GuruMeditation, self).__enter__()
                return self
                
        def __exit__(self, type, value, trace):
                super(GuruMeditation, self).__exit__()
                if isinstance(value, RuntimeError):
                        traceback.print_tb(trace)
                        #def halt(smsg):
                        print (Fore.RED + "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
                        print (Fore.RED + "┃ Software Failure. Press left mouse button to continue ┃")
                        print (Fore.RED + "┃        Guru Meditation 00000004, 0000AAC0             ┃")
                        print (Fore.RED + "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
                        print(Style.RESET_ALL)
                        print (str(value))
                        pdb.set_trace()
                        #halt(str(value))
#XXX---------------------------------------------> DEBBUGING



def reproducability(seed):
        #ENBALE CuDNN benchmark
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        
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
                word = word2vec.index_to_word_mapping(index_word, tokenizer)
                #we are have two condition, the first one assumes that the model has learned when to stop 'speaking' and generates the word 'eos'
                #LIMITATIONS
                if (index==len(prediction)-1):
                        word='eos'
                
                predicted_str = predicted_str+' '+str(word)     
                
                if (print_sentence):   
                        print(word,end =" ")        
                count_words+=1
                
                if( (word=='eos') or (count_words==max_review_length) ): #check if it is the END OF SEQUENCE SYMBOL
                        if (print_sentence):
                                print('   |----- end of sentence -----|   ')
                        break
                                          
        if (print_sentence):
                print('\n')
                print('\n ---------- Target ---------- \n\n')
        for word in ground_truth:
                word = word2vec.index_to_word_mapping(word, tokenizer)
                target_str = target_str+' '+str(word)
                if (print_sentence): #check if the sentence must be printed
                        print(word,end =" ")               
                if ( (word=='eos') ): #check if the symbol if the END OF SEQUENCE symbol
                        if (print_sentence):
                                print('   |----- end of sentence -----|   ')
                        break                   
        score = rouge.get_scores(predicted_str, target_str) #get ROUGE-1 score
        if (print_sentence):
                print('\n ---------- Rouge score ---------- \n')
                print(score)                                      
        if (print_sentence):        
                print('\n\n\n')
        #return only the rouge-1 f score
        return predicted_str, target_str, score[0]['rouge-1']['f']
       
#initialize loss weights based on Inverse Word Frequency
def initialize_loss_weights(language_output):
        
        word_counts = language_output.word2count
        word_index = language_output.word2index
        
        #initialize a list with all the words
        weights = [0]*(len(word_index)+1)
        for word, count in word_counts.items():
                #assign in its word its count
                weights[word_index[word]] = count
        #normalize the counts based on the max count, and inverse them (1-normalize)
        weights = [ (1-float(i)/max(weights)) for i in weights]
        #assign the the class 0, which is padding 0 weight
        #weights[pad_token] = 0.0 # PAD
        weights[0] = 0.0001 #PAD
        weights[1] = 0.001 #most common word
        weights[word_index["eos"]] = 0.1 # EOS
        weights[word_index["sos"]] = 0.1 # SOS
        return weights
        
#kill all the distributed processes
def clean_up_parallel():
        torch.distributed.destroy.process.group()
        
        
#set up the environment for the multiple processes
def set_up(rank, size):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=size)
        
        

# A seperate function to initialize mask
def initialize_mask(text_reviews):
        #take the elements that are non zero, 1 to the actuall tokens, 0 to the padded one
        mask = (text_reviews>0).float()
        
        return mask
        
        
#make a function which is masking our prediction, because we do not want to calculate the loss for the padded elements
def loss_mask(activations, text_reviews, mask):

        #print('Activations',activations.shape) ---> (b*seq, vocab)
        #print('Mask shape',mask.shape) ---> (b*seq, vocab)
        #apply the mask to the activations
        for index, activation in enumerate(activations):
                activation = activation*mask[index]

        return activations, text_reviews    
   
#append the sos token and erase the last one token    
def format_decoder_input(text_reviews,language_input):
    text_reviews_update = []
    for text_review in text_reviews:
        text_review = text_review.tolist()
        text_review.insert(0,language_input.word2index["sos"])
        text_review.pop()
        text_review = torch.LongTensor(text_review)
        text_reviews_update.append(text_review)
    return text_reviews_update
    
    
#shuffle multiple lists with the same order
def shuffle_dataset(user_reviews,product_reviews,neighbourhood_reviews,product_ratings,text_reviews,rating_reviews):
    print(len(user_reviews))
    print(len(product_reviews))
    print(len(neighbourhood_reviews))
    print(len(product_ratings))
    print(len(text_reviews))
    print(len(rating_reviews))
    join = list(zip(user_reviews,product_reviews,neighbourhood_reviews,product_ratings,text_reviews,rating_reviews))
    random.shuffle(join)
    user_reviews,product_reviews,neighbourhood_reviews,product_ratings,text_reviews,rating_reviews = zip(*join)
    print(len(user_reviews))
    print(len(product_reviews))
    print(len(neighbourhood_reviews))
    print(len(product_ratings))
    print(len(text_reviews))
    print(len(rating_reviews))
    return user_reviews,product_reviews,neighbourhood_reviews,product_ratings,text_reviews,rating_reviews

        

#training using minibatches
def minibatch_training():        
        
        sos_token = language_input.word2index["sos"]
        eos_token = language_input.word2index["eos"]
        pad_token = 0
        
        #initialize the network, encoder
        recommender_encoder = encoder.Overall_Encoder(hidden_units_encoder, encoder_input_dropout, episodes, normalize_product_reviews, number_of_layers_encoder, parallel,len(language_input.word2index),embedding_d_encoder,batch_size)
                
        beam_decode=False
        #Dynamic Memory Network
        recommender_decoder = decoder.DMN(hidden_units_decoder, output_space, dropout_after_linear, dropout_lstm_output, number_of_layers_decoder, parallel, device1, pad_token, eos_token, sos_token, beam_decode, language_output, embedding_d_decoder)
        
        
        if (pretrained=="True"):
                recommender_encoder.load_state_dict(torch.load('./pytorch_models/new_models/training/encoder_LAST'+str(model)+'.txt'))
                recommender_decoder.load_state_dict(torch.load('./pytorch_models/new_models/training/decoder_LAST'+str(model)+'.txt'))
        
        recommender_encoder.train()
        recommender_decoder.train()
        
        recommender_encoder = recommender_encoder.to(device1)
        recommender_decoder = recommender_decoder.to(device1)

        
        #------------------------------------------------------------------------------------------------------------> Loss
        #define a new Loss function based on Cross Entropy but giving much more weight on the difficult sample
        weights = initialize_loss_weights(language_output)
        class_weights = torch.FloatTensor(weights)
        criterion_review = torch.nn.CrossEntropyLoss(ignore_index=0, weight=class_weights, reduction='mean').to(device1)      
        
        #define the loss function for the rating prediction, we want to use RMSE so we have to use torch.sqrt(criterion(predicted, gound_truth))
        criterion_rating = torch.nn.MSELoss(reduction='mean').to(device1)


        ''' we try to penalize complexity using weight_decay. We add the sum of all weights and add it to the loss, If this is to big might we end up with a model which has all its weights to zero '''            
        encoder_lr = 0.001
        decay_encoder = 1e-6
        eps = 1e-7
        #encoder_optimizer = torch.optim.Adam(recommender_encoder.parameters(), lr=encoder_lr, weight_decay=decay_encoder, amsgrad=True)
        encoder_optimizer = torch.optim.AdamW(recommender_encoder.parameters(), lr=encoder_lr, weight_decay=decay_encoder,eps=eps)
        # gamma = decaying factor
        #encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=10, gamma=0.1)
        #encoder_scheduler = torch.optim.lr_scheduler.CyclicLR(encoder_optimizer, base_lr=0.00001, max_lr=0.01, step_size_up=5,step_size_down=10,cycle_momentum=False,mode='triangular2')# Scheduler
        
        decoder_lr = 0.001
        decay_decoder = 1e-6
        eps = 1e-7
        #decoder_optimizer = torch.optim.Adam(recommender_decoder.parameters(), lr=decoder_lr, weight_decay=decay_decoder, amsgrad=True)
        decoder_optimizer = torch.optim.AdamW(recommender_decoder.parameters(), lr=decoder_lr, weight_decay=decay_decoder,eps=eps)
        #decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=10, gamma=0.1)
        #decoder_scheduler = torch.optim.lr_scheduler.CyclicLR(decoder_optimizer, base_lr=0.00001, max_lr=0.01, step_size_up=5,step_size_down=10,cycle_momentum=False,mode='triangular2')# Scheduler   
        
        
        ''' define the decay function in order to control the trade off between feeding the model with it's output or the real output | #initialize the decay '''
        teacher_forcing = True
        scheduled_sampling = lambda k,i : k/(k + math.exp(i/k))
        teacher_forcing_ratio = scheduled_sampling(5,0)        
        teacher_forcing_ratio_decay = 0
        
        ''' DEFINE THE DISTRIBUTED SAMPLER: use it in order to ensure that every process will process different samples '''
        #train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, size, rank)

        #keep record of every loss of every batch
        overall_loss_review = []
        overall_loss_rating = []

        #training mode
        #encoder_decoder_parallel.train()
        
        print('----- Properties -----')
        print('encoder_input_dropout',encoder_input_dropout)
        print('dropout_lstm_output',dropout_lstm_output)
        print('dropout_after_linear',dropout_after_linear)
        print('episodes',episodes)
        print('output_space',output_space)
        print('number_of_layers_encoder',number_of_layers_encoder)
        print('number_of_layers_decoder',number_of_layers_decoder)
        print('encoder_lr',encoder_lr)
        print('decay_encoder',decay_encoder)
        print('eps',eps)
        print('decoder_lr',decoder_lr)
        print('decay_decoder',decay_decoder)
        print('batch_size',batch_size)
        
        min_review = 100 # just to keep track of the loss
        nan = False
        
        max_rouge = 0
        max_review_length = 300
        
        #TAKE ONLY THE FIRST BATCH
        #first = next(iter(train_dataset_loader))
        
        #with GuruMeditation():
        for epoch in range(epochs):
                
                epoch_time = 0
        
                overall_review_loss = []
                overall_rating_loss = []
                
                overall_review_loss_test = []
                overall_rating_loss_test = []
                
                overall_rouge_score = []
                
                bar = Bar("Training",max=round(len(train_dataset.text_reviews)/batch_size)-1)
                
                '''
                if(shuffle):
                    user_reviews_shuffled,product_reviews_shuffled,neighbourhood_reviews_shuffled,product_ratings_shuffled,text_reviews_shuffled,rating_reviews_shuffled = shuffle_dataset(train_dataset.user_reviews,train_dataset.product_reviews,train_dataset.neighbourhood_reviews,train_dataset.product_ratings,train_dataset.text_reviews,train_dataset.rating_reviews)
                else:
                    user_reviews_shuffled = train_dataset.user_reviews
                    product_reviews_shuffled = train_dataset.product_reviews
                    neighbourhood_reviews_shuffled = train_dataset.neighbourhood_reviews
                    product_ratings_shuffled = train_dataset.product_ratings
                    text_reviews_shuffled = train_dataset.text_reviews
                    rating_reviews_shuffled = train_dataset.rating_reviews
                '''
                
                hidden_encoder = recommender_encoder.initHidden()
                hidden_encoder = (hidden_encoder[0].to(device1),hidden_encoder[1].to(device1))
                #iterate all the batches
                for index in range(0,len(train_dataset.text_reviews)-batch_size,batch_size):
                #for index, (user_reviews, product_reviews, neighbourhood_reviews, product_ratings, text_reviews, rating_reviews, decoder_inputs) in enumerate([first]):
          
                        #XXX make the batch
                        user_reviews_list = train_dataset.user_reviews[index:index+batch_size]
                        product_reviews_list = train_dataset.product_reviews[index:index+batch_size]
                        product_ratings_list = train_dataset.product_ratings[index:index+batch_size]
                        text_reviews_list = train_dataset.text_reviews[index:index+batch_size]
                        decoder_inputs_list = format_decoder_input(text_reviews_list,language_input)
                        rating_reviews_list = train_dataset.rating_reviews[index:index+batch_size]

                        #XXX make batch tensors
                        user_reviews = torch.nn.utils.rnn.pad_sequence(user_reviews_list, batch_first=True, padding_value=0)
                        product_reviews = torch.nn.utils.rnn.pad_sequence(product_reviews_list, batch_first=True, padding_value=0)
                        text_reviews = torch.nn.utils.rnn.pad_sequence(text_reviews_list, batch_first=True, padding_value=0)
                        decoder_inputs = torch.nn.utils.rnn.pad_sequence(decoder_inputs_list, batch_first=True, padding_value=0)
                        product_ratings = torch.stack(product_ratings_list).float()
                        rating_reviews = torch.stack(rating_reviews_list).double()
                        
                        #free up memory
                        del user_reviews_list
                        del product_reviews_list
                        del text_reviews_list
                        del decoder_inputs_list
                        del product_ratings_list
                        del rating_reviews_list
                
                        #nan = torch.isnan(decoder_inputs)
                        #print('decoder_inputs',(nan!=0).nonzero())
                
                        #start counting
                        time_measure = timer.time()
                        
                        #make a list
                        orthogonality = []
                            
                        
                        #initialize lstm weights, we call this to denote that every input is independet of the previous
                        # XXX issue: https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426
                        #recommender_encoder.initHidden()
                        #recommender_decoder.initHidden()
                
                        #make grads zero
                        encoder_optimizer.zero_grad()
                        decoder_optimizer.zero_grad()
                        #mlt_optimizer.zero_grad()
                        
                        #take the output size, this is different for every batch
                        output_length =  text_reviews.shape[1]
                        
                        #print('User Input reviews',user_reviews.shape)
                        #print('Product input reviews',product_reviews.shape)
                        #print('Product reviews ratings',product_ratings.shape)
                        #print('Target review',text_reviews.shape)
                        #print('Target rating',rating_reviews.shape)
                        #print('Decoder inputs',decoder_inputs.shape)
                        #print('Output length:',output_length)
                        #print('Batch:',index)device1
                        #print('---------- new ----------')
                        
                        #initialize mask to use it in the prediction of rating
                        text_reviews = text_reviews.view(-1).to(device1)
                        mask = initialize_mask(text_reviews)
                        
                        
                        user_reviews = user_reviews.to(device1)
                        product_reviews = product_reviews.to(device1)
                        product_ratings = product_ratings.to(device1)
                        
                        #run the encoder
                        overall_context_vector_encoder, overall_attention_weights, decoder_hidden = recommender_encoder(user_reviews, product_reviews,index,hidden_encoder)
                        
                        
                        #activations_loss, activations, rating_prediction, repetitiveness_loss_back = encoder_decoder_parallel(user_reviews, product_reviews, output_length, product_ratings, decoder_inputs, mask, teacher_forcing_ratio)
                        
                        #overall_context_vector_encoder = overall_context_vector_encoder.to(self.device1)
                        #decoder_hidden = (decoder_hidden[0].to(self.device1),decoder_hidden[1].to(self.device1))
                        #overall_attention_weights = overall_attention_weights.to(self.device1)
                        decoder_inputs = decoder_inputs.to(device1)
                        #run the decoder
                        activations_loss, rating_prediction, repetitiveness_loss_back, target_length = recommender_decoder(overall_context_vector_encoder, decoder_hidden, decoder_inputs, output_length, overall_attention_weights, product_ratings, mask, teacher_forcing_ratio)
                        
                        # Each successive batch backpropagates through the timesteps of that batch AND the timesteps of all the previous batches
                        
                        
                        #make the teacher forcing ratio smaller as the samples are passing by
                        teacher_forcing_ratio_decay+=5
                        #change the rate every time a batch is passing through
                        teacher_forcing_ratio = scheduled_sampling(1000,teacher_forcing_ratio_decay)

                        #----------------------------------------------------------------------------------------> TEXT LOSS
                        #print(activations_loss)
                        nan = torch.isnan(activations_loss)
                        #print('overall',(nan!=0).nonzero())
                        #print(activations_loss[activations_loss==float('inf')])
                        activations_loss = activations_loss.view(-1, output_space)
                        review_loss = criterion_review(activations_loss, text_reviews)
                        overall_review_loss.append(review_loss.item())
                        #----------------------------------------------------------------------------------------> RATING LOSS
                        #move target to GPU, and change it's type
                        rating_reviews = rating_reviews.to(device1)
                        rating_prediction = rating_prediction.view(-1)
                        rating_prediction = rating_prediction.type(torch.DoubleTensor).to(device1)
                        #keep record of the rating prediction loss
                        rating_loss = criterion_rating(rating_reviews.squeeze(1), rating_prediction).type(torch.FloatTensor).to(device1)
                        overall_rating_loss.append(rating_loss.item())
                        
                        
                        #----------------------------------> XXX COMMAND TO CHECK IF THE LOSS IS NAN
                        if ( (math.isnan(review_loss)) or (math.isnan(rating_loss)) ):
                                print(epoch,index,'Deal with nan value in loss')
                                nan = True
                                sys.exit(0)
                                
                        #sum the loss
                        #loss = mlt(torch.stack([review_loss,rating_loss]))# + ortho_loss
                        loss = output_length*review_loss + rating_loss
                        
                        #backpropagate
                        loss.backward()
                        
                        del review_loss
                        del rating_loss


                        torch.nn.utils.clip_grad_norm_(recommender_encoder.parameters(), max_norm=0.5) #gradient clipping
                        torch.nn.utils.clip_grad_norm_(recommender_decoder.parameters(), max_norm=0.5) #gradient clipping
                        
                        encoder_optimizer.step()
                        decoder_optimizer.step()
                        #mlt_optimizer.step()
                        
                        #count the time for a batch
                        epoch_time = epoch_time + (timer.time() - time_measure)
                
                        activations_loss.detach()# DETACH THE HIDDEN STATES OF THE DECODER
                        
                #do not build the computational graph
                #with torch.no_grad():
        
                        #recommender_encoder.eval()
                        #recommender_decoder.eval()
                        
                        #role='Development'
                        #overall_rouge_loss, rating_loss_test, review_loss = validation(recommender_encoder,recommender_decoder,epoch,role)
                        
                        #role='Training'
                        #rouge_score_train, rating_loss_train = validation(recommender_encoder,recommender_decoder,epoch,role)
                        
                #print('Epoch:',str(epoch+1)+'/'+str(epochs),'Batch:',(str(index)+'/'+str(int(overall_batch))),'Samples',str(batch_size*(index+1))+'/'+str(length),'Output length:',output_length,'----- LEARNING ----- review:',"{0:.4f}".format(sum(overall_review_loss)/len(overall_review_loss)),'rating:',"{0:.2f}".format(sum(overall_rating_loss)/len(overall_rating_loss)),'Time:',"{0:.2f}".format(epoch_time),'----- DEV ----- review:',"{0:.2f}".format(review_loss),'rating',"{0:.2f}".format(rating_loss_test),'teacher_f_r',"{0:.2f}".format(teacher_forcing_ratio))
                
                        bar.next()
                        
                        
                        #free up memory
                        del user_reviews
                        del product_reviews
                        del text_reviews
                        del decoder_inputs
                        del product_ratings
                        del rating_reviews
                        torch.cuda.empty_cache()
                  
                        
                bar.finish()
                
                print('Ep:',str(epoch)+'/'+str(epochs),'--- LEARNING --- review:',"{0:.4f}".format(sum(overall_review_loss)/len(overall_review_loss)),'rating:',"{0:.4f}".format(sum(overall_rating_loss)/len(overall_rating_loss)),'Time:',"{0:.2f}".format(epoch_time)) 
                
                #recommender_encoder.train()
                #recommender_decoder.train()      
        
        
                #encoder_scheduler.step()
                #decoder_scheduler.step()
                #only from the first process
                #if (rank==0):
                #save the model after each epoch
                #encoder_decoder_parallel_name = 'encoder_decoder_parallel'
                #encoder_decoder_parallel.to(device2) #move the model to the other GPU
                
                #if(sum(overall_review_loss)/len(overall_review_loss)<min_review):
                #    min_review=sum(overall_review_loss)/len(overall_review_loss)
                torch.save(recommender_encoder.state_dict(), './pytorch_models/new_models/training/encoder_LAST'+str(epoch)+'.txt')
                torch.save(recommender_decoder.state_dict(), './pytorch_models/new_models/training/decoder_LAST'+str(epoch)+'.txt')
                #print("Save model")
                #encoder_decoder_parallel.to(device1) #move the mode back to other GPU
                #print('\n\n--------------------------- Saved model to disk with name:',encoder_decoder_parallel_name,' ---------------------------\n\n')
                        
        return recommender_encoder, recommender_decoder
        
        
#test and make the sentences
def validation(recommender_encoder,recommender_decoder,epoch,role):

        #print("\n\n--------------------------- Loaded model from disk ---------------------------\n\n")
        
        ''' define the decay function in order to control the trade off between feeding the model with it's output or the real output | #initialize the decay '''
        teacher_forcing_ratio = 0 #because we want the decoder to feed itself only with it's own outputs       
        
        if (role=='Testing'):
                #define the datalodaer
                dataloader = dataset_loader.Dataset_yelp('./data/testing_set.pkl', transform=None) #output_space = 8171
        elif (role=='Training'):
                #define the datalodaer
                dataloader = dataset_loader.Dataset_yelp('./data/training_set.pkl', transform=None) #output_space = 8171
        elif (role=='Development'):      
                #define the datalodaer
                dataloader = dataset_loader.Dataset_yelp('./data/development_set.pkl', transform=None) #output_space = 8171
        
        batch_size = 1
        train_sampler = ShuffleBatchSampler(SortedSampler(dataloader,sort_key=lambda i: len(i[4])),batch_size=batch_size,drop_last=False,shuffle=False)
        
        
        #define some proparties for the dataloader
        loader_params = {'num_workers':4, 'pin_memory':True, 'collate_fn':custom_collate, 'batch_sampler':train_sampler}
        dataloader = torch.utils.data.DataLoader(dataloader, **loader_params)
        
        
        #define the loss function for the review prediction
        weights = initialize_loss_weights()
        class_weights = torch.FloatTensor(weights).cuda()
        criterion_review = torch.nn.CrossEntropyLoss(ignore_index=0, weight=class_weights, reduction='mean').cuda() #to put the calcualtion into GPU
        
        
        #for the review maybe we use Negative Log likelihood error
        #criterion_review = torch.nn.NLLLoss().cuda()        
        #define the loss function for the rating prediction, we want to use RMSE so we have to use torch.sqrt(criterion(predicted, gound_truth))
        criterion_rating = torch.nn.MSELoss(reduction='mean').cuda()

        overall_review_loss = []
        overall_rating_loss = []
        
        overall_rouge_score = []
        
        max_review_length = 100
        
        #TAKE ONLY THE FIRST BATCH
        #first = next(iter(dataloader))
                
        #iterate all the batches
        for index, (user_reviews, product_reviews, neighbourhood_reviews, product_ratings, text_reviews, rating_reviews, decoder_inputs) in enumerate(dataloader):
        #for index, (user_reviews, product_reviews, neighbourhood_reviews, product_ratings, text_reviews, rating_reviews, decoder_inputs) in enumerate([first]):
                
                #change some properties of the input. Move it to GPU and make it float tensor
                user_reviews = user_reviews.type(torch.FloatTensor)
                product_reviews = product_reviews.type(torch.FloatTensor)
                #format the ratings
                product_ratings = product_ratings.type(torch.FloatTensor)
                #take the output size, this is different for every batch
                output_length =  text_reviews.shape[1]
                
                decoder_inputs[decoder_inputs!=decoder_inputs] = 0 #replace the nan values with zeros
                
                #print('User Input reviews',user_reviews.shape)
                #print('Product input reviews',product_reviews.shape)
                #print('Neighbourhood input reviews',neighbourhood_reviews.shape)
                #print('Product reviews ratings',product_ratings.shape)
                #print('Target review',text_reviews.shape)
                #print('Target rating',rating_reviews.shape)
                #print('Decoder inputs',decoder_inputs.shape)
                #print('Batch:',index)
                #print('---------- new ----------')
                
                #initialize mask to use it in the prediction of rating
                text_reviews_mask = text_reviews.view(-1).type(torch.LongTensor).to(device1)
                mask = initialize_mask(text_reviews_mask)
                
                user_reviews = user_reviews.to(device1)
                product_reviews = product_reviews.to(device1)
                product_ratings = product_ratings.to(device1)
                
                #run the encoder
                overall_context_vector_encoder, overall_attention_weights, decoder_hidden = recommender_encoder(user_reviews, product_reviews)
                
                
                #activations_loss, activations, rating_prediction, repetitiveness_loss_back = encoder_decoder_parallel(user_reviews, product_reviews, output_length, product_ratings, decoder_inputs, mask, teacher_forcing_ratio)
                
                #overall_context_vector_encoder = overall_context_vector_encoder.to(self.device1)
                #decoder_hidden = (decoder_hidden[0].to(self.device1),decoder_hidden[1].to(self.device1))
                #overall_attention_weights = overall_attention_weights.to(self.device1)
                decoder_inputs = decoder_inputs.to(device1)
                
                #run the decoder
                activations_loss, rating_prediction, repetitiveness_loss_back, target_length = recommender_decoder(overall_context_vector_encoder, decoder_hidden, decoder_inputs, output_length, overall_attention_weights, product_ratings, mask, teacher_forcing_ratio)
                

                #print(rating_prediction)
                #----------------------------------------------------------------------------------------> TEXT LOSS
                #replace Nan values with 0s
                #activations_loss[activations_loss==float('nan')] = 0
                activations = activations_loss.view(-1, output_space)
                text_reviews_loss = text_reviews.view(-1).to(device1)
                #print(activations_loss,text_reviews)
                
                review_loss = criterion_review(activations, text_reviews_loss)
                overall_review_loss.append(review_loss.item())
                
                
                #----------------------------------------------------------------------------------------> RATING LOSS
                #move target to GPU, and change it's type
                rating_reviews = rating_reviews.to(device1)
                rating_prediction = rating_prediction.view(-1)
                rating_prediction = rating_prediction.type(torch.DoubleTensor).to(device1)
                #keep record of the rating prediction loss
                rating_loss = criterion_rating(rating_reviews, rating_prediction).type(torch.FloatTensor).to(device1)
                #print(rating_loss)
                overall_rating_loss.append(rating_loss.item())
                #print(overall_rating_loss)
                
                #format the ratings
                #rating_prediction = rating_prediction.squeeze(1)
                
                activations_loss = activations_loss.detach()
                #iterate every sample in the batch
                for sample in range(activations_loss.shape[0]):
                        predicted_sentence = []
                        ground_truth = []
                        target_reviews_length = activations_loss[sample].shape[0] # take the number of timesteps
                        for word in range(activations_loss[sample].shape[0]):  
                                #find the position of the max element, which represents the word ID
                                predicted_index = torch.argmax(activations_loss[sample][word]).item()
                                ground_truth.append(text_reviews[sample][word].item())
                                predicted_sentence.append(predicted_index)
                                
                                
                        ground_truth_rating = rating_reviews[sample].item()
                        rating_prediction_item = rating_prediction[sample].item()
                        count_predictions=1
                        print_sentence = False

                        predicted_str, target_str, rouge_score = make_sentence_one_by_one(predicted_sentence, ground_truth, target_reviews_length, tokenizer, role, count_predictions, rating_prediction_item, ground_truth_rating,print_sentence,max_review_length)
                        overall_rouge_score.append(rouge_score)
                #print('-------------------- '+str(index)+' -------------------- \n\n')

        return sum(overall_rouge_score)/len(overall_rouge_score), sum(overall_rating_loss)/len(overall_rating_loss), sum(overall_review_loss)/len(overall_review_loss)
        
        
if __name__=="__main__":
    
        #initialize CUDE dependencies
        device1 = torch.device('cuda:0')
        device2 = torch.device('cuda:1')

        reproducability(12)
        
        epochs = int(args["epochs"])
        batch_size = int(args["batch_size"]) #run got one sample each tim
        embedding_d = int(args["hidden_units"])
        shuffle = args["hidden_units"]
        pretrained = args["pretrained"]
        model = args["model"]
        
        #XXX read the train dataset
        #train_dataset = pickle.load(open("./data/raw_train_set_PILOT.pkl",'rb'))
        train_dataset = pickle.load(open("./data/raw_train_set.pkl",'rb'))
        #load tokenizers
        language_input = pickle.load(open("./data/language_input_train.pkl",'rb'))
        language_output = pickle.load(open("./data/language_output_train.pkl",'rb'))

        #define the number of maximum reviews for every sample, if None we are taking all the existing reviews
        normalize_user_reviews = 40
        normalize_product_reviews = 100
        normalize_neighbourhood_reviews = 5

        #define the number of hidden units to use in the LSTM
        embedding_d_encoder = embedding_d
        embedding_d_decoder = embedding_d*2 # if lstm is bidirectional
        hidden_units_encoder = embedding_d
        hidden_units_decoder = embedding_d*2

        #define training properties
        encoder_input_dropout = float(args["encoder_input_dropout"])
        dropout_lstm_output = float(args["dropout_lstm_output"])
        dropout_after_linear = float(args["dropout_after_linear"])
        episodes = int(args["episodes"])
        output_space = len(language_output.word2index)+1
        number_of_layers_encoder = int(args["number_of_layers_encoder"])
        number_of_layers_decoder = int(args["number_of_layers_decoder"])
        parallel = args["parallel"]
        
        minibatch_training()
    
        
        
        
        
        
        
