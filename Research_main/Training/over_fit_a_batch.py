#!usr/bin/python
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_models import batch_overall_encoder
from pytorch_models import batch_decoder_DMN
from pytorch_models import focal_loss
from pytorch_models import multitask_loss
import dataset_loader
import os
from Preprocessing import word2vec
from Preprocessing import preprocessing
import pickle
import sys
import math
import time as timer
from pytorch_models import orthogonal_regularization
from torchnlp.samplers.shuffle_batch_sampler import ShuffleBatchSampler
from torchnlp.samplers import SortedSampler
import random
from Keras import evaluation

#because we want to run on GPUs
device1 = torch.device('cuda:0')
device2 = torch.device('cuda:1')

#load the tokenizer
tokenizer_path = '/media/data/gionanide/OpinionRecommendation/data/REAL_tokenizer.pkl'
#tokenizer_path = '/media/data/gionanide/OpinionRecommendation/data/2088_reviews40_max100_minibatch_tokenizer.pkl'
#tokenizer_path = '/media/data/gionanide/OpinionRecommendation/data/2088_reviews40_tokenizer.pkl'

tokenizer = pickle.load(open(tokenizer_path,'rb'))

''' make the appropriate format in order to feed the previous word to the decoder -- the first word to feed decoder is SOS
but their word embeddings can be arbitrary, small random numbers will do fine, because this vector would be equally far from all "normal" vectors. '''
pad_emb = torch.load('/media/data/gionanide/OpinionRecommendation/data/special_chars/pad.pt')
eos_emb = torch.load('/media/data/gionanide/OpinionRecommendation/data/special_chars/eos.pt')
sos_emb = torch.load('/media/data/gionanide/OpinionRecommendation/data/special_chars/sos.pt')

#DEFINE EOS AND SOS TOKENs
sos_token = tokenizer.word_index['sos']
eos_token = tokenizer.word_index['eos']
pad_token = 0

print('<SOS>:',sos_token,'<EOS>:',eos_token,'<PAD>:',pad_token)

#ENBALE CuDNN benchmark
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

#reproducability
np.random.seed(1)
random.seed(0)

#define the number of maximum reviews for every sample, if None we are taking all the existing reviews
normalize_user_reviews = 40
normalize_product_reviews = 40
normalize_neighbourhood_reviews = 5

#define the number of hidden units to use in the LSTM
hidden_units_encoder = 300
hidden_units_decoder = 600

#define training properties
encoder_input_dropout = 0.2
dropout_lstm_output = 0.2
dropout_after_linear = 0.3
episodes = 3
output_space = 1903
number_of_layers_encoder = 2
number_of_layers_decoder = 2
parallel = False

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

#collect and combine
def custom_collate(batch):
        ''' For every batch we are finding the target with the biggest length, and we are padding all the samples in this batch based on this lenght '''
        
        #make a list ot keep only the target text reviews        
        target_reviews = []
        decoder_inputs = []
        
        #take only the target review of every sample
        for item in batch:
        
                        #removing the SOS symbol from the ground truth and then apply the padding
                        target_reviews.append(torch.tensor(np.delete(item[4], 0)))
                        decoder_inputs.append(torch.tensor(item[4]))
                        
        #pad all the target reviews based on the length of the bigger review             
        target_reviews_padded = torch.nn.utils.rnn.pad_sequence(target_reviews, batch_first=True, padding_value=0)
        decoder_inputs = torch.nn.utils.rnn.pad_sequence(decoder_inputs, batch_first=True, padding_value=0)
        
        ''' Use this function to transform decoder inputs from [b, seq] to [b, seq, features]'''
        decoder_inputs_tensor = torch.empty(decoder_inputs.shape[0], decoder_inputs.shape[1]+1, 300)
        for input_index, decoder_input  in enumerate(decoder_inputs[:][:-1]):
                for word_index, word_id in enumerate(decoder_input):
                        if (word_id==sos_token):
                                decoder_inputs_tensor[input_index][word_index] = sos_emb.squeeze(0).squeeze(0)
                        elif (word_id==eos_token):
                                decoder_inputs_tensor[input_index][word_index] = eos_emb.squeeze(0).squeeze(0)
                        elif (word_id==pad_token):
                                decoder_inputs_tensor[input_index][word_index] = pad_emb.squeeze(0).squeeze(0)
                        else:
                                word = preprocessing.index_to_word_mapping(word_id, tokenizer)
                                word = word2vec.word2vec(word, word_id)
                                word = torch.tensor(word, dtype=torch.torch.float32)
                                decoder_inputs_tensor[input_index][word_index] = word
        
        
        #assign the padded review to each sample        
        for index, item in enumerate(batch):
                        #update sample value in the batch
                        item_update = list(item)
                        item_update[4] = target_reviews_padded[index]
                        item_update[6] = decoder_inputs_tensor[index][:-1] #except the last element which is the EOS symbol
                        item = tuple(item_update)
                        batch[index] = item

        #return based on the default function
        return torch.utils.data.dataloader.default_collate(batch)
       
#initialize loss weights based on Inverse Word Frequency
def initialize_loss_weights():

        #we have the tokinzer ready
        word_counts = tokenizer.word_counts
        word_index = tokenizer.word_index
        
        #initialize a list with all the words
        weights = [0]*(len(word_index)+1)
        for word, count in word_counts.items():
                #assign in its word its count
                weights[word_index[word]] = count
        #normalize the counts based on the max count, and inverse them (1-normalize)
        weights = [ (1-float(i)/max(weights)) for i in weights]
        #assign the the class 0, which is padding 0 weight
        #weights[pad_token] = 0.0 # PAD
        weights[pad_token] = 0.0001 #PAD
        weights[1] = 0.001 #most common word
        weights[eos_token] = 0.1 # EOS
        weights[sos_token] = 0.1 # SOS
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
        

#training using minibatches
def minibatch_training():

        records_file = open('/media/data/gionanide/OpinionRecommendation/Training_proceedings/47202_reviews40_max100_minibatch_training_proceedings.txt','w')

        #initialize the processes
        #set_up(rank, size)     
                
        #modula the number of gpus with the rank of the process to assign equal processes to GPUs
        #gpus_number = torch.cuda.device_count()
        #device1 = (rank % gpus_number)
        
        #print('Rank:',rank,'Size:',size,'GPU:',device1)
        
        '''
        #A joint network
        #encoder_decoder_parallel = minibatch_enc_dec_parallel.Encoder_Decoder_Parallel(hidden_units_encoder, encoder_input_dropout, episodes, normalize_product_reviews, number_of_layers, parallel, hidden_units_decoder, output_space, dropout_after_linear, dropout_lstm_output, device1, tokenizer, pad_emb, eos_emb, sos_emb, pad_token, eos_token, sos_token)     
        
        #after initializing the model, if we want to continue training we load the model      
        #encoder_decoder_parallel.load_state_dict(torch.load('/media/data/gionanide/OpinionRecommendation/pytorch_models/new_models/encoder_decoder_parallel.txt',map_location={'cuda:0':'cuda:0'}))
        
        encoder_decoder_parallel.train()
        
        encoder_decoder_parallel = encoder_decoder_parallel.to(device1)
        
        #encoder_decoder_parallel = torch.nn.parallel.DistributedDataParallel(encoder_decoder_parallel, device_ids=[device1], find_unused_parameters=True)
           
        #check for paralellism
        #if (parallel):
                #return the parallel models
        #        encoder_decoder_parallel = torch.nn.DataParallel(encoder_decoder_parallel).to(device1)
        '''
        
        #initialize the network, encoder
        recommender_encoder = batch_overall_encoder.Overall_Encoder(hidden_units_encoder, encoder_input_dropout, episodes, normalize_product_reviews, number_of_layers_encoder, parallel)
                
        beam_decode=False
        #Dynamic Memory Network
        recommender_decoder = batch_decoder_DMN.DMN(hidden_units_decoder, output_space, dropout_after_linear, dropout_lstm_output, number_of_layers_decoder, parallel, tokenizer, device1, pad_emb, eos_emb, sos_emb, pad_token, eos_token, sos_token,beam_decode)
        
        pretrained=False
        if(pretrained):
                recommender_encoder.load_state_dict(torch.load('/media/data/gionanide/OpinionRecommendation/pytorch_models/new_models/1925_encoder_1_58_0_49.txt'))
                recommender_decoder.load_state_dict(torch.load('/media/data/gionanide/OpinionRecommendation/pytorch_models/new_models/1925_decoder_1_58_0_49.txt'))
        
        recommender_encoder.train()
        recommender_decoder.train()
        
        recommender_encoder = recommender_encoder.to(device1)
        recommender_decoder = recommender_decoder.to(device1)

        
        #------------------------------------------------------------------------------------------------------------> Loss
        #define a new Loss function based on Cross Entropy but giving much more weight on the difficult sample
        weights = initialize_loss_weights()
        class_weights = torch.FloatTensor(weights)
        #print('Class weights',class_weights)
        #criterion_review = focal_loss.FocalLoss(gamma=2, alpha=class_weights, ignore_index=0, size_average=True)#.to(device1)
        criterion_review = torch.nn.CrossEntropyLoss(ignore_index=0, weight=class_weights, reduction='mean').to(device1)
        
        '''
        #define the loss function for the review prediction
        weights = initialize_loss_weights()
        class_weights = torch.FloatTensor(weights).cuda()
        print(class_weights)
        print(class_weights.shape)
        criterion_review = torch.nn.CrossEntropyLoss(class_weights).cuda() #to put the calcualtion into GPU
        '''
        
        #for the review maybe we use Negative Log likelihood error
        #criterion_review = torch.nn.NLLLoss().to(device1)       
        
        #define the loss function for the rating prediction, we want to use RMSE so we have to use torch.sqrt(criterion(predicted, gound_truth))
        criterion_rating = torch.nn.MSELoss().to(device1)
        
        
        #MULTITASK LOSS COMBINE XXX
        is_regression = torch.Tensor([False,True])
        mlt = multitask_loss.MultiTaskLoss(is_regression,reduction='sum').to(device1)
        mlt_lr = 0.001
        mlt_optimizer = torch.optim.AdamW(mlt.parameters(), lr=mlt_lr)
        
        mlt.train()



        ''' we try to penalize complexity using weight_decay. We add the sum of all weights and add it to the loss, If this is to big might we end up with a model which has all its weights to zero '''            
        encoder_lr = 0.001#3e-4
        decay_encoder = 1e-8
        eps = 1e-8
        #encoder_optimizer = torch.optim.Adam(recommender_encoder.parameters(), lr=encoder_lr, weight_decay=decay_encoder, amsgrad=True)
        encoder_optimizer = torch.optim.AdamW(recommender_encoder.parameters(), lr=encoder_lr, weight_decay=decay_encoder,eps=eps)
        # gamma = decaying factor
        #encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=10, gamma=0.1)
        #encoder_scheduler = torch.optim.lr_scheduler.CyclicLR(encoder_optimizer, base_lr=0.00001, max_lr=0.01, step_size_up=5,step_size_down=10,cycle_momentum=False,mode='triangular2')# Scheduler
        
        decoder_lr = 0.001#3e-4
        decay_decoder = 1e-8
        eps = 1e-8
        #decoder_optimizer = torch.optim.Adam(recommender_decoder.parameters(), lr=decoder_lr, weight_decay=decay_decoder, amsgrad=True)
        decoder_optimizer = torch.optim.AdamW(recommender_decoder.parameters(), lr=decoder_lr, weight_decay=decay_decoder,eps=eps)
        #decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=10, gamma=0.1)
        #decoder_scheduler = torch.optim.lr_scheduler.CyclicLR(decoder_optimizer, base_lr=0.00001, max_lr=0.01, step_size_up=5,step_size_down=10,cycle_momentum=False,mode='triangular2')# Scheduler   
        
        
        ''' define the decay function in order to control the trade off between feeding the model with it's output or the real output | #initialize the decay '''
        teacher_forcing = True
        scheduled_sampling = lambda k,i : k/(k + math.exp(i/k))
        teacher_forcing_ratio = scheduled_sampling(5,0)        
        teacher_forcing_ratio_decay = 0
        
        
        ''' DEFINE THE DATASET '''        
        train_dataset = dataset_loader.Dataset_yelp('data/REAL_minibatch_training.pkl', transform=None) #output_space = 3519
        #train_dataset = dataset_loader.Dataset_yelp('data/1000_reviews40_max100_minibatch_training.pkl', transform=None) #output_space = 222
        #train_dataset = dataset_loader.Dataset_yelp('data/minibatch_training_5000.pkl', transform=None) #output_space = 2000
        #train_dataset = dataset_loader.Dataset_yelp('data/2088_reviews40_max100_minibatch_training.pkl', transform=None) #output_space = 704
        #train_dataset = dataset_loader.Dataset_yelp('data/minibatch_training_512.pkl', transform=None) #output_space = 351
        
        test_dataset = dataset_loader.Dataset_yelp('data/REAL_minibatch_testing.pkl', transform=None) #output_space = 3519
        
        ''' DEFINE THE DISTRIBUTED SAMPLER: use it in order to ensure that every process will process different samples '''
        #train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, size, rank)
        batch_size = 64
        train_sampler = ShuffleBatchSampler(SortedSampler(train_dataset,sort_key=lambda i: len(i[4])),batch_size=batch_size,drop_last=False,shuffle=True)
        
        ''' define the dataloader || define some proparties for the dataloader, NUMBER_OF_WORKERS = NUMBER_OF_CPU_CORES '''
        #loader_params = {'batch_size':32, 'shuffle':True, 'num_workers':10, 'pin_memory':True, 'collate_fn':custom_collate}
        loader_params = {'num_workers':10, 'pin_memory':True, 'collate_fn':custom_collate, 'batch_sampler':train_sampler}
        train_dataset_loader = torch.utils.data.DataLoader(train_dataset, **loader_params)
        overall_batch = round(train_dataset.__len__()/batch_size)
        length = train_dataset.__len__()
        print('Dataset length:',length)

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
        print('eps',eps)
        print('batch_size',batch_size)
        
        
        epochs=500
        nan = False
        
        max_rouge = 0
        max_review_length = 10000
        
        #TAKE ONLY THE FIRST BATCH
        first = next(iter(train_dataset_loader))
        
        #with GuruMeditation():
        for epoch in range(epochs):
                
                epoch_time = 0
        
                overall_review_loss = []
                overall_rating_loss = []
                
                overall_review_loss_test = []
                overall_rating_loss_test = []
                
                overall_rouge_score = []
        
                #iterate all the batches
                #for index, (user_reviews, product_reviews, neighbourhood_reviews, product_ratings, text_reviews, rating_reviews, decoder_inputs) in enumerate(train_dataset_loader):
                for index, (user_reviews, product_reviews, neighbourhood_reviews, product_ratings, text_reviews, rating_reviews, decoder_inputs) in enumerate([first]):
                
                
                        #nan = torch.isnan(decoder_inputs)
                        #print('decoder_inputs',(nan!=0).nonzero())
                        
                        #decoder_inputs[decoder_inputs!=decoder_inputs] = 0 #replace the nan values with zeros
                
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
                        
                        #change some properties of the input. Move it to GPU and make it float tensor
                        user_reviews = user_reviews.type(torch.FloatTensor)
                        product_reviews = product_reviews.type(torch.FloatTensor)
                        #format the ratings
                        product_ratings = product_ratings.type(torch.FloatTensor)
                        
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
                        overall_context_vector_encoder, overall_attention_weights, decoder_hidden = recommender_encoder(user_reviews, product_reviews)
                        
                        
                        #activations_loss, activations, rating_prediction, repetitiveness_loss_back = encoder_decoder_parallel(user_reviews, product_reviews, output_length, product_ratings, decoder_inputs, mask, teacher_forcing_ratio)
                        
                        #overall_context_vector_encoder = overall_context_vector_encoder.to(self.device1)
                        #decoder_hidden = (decoder_hidden[0].to(self.device1),decoder_hidden[1].to(self.device1))
                        #overall_attention_weights = overall_attention_weights.to(self.device1)
                        decoder_inputs = decoder_inputs.to(device1)
                        
                        #run the decoder
                        activations_loss, rating_prediction, repetitiveness_loss_back, target_length = recommender_decoder(overall_context_vector_encoder, decoder_hidden, decoder_inputs, output_length, overall_attention_weights, product_ratings, mask, teacher_forcing_ratio)
                        
                        # Each successive batch backpropagates through the timesteps of that batch AND the timesteps of all the previous batches
                        
                        
                        #make the teacher forcing ratio smaller as the samples are passing by
                        teacher_forcing_ratio_decay+=1 
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
                        rating_loss = criterion_rating(rating_reviews, rating_prediction).type(torch.FloatTensor).to(device1)
                        overall_rating_loss.append(rating_loss.item())
                        
                        
                        #----------------------------------> XXX COMMAND TO CHECK IF THE LOSS IS NAN
                        if ( (math.isnan(review_loss)) or (math.isnan(rating_loss)) ):
                                print('Deal with nan value in loss')
                                nan = True
                                sys.exit(0)
                                
                                
                        ''' CAPTURE ORTHOGONAL REGULARIZER '''
                        #orthogonality.append(orthogonal_regularization.ortho_reg(recommender_encoder,device1))
                        #orthogonality.append(orthogonal_regularization.ortho_reg(recommender_decoder,device1))
                        #ortho_loss = sum(orthogonality)/len(orthogonality)
                                
                        #sum the loss
                        #loss = mlt(torch.stack([review_loss,rating_loss]))# + ortho_loss
                        loss = target_length*review_loss + rating_loss
                        
                        #backpropagate
                        loss.backward()


                        torch.nn.utils.clip_grad_norm_(recommender_encoder.parameters(), max_norm=0.25) #gradient clipping
                        torch.nn.utils.clip_grad_norm_(recommender_decoder.parameters(), max_norm=0.25) #gradient clipping
                        
                        encoder_optimizer.step()
                        decoder_optimizer.step()
                        #mlt_optimizer.step()
                        
                        #count the time for a batch
                        epoch_time = epoch_time + (timer.time() - time_measure)
                
                        activations_loss.detach()# DETACH THE HIDDEN STATES OF THE DECODER
                        
                #do not build the computational graph
                with torch.no_grad():
        
                        recommender_encoder.eval()
                        recommender_decoder.eval()
                        
                        #role='Testing'
                        #rouge_score_test, rating_loss_test = validation(recommender_encoder,recommender_decoder,epoch,role)
                        
                        role='Training'
                        rouge_score_train, rating_loss_train = validation(recommender_encoder,recommender_decoder,epoch,role,first)
                        
                #print('Epoch:',str(epoch+1)+'/'+str(epochs),'Batch:',(str(index)+'/'+str(int(overall_batch))),'Samples',str(batch_size*(index+1))+'/'+str(length),'Output length:',output_length,'Train_review_l:',sum(overall_review_loss)/len(overall_review_loss),'Train_rate_l:',sum(overall_rating_loss)/len(overall_rating_loss),'Time:',epoch_time)
                
                print('Ep:',str(epoch+1)+'/'+str(epochs),'Out_length:',output_length,'--- LEARNING --- review:',"{0:.4f}".format(sum(overall_review_loss)/len(overall_review_loss)),'rating:',"{0:.4f}".format(sum(overall_rating_loss)/len(overall_rating_loss)),'Time:',"{0:.2f}".format(epoch_time),'--- DEV --- review:',"{0:.6f}".format(rouge_score_train),'rating:',"{0:.6f}".format(rating_loss_train)) 
                
                recommender_encoder.train()
                recommender_decoder.train()      
        
        
                #encoder_scheduler.step()
                #decoder_scheduler.step()
        
                records_file.write(str(sum(overall_review_loss)/len(overall_review_loss)))
                records_file.write('\n')
                #only from the first process
                #if (rank==0):
                #save the model after each epoch
                #encoder_decoder_parallel_name = 'encoder_decoder_parallel'
                #encoder_decoder_parallel.to(device2) #move the model to the other GPU
                
                #if(rouge_score>max_rouge):
                #        max_rouge=rouge_score
                #torch.save(recommender_encoder.state_dict(), '/media/data/gionanide/OpinionRecommendation/pytorch_models/new_models/encoder.txt')
                #torch.save(recommender_decoder.state_dict(), '/media/data/gionanide/OpinionRecommendation/pytorch_models/new_models/decoder.txt')
                #encoder_decoder_parallel.to(device1) #move the mode back to other GPU
                #print('\n\n--------------------------- Saved model to disk with name:',encoder_decoder_parallel_name,' ---------------------------\n\n')
                        
        return recommender_encoder, recommender_decoder
        
        
#test and make the sentences
def validation(recommender_encoder,recommender_decoder,epoch,role,first):

        #only in the validation
        from Keras import evaluation

        #print("\n\n--------------------------- Loaded model from disk ---------------------------\n\n")
        
        ''' define the decay function in order to control the trade off between feeding the model with it's output or the real output | #initialize the decay '''
        teacher_forcing_ratio = 1 #because we want the decoder to feed itself only with it's own outputs       
        
        if (role=='Testing'):
                #define the datalodaer
                dataloader = dataset_loader.Dataset_yelp('data/REAL_minibatch_testing.pkl', transform=None) #output_space = 8171
        elif (role=='Training'):
                #define the datalodaer
                dataloader = dataset_loader.Dataset_yelp('data/REAL_minibatch_training.pkl', transform=None) #output_space = 8171
        elif (role=='Development'):      
                #define the datalodaer
                dataloader = dataset_loader.Dataset_yelp('data/REAL_minibatch_development.pkl', transform=None) #output_space = 8171
        
        batch_size = 64
        train_sampler = ShuffleBatchSampler(SortedSampler(dataloader,sort_key=lambda i: len(i[4])),batch_size=batch_size,drop_last=False,shuffle=False)
        
        
        #define some proparties for the dataloader
        loader_params = {'num_workers':4, 'pin_memory':True, 'collate_fn':custom_collate, 'batch_sampler':train_sampler}
        dataloader = torch.utils.data.DataLoader(dataloader, **loader_params)
        
        
        #define the loss function for the review prediction
        weights = initialize_loss_weights()
        class_weights = torch.FloatTensor(weights).cuda()
        criterion_review = torch.nn.CrossEntropyLoss(class_weights).cuda() #to put the calcualtion into GPU
        
        
        #for the review maybe we use Negative Log likelihood error
        #criterion_review = torch.nn.NLLLoss().cuda()        
        #define the loss function for the rating prediction, we want to use RMSE so we have to use torch.sqrt(criterion(predicted, gound_truth))
        criterion_rating = torch.nn.MSELoss().cuda()

        overall_review_loss = []
        overall_rating_loss = []
        
        overall_rouge_score = []
        
        max_review_length = 500
        
        #iterate all the batches
        #for index, (user_reviews, product_reviews, neighbourhood_reviews, product_ratings, text_reviews, rating_reviews, decoder_inputs) in enumerate(dataloader):
        for index, (user_reviews, product_reviews, neighbourhood_reviews, product_ratings, text_reviews, rating_reviews, decoder_inputs) in enumerate([first]):
                
                #change some properties of the input. Move it to GPU and make it float tensor
                user_reviews = user_reviews.type(torch.FloatTensor)
                product_reviews = product_reviews.type(torch.FloatTensor)
                #format the ratings
                product_ratings = product_ratings.type(torch.FloatTensor)
                #take the output size, this is different for every batch
                output_length =  text_reviews.shape[1]
                
                #print('User Input reviews',user_reviews.shape)
                #print('Product input reviews',product_reviews.shape)
                #print('Product reviews ratings',product_ratings.shape)
                #print('Target review',text_reviews.shape)
                #print('Target rating',rating_reviews.shape)
                #print('Batch:',index)
                #print('---------- new ----------')
                
                #initialize mask to use it in the prediction of rating
                text_reviews_mask = text_reviews.view(-1).type(torch.LongTensor).to(device1)
                mask = initialize_mask(text_reviews_mask)
                
                user_reviews = user_reviews.to(device1)
                product_reviews = product_reviews.to(device1)
                product_ratings = product_ratings.to(device1)
                
                #run the encoder
                overall_context_vector_encoder, overall_attention_weights, decoder_hidden= recommender_encoder(user_reviews, product_reviews)
                
                
                #activations_loss, activations, rating_prediction, repetitiveness_loss_back = encoder_decoder_parallel(user_reviews, product_reviews, output_length, product_ratings, decoder_inputs, mask, teacher_forcing_ratio)
                
                #overall_context_vector_encoder = overall_context_vector_encoder.to(self.device1)
                #decoder_hidden = (decoder_hidden[0].to(self.device1),decoder_hidden[1].to(self.device1))
                #overall_attention_weights = overall_attention_weights.to(self.device1)
                decoder_inputs = decoder_inputs.to(device1)
                
                #run the decoder
                activations_loss, rating_prediction, repetitiveness_loss_back, target_length = recommender_decoder(overall_context_vector_encoder, decoder_hidden, decoder_inputs, output_length, overall_attention_weights, product_ratings, mask, teacher_forcing_ratio)
                

                
                #----------------------------------------------------------------------------------------> TEXT LOSS
                #replace Nan values with 0s
                #activations_loss[activations_loss==float('nan')] = 0
                activations_loss = activations_loss.view(-1, output_space)
                text_reviews = text_reviews.view(-1).to(device1)
                #print(activations_loss,text_reviews)
                
                review_loss = criterion_review(activations_loss, text_reviews)
                overall_review_loss.append(review_loss.item())
                
                
                #----------------------------------------------------------------------------------------> RATING LOSS
                #move target to GPU, and change it's type
                rating_reviews = rating_reviews.to(device1)
                rating_prediction = rating_prediction.view(-1)
                rating_prediction = rating_prediction.type(torch.DoubleTensor).to(device1)
                #keep record of the rating prediction loss
                rating_loss = criterion_rating(rating_reviews, rating_prediction).type(torch.FloatTensor).to(device1)
                overall_rating_loss.append(rating_loss.item())
                
                #format the ratings
                #rating_prediction = rating_prediction.squeeze(1)
                
                '''
                #iterate every sample in the batch
                for sample in range(activations_loss.shape[0]):
                        predicted_sentence = []
                        ground_truth = []
                        target_reviews_length = activations_loss[sample].shape[0]
                        for word in range(activations_loss[sample].shape[0]):  
                                #find the position of the max element, which represents the word ID
                                predicted_index = torch.argmax(activations_loss[sample][word]).item()
                                ground_truth.append(text_reviews[sample][word].item())
                                predicted_sentence.append(predicted_index)
                                
                                
                        ground_truth_rating = rating_reviews[sample].item()
                        rating_prediction_item = rating_prediction[sample].item()
                        count_predictions=1
                        #if(len(ground_truth)>max_review_length):
                        #        ground_truth=ground_truth[:max_review_length]
                        #        ground_truth.append(eos_token)
                        
                        #for every predicted sentence
                        print_sentence = False
                        

                        rouge_score = evaluation.make_sentence_one_by_one(predicted_sentence, ground_truth, target_reviews_length, tokenizer, role, count_predictions, rating_prediction_item, ground_truth_rating,print_sentence,max_review_length)
                        overall_rouge_score.append(rouge_score)
                        '''

                return sum(overall_review_loss)/len(overall_review_loss), sum(overall_rating_loss)/len(overall_rating_loss)
                        
                        #break
        
        
        
        
        
        
        
        
