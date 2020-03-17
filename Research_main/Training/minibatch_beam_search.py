#!usr/bin/python
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
from Preprocessing import preprocessing
from Preprocessing import word2vec
import dataset_loader
import math
import os
import random
import pickle
import numpy as np
from pytorch_models import DMN_beam_search
from pytorch_models import batch_overall_encoder

device1 = torch.device('cuda:0')
#load the tokenizer
tokenizer_path = '/media/data/gionanide/OpinionRecommendation/data/REAL_tokenizer.pkl'
#tokenizer_path = '/media/data/gionanide/OpinionRecommendation/data/100_reviews40_max100_minibatch_tokenizer.pkl'
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
SOS_token = sos_token
EOS_token = eos_token
max_queue = 2000
print('<SOS>:',sos_token,'<EOS>:',eos_token,'<PAD>:',pad_token)
#reroducability
np.random.seed(1)
torch.manual_seed(0)
#ENBALE CuDNN benchmark
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
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
number_of_layers = 2
parallel = False                
         
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
        weights[pad_token] = 0.0 # PAD
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
         
#---> XXX XXX XXX XXX XXX XXX XXX XXX XXX <---

#only in the validation
from Keras import evaluation
#initialize the network, encoder
recommender_encoder = batch_overall_encoder.Overall_Encoder(hidden_units_encoder, encoder_input_dropout, episodes, normalize_product_reviews, number_of_layers, parallel)
       
#Dynamic Memory Network
recommender_decoder = DMN_beam_search.DMN_beam_search(hidden_units_decoder, output_space, dropout_after_linear, dropout_lstm_output, number_of_layers, parallel, tokenizer, device1, pad_emb, eos_emb, sos_emb, pad_token, eos_token, sos_token, max_queue)      
#load weights into new model
#initialize the network, encoder        
recommender_encoder.load_state_dict(torch.load('/media/data/gionanide/OpinionRecommendation/pytorch_models/new_models/encoder_train_overfit.txt'))
recommender_decoder.load_state_dict(torch.load('/media/data/gionanide/OpinionRecommendation/pytorch_models/new_models/decoder_train_overfit.txt'))
recommender_encoder.eval()
recommender_decoder.eval()
recommender_encoder.to(device1)
recommender_decoder.to(device1)
print("\n\n--------------------------- Loaded model from disk ---------------------------\n\n")
''' define the decay function in order to control the trade off between feeding the model with it's output or the real output | #initialize the decay '''
teacher_forcing_ratio = 1 #because we want the decoder to feed itself only with it's own outputs       
#role='Training'
role='Testing'
if (role=='Testing'):
        #define the datalodaer
        dataloader = dataset_loader.Dataset_yelp('data/REAL_minibatch_testing.pkl', transform=None) #output_space = 8171
        #dataloader = dataset_loader.Dataset_yelp('data/100_reviews40_max100_minibatch_testing.pkl', transform=None) #output_space = 17
        #dataloader = dataset_loader.Dataset_yelp('data/2088_reviews40_minibatch_testing.pkl', transform=None) #output_space = 2000
        #dataloader = dataset_loader.Dataset_yelp('data/2088_reviews40_minibatch_testing.pkl', transform=None) #output_space = 704
        #train_dataset = dataset_loader.Dataset_yelp('data/minibatch_testing_512.pkl', transform=None) #output_space = 351
elif (role=='Training'):
        #define the datalodaer
        dataloader = dataset_loader.Dataset_yelp('data/REAL_minibatch_training.pkl', transform=None) #output_space = 8171
        #dataloader = dataset_loader.Dataset_yelp('data/100_reviews40_max100_minibatch_training.pkl', transform=None) #output_space = 17
        #dataloader = dataset_loader.Dataset_yelp('data/2088_reviews40_minibatch_training.pkl', transform=None) #output_space = 2000
        #dataloader = dataset_loader.Dataset_yelp('data/2088_reviews40_minibatch_training.pkl', transform=None) #output_space = 704
        #train_dataset = dataset_loader.Dataset_yelp('data/minibatch_training_512.pkl', transform=None) #output_space = 351       
#define some proparties for the dataloader
loader_params = {'batch_size':1, 'shuffle':False, 'num_workers':4, 'pin_memory':True, 'collate_fn':custom_collate}
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
max_review =10000
print_sentence = True
beam_decode = True
#do not build the computational graph
with torch.set_grad_enabled(False):
        #iterate all the batches
        for index, (user_reviews, product_reviews, neighbourhood_reviews, product_ratings, text_reviews, rating_reviews, decoder_inputs) in enumerate(dataloader):
                #change some properties of the input. Move it to GPU and make it float tensor
                user_reviews = user_reviews.type(torch.FloatTensor)
                product_reviews = product_reviews.type(torch.FloatTensor)
                #format the ratings
                product_ratings = product_ratings.type(torch.FloatTensor)
                #take the output size, this is different for every batch
                output_length =  text_reviews.shape[1]
                #print('User Input reviews',user_reviews.shape)
                #print('Product input reviews',product_reviews.shape)
                #print('Neighbourhood input reviews',neighbourhood_reviews.shape)
                #print('Product reviews ratings',product_ratings.shape)
                #print('Target review',text_reviews.shape)
                #print('Target rating',rating_reviews.shape)
                #print('Batch:',index)
                #print('---------- new ----------')
                #initialize mask to use it in the prediction of rating
                text_reviews_loss = text_reviews.clone()
                text_reviews_loss = text_reviews_loss.view(-1).to(device1)
                mask = initialize_mask(text_reviews_loss)
                user_reviews = user_reviews.to(device1)
                product_reviews = product_reviews.to(device1)
                product_ratings = product_ratings.to(device1)
                #run the encoder
                overall_context_vector_encoder, overall_attention_weights, decoder_hidden = recommender_encoder(user_reviews, product_reviews)  
                overall_context_vector_encoder = overall_context_vector_encoder.to(device1)
                decoder_hidden = (decoder_hidden[0].to(device1),decoder_hidden[1].to(device1))
                overall_attention_weights = overall_attention_weights.to(device1)
                product_ratings = product_ratings.to(device1)
                decoder_inputs = decoder_inputs.to(device1)
                #run the decoder
                activations_loss, rating_prediction, repetitiveness_loss_back, decoded_batch = recommender_decoder(overall_context_vector_encoder, decoder_hidden, decoder_inputs, output_length, overall_attention_weights, product_ratings, mask, beam_decode)
                #move target to GPU, and change it's type
                #rating_reviews = rating_reviews.to(device1)
                rating_prediction_loss = rating_prediction.clone()
                rating_prediction_loss = rating_prediction_loss.view(-1)
                rating_prediction_loss = rating_prediction_loss.type(torch.DoubleTensor)
                #keep record of the rating prediction loss
                rating_loss = criterion_rating(rating_reviews, rating_prediction_loss).type(torch.FloatTensor)
                overall_rating_loss.append(rating_loss.item())
                #format the ratings
                rating_prediction_item = rating_prediction.squeeze(1).squeeze(1).item()
                ground_truth = text_reviews.tolist()[0]
                target_reviews_length = len(ground_truth)
                count_predictions=1
                ground_truth_rating = rating_reviews.item()
                max_rouge = 0 # for every beam search we are keeping the sentence with the higher ROUGE score
                #iterate every sample in the batch
                for decoded_sentence in decoded_batch:
                        predicted_sentence = np.trim_zeros(decoded_sentence)                            
                        #for every predicted sentence
                        rouge_score = evaluation.make_sentence_one_by_one(predicted_sentence, ground_truth, target_reviews_length, tokenizer, role, count_predictions, rating_prediction_item, ground_truth_rating,print_sentence,max_review)
                        if (rouge_score>max_rouge):
                                max_rouge=rouge_score
                overall_rouge_score.append(max_rouge)
                print('-------------------- '+str(index)+' --------------------')
        print(role+' rating loss:',sum(overall_rating_loss)/len(overall_rating_loss),role+' Rouge score:',sum(overall_rouge_score)/len(overall_rouge_score))                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
