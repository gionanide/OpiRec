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
