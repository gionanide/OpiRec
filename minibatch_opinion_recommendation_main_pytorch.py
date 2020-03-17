#!usr/bin/python
import minibatch_training
from pytorch_models import n_gram_overlap
import numpy as np
import random
import pickle
import torch
from pytorch_models import batch_overall_encoder
from pytorch_models import batch_decoder_DMN

#torch.backends.cudnn.deterministic = True
#device1 = torch.device('cuda:0')


if __name__ == '__main__':        

        #'''
        
        #size = 2 #DEFINE THE NUMBER OF PARALLLE PROCESSES TO RUN
        
        #encoder_decoder_parallel = torch.multiprocessing.spawn(minibatch_training.minibatch_training, args=(size,),nprocs=size,join=True) #INITIALIZE PARALLEL TRAININGS
        
        #minibatch_training.minibatch_training()
        
        
        
        #----------------------------------------------> EVALUATION
        #because we want to run on GPUs
        device1 = torch.device('cuda:0')

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
        episodes = 1
        output_space = 1903
        number_of_layers_encoder = 2
        number_of_layers_decoder = 2
        parallel = False

        #load the tokenizer
        tokenizer_path = '/media/data/gionanide/OpinionRecommendation/data/REAL_tokenizer.pkl'
        #tokenizer_path = '/media/data/gionanide/OpinionRecommendation/data/2088_reviews40_max100_minibatch_tokenizer.pkl'
        #tokenizer_path = '/media/data/gionanide/OpinionRecommendation/data/2088_reviews40_tokenizer.pkl'

        tokenizer = pickle.load(open(tokenizer_path,'rb'))

        #make the appropriate format in order to feed the previous word to the decoder -- the first word to feed decoder is SOS
        #but their word embeddings can be arbitrary, small random numbers will do fine, because this vector would be equally far from all "normal" vectors.
        pad_emb = torch.load('/media/data/gionanide/OpinionRecommendation/data/special_chars/pad.pt')
        eos_emb = torch.load('/media/data/gionanide/OpinionRecommendation/data/special_chars/eos.pt')
        sos_emb = torch.load('/media/data/gionanide/OpinionRecommendation/data/special_chars/sos.pt')

        #DEFINE EOS AND SOS TOKENs
        sos_token = tokenizer.word_index['sos']
        eos_token = tokenizer.word_index['eos']
        pad_token = 0
        
        deep_memory_calculations = 0

        #initialize the network, encoder
        recommender_encoder = batch_overall_encoder.Overall_Encoder(hidden_units_encoder, encoder_input_dropout, episodes, normalize_product_reviews, number_of_layers_encoder, parallel)
                
        beam_decode=False
        #Dynamic Memory Network
        recommender_decoder = batch_decoder_DMN.DMN(hidden_units_decoder, output_space, dropout_after_linear, dropout_lstm_output, number_of_layers_decoder, parallel, tokenizer, device1, pad_emb, eos_emb, sos_emb, pad_token, eos_token, sos_token,beam_decode)
        
	#convert the models to evaluation mode (especially for dropout layers)
        recommender_encoder.eval()
        recommender_decoder.eval()
        
        recommender_encoder = recommender_encoder.to(device1)
        recommender_decoder = recommender_decoder.to(device1)

        recommender_encoder.load_state_dict(torch.load('/media/data/gionanide/OpinionRecommendation_working_demo/pytorch_models/new_models/trained/encoder_epoch_38.txt'))
        recommender_decoder.load_state_dict(torch.load('/media/data/gionanide/OpinionRecommendation_working_demo/pytorch_models/new_models/trained/decoder_epoch_38.txt'))

        role='Testing'
        epoch=1

        overall_rouge_loss, rating_mean, rating_std, overall_review_loss, n_gram_overlap_predictions, n_gram_overal_groundtruth, deep_memory_calculations,rating_std  = minibatch_training.validation(recommender_encoder,recommender_decoder,epoch,role,deep_memory_calculations)
        
        duplicate_score_1_predictions = n_gram_overlap.count_grams_duplicate(n_gram_overlap_predictions, n=1)
        duplicate_score_2_predictions = n_gram_overlap.count_grams_duplicate(n_gram_overlap_predictions, n=2)
        duplicate_score_3_predictions = n_gram_overlap.count_grams_duplicate(n_gram_overlap_predictions, n=3)
        duplicate_score_4_predictions = n_gram_overlap.count_grams_duplicate(n_gram_overlap_predictions, n=4)
        duplicate_score_5_predictions = n_gram_overlap.count_grams_duplicate(n_gram_overlap_predictions, n=5)
        duplicate_score_1_groundtruth = n_gram_overlap.count_grams_duplicate(n_gram_overal_groundtruth, n=1)
        duplicate_score_2_groundtruth = n_gram_overlap.count_grams_duplicate(n_gram_overal_groundtruth, n=2)
        duplicate_score_3_groundtruth = n_gram_overlap.count_grams_duplicate(n_gram_overal_groundtruth, n=3)
        duplicate_score_4_groundtruth = n_gram_overlap.count_grams_duplicate(n_gram_overal_groundtruth, n=4)
        duplicate_score_5_groundtruth = n_gram_overlap.count_grams_duplicate(n_gram_overal_groundtruth, n=5)

        print('\n N-gram overalap rate % \n')

        print('unigrams overlap -----> predictions: ',"{0:.6f}".format(duplicate_score_1_predictions),'%','  ground truth: ',"{0:.6f}".format(duplicate_score_1_groundtruth),'%')
        print('bigrams overlap -----> predictions: ',"{0:.6f}".format(duplicate_score_2_predictions),'%','  ground truth: ',"{0:.6f}".format(duplicate_score_2_groundtruth),'%')
        print('trigrams overlap -----> predictions: ',"{0:.6f}".format(duplicate_score_3_predictions),'%','  ground truth: ',"{0:.6f}".format(duplicate_score_3_groundtruth),'%')
        print('4-grams overlap -----> predictions: ',"{0:.6f}".format(duplicate_score_4_predictions),'%','  ground truth: ',"{0:.6f}".format(duplicate_score_4_groundtruth),'%')
        print('sentences overlap -----> predictions: ',"{0:.6f}".format(duplicate_score_5_predictions),'%','  ground truth: ',"{0:.6f}".format(duplicate_score_5_groundtruth),'%')
        
        print('\n Validation \n')

        print('overall_rouge_loss',overall_rouge_loss,'overall_review_loss',overall_review_loss,'overall_rating_loss: mean ',rating_mean,'deep_memory_calculations',deep_memory_calculations,'std',rating_std)
        
        
        
        
        
        
        
