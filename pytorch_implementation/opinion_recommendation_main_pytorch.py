#!usr/bin/python
import preprocessing
from matplotlib import pyplot as plt
import numpy as np
from models import recommender_v2
from models import recommender_v1
from models import recommender_v3
from models import recommender_v4
import evaluation
import pytorch_model_training
import torch
from pytorch_models import recommender_v1_encoder
from pytorch_models import recommender_v1_decoder


#reproducability
np.random.seed(7)




if __name__ == '__main__':

        #file to read the sampes
        #path = '/media/data/gionanide/OpinionRecommendation/Feeding_samples_90-10_big' # -----------------> 53565 samples
        #path = '/media/data/gionanide/OpinionRecommendation/Feeding_samples_90-10_medium/' # -------------> 2089 samples
        path = '/media/data/gionanide/OpinionRecommendation/Feeding_smaples_90-10_small/' # ---------------> 512 samples
        #path = '/media/data/gionanide/OpinionRecommendation/Feeding_smaples_90-10_oneshot/' # ------------> 1 sample
        
        #return a list with all the samples, as paths
        samples = preprocessing.handle_input(path)
        
        #-----------------------------------------------------------------> Preprocessing properties
        
        #make this as True if you want to encode your target to one-hot encoding istead of a sequence of words IDs
        one_hot = False
        
        #define if you want padding or not
        padding = False
        
        #define the number of maximum reviews for every sample
        normalize_reviews = 5
        
        #define the number of hidden units to use in the LSTM
        hidden_units = 300
        
        #define if you want to crop your vocabulary
        cut_bad_words = True
                
        #make the appropriatte format for the target
        targets, output_vocabulary_size, max_review_length, empty_text_reviews, eos, sos, tokenizer, target_reviews_length_train, target_reviews_length_test = preprocessing.format_target(path, samples, padding, cut_bad_words)
        
        
        
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------> Trai and save the model
        
        epochs = 100
        
        teacher_forcing = True
        
        
        encoder, decoder = pytorch_model_training.training(path, samples, output_vocabulary_size, targets, empty_text_reviews, normalize_reviews, one_hot, padding, hidden_units, epochs, tokenizer, teacher_forcing)
        
        
        #save the model
        torch.save(encoder, "/media/data/gionanide/OpinionRecommendation/pytorch_models/encoder.txt")
        torch.save(decoder, "/media/data/gionanide/OpinionRecommendation/pytorch_models/decoder.txt")
        print("\n\n--------------------------- Saved model to disk ---------------------------n\n")
        
        
        
        
        
        
        #--------------------------------------------------------------------------------------------------------------------------------------> Load and test the model
        
        '''
        #initialize the models first
        #initialize the network, encoder
        recommender_encoder = recommender_v1_encoder.Recommender_v1_encoder(hidden_units)
        #load weights into new model
        recommender_encoder = torch.load("/media/data/gionanide/OpinionRecommendation/pytorch_models/encoder.txt")
        recommender_encoder.eval()
        
        #initialize the network decoder
        recommender_decoder = recommender_v1_decoder.Recommender_v1_decoder(hidden_units, output_vocabulary_size)
        #load weights into new model
        recommender_decoder = torch.load("/media/data/gionanide/OpinionRecommendation/pytorch_models/decoder.txt")
        recommender_decoder.eval()
        print("\n\n--------------------------- Loaded model from disk ---------------------------\n\n")
        


        #----------------------------------> define some properties
        role = 'Training'
        
        mode = 'greedy_search'
        #mode = 'beam_search'
        beam_deapth = 3
        
        implementation = 'pytorch'
        
        pytorch_model_training.make_prediction(path, recommender_encoder, recommender_decoder, samples, output_vocabulary_size, targets, empty_text_reviews, normalize_reviews, one_hot, padding, tokenizer, target_reviews_length_train, role)
        '''
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
