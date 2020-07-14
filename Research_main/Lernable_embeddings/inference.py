#!usr/bin/python
import numpy as np
import random
import pickle
import torch
import math
import os
from classes import dataset_loader
from classes import word2vec
from rouge import Rouge
from pytorch_models import encoder
from pytorch_models import decoder
from pytorch_models import n_gram_overlap
from torchnlp.samplers.shuffle_batch_sampler import ShuffleBatchSampler
from torchnlp.samplers import SortedSampler
from progress.bar import Bar
import argparse




ap = argparse.ArgumentParser()
ap.add_argument("--epoch", "--epoch", required=True, help="define from which epoch you want to load a model")
ap.add_argument("--batch_size", "--batch_size", required=True, help="define the number of samples to feed the model in every step")
ap.add_argument("--mode", "--mode", required=True, help="choose between train/dev/test set")
args = vars(ap.parse_args())


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
        
        
# A seperate function to initialize mask
def initialize_mask(text_reviews):
        #take the elements that are non zero, 1 to the actuall tokens, 0 to the padded one
        mask = (text_reviews>0).float()
        return mask
        
        
#convert prediction into sentence
def make_sentence_one_by_one(prediction, ground_truth, target_reviews_length, language_output, role, count_predictions, rating_prediction, ground_truth_rating, print_sentence,max_review_length):

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
                word = language_output.index2word[index_word]
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
                word = language_output.index2word[word]
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
        


if __name__ == '__main__':
        #---------------------------------------------------------------------------------------------------------------------------------> EVALUATION

        #because we want to run on GPUs
        device1 = torch.device('cuda:0')
        
        epoch=int(args["epoch"])
        batch_size = int(args["batch_size"]) #run got one sample each tim
        role = args["mode"]
        
        reproducability(12)

        #XXX read the train dataset
        language_input = pickle.load(open("./data/language_input_"+role+".pkl",'rb'))
        language_output = pickle.load(open("./data/language_output_"+role+".pkl",'rb'))
        embedding_d = 128

        #define the number of maximum reviews for every sample, if None we are taking all the existing reviews
        normalize_user_reviews = 40
        normalize_product_reviews = 40
        normalize_neighbourhood_reviews = 5

        #define the number of hidden units to use in the LSTM
        hidden_units_encoder = 128
        hidden_units_decoder = 128

        #define training properties
        encoder_input_dropout = 0.2
        dropout_lstm_output = 0.2
        dropout_after_linear = 0.3
        episodes = 3
        output_space = len(language_output.word2index)+1
        number_of_layers_encoder = 2
        number_of_layers_decoder = 2
        parallel = False

        sos_token = language_input.word2index["sos"]
        eos_token = language_input.word2index["eos"]
        pad_token = 0
        
        #initialize the network, encoder
        recommender_encoder = encoder.Overall_Encoder(hidden_units_encoder, encoder_input_dropout, episodes, normalize_product_reviews, number_of_layers_encoder, parallel,len(language_input.word2index),embedding_d)
                
        beam_decode=False
        #Dynamic Memory Network
        recommender_decoder = decoder.DMN(hidden_units_decoder, output_space, dropout_after_linear, dropout_lstm_output, number_of_layers_decoder, parallel, device1, pad_token, eos_token, sos_token, beam_decode, language_output, embedding_d)
        
        recommender_encoder.eval()
        recommender_decoder.eval()
        
        recommender_encoder = recommender_encoder.to(device1)
        recommender_decoder = recommender_decoder.to(device1)

        role='Training' #BY DEAFULT IN TESTING PHASE

        #print("\n\n--------------------------- Loaded model from disk ---------------------------\n\n")
        
        ''' define the decay function in order to control the trade off between feeding the model with it's output or the real output | #initialize the decay '''
        teacher_forcing_ratio = 0 #because we want the decoder to feed itself only with it's own outputs       
        
        
        
        #choose in which set you want to run inference (DEFAULT: Testing ---> if you want to check something else turn to Development or Training.)
        if (role=='Testing'):
                #define the datalodaer
                train_dataset = pickle.load(open("./data/raw_train_set_PILOT.pkl",'rb'))
        elif (role=='Training'):
                #define the datalodaer
                train_dataset = pickle.load(open("./data/raw_train_set_PILOT.pkl",'rb'))
        elif (role=='Development'):      
                #define the datalodaer
                train_dataset = pickle.load(open("./data/raw_train_set_PILOT.pkl",'rb'))
        
        #define the loss function for the review prediction
        weights = initialize_loss_weights(language_output)
        class_weights = torch.FloatTensor(weights).cuda()
        criterion_review = torch.nn.CrossEntropyLoss(ignore_index=0, weight=class_weights, reduction='mean').cuda() #to put the calcualtion into GPU     
        #define the loss function for the rating prediction, we want to use RMSE so we have to use torch.sqrt(criterion(predicted, gound_truth))
        criterion_rating = torch.nn.MSELoss(reduction='mean').cuda()

        overall_review_loss = []
        overall_rating_loss = []
        overall_rouge_score = []
        max_review_length = 300 # maybe we want to cut very long reviews (not during testing)
        n_gram_overlap_predictions = []
        n_gram_overal_groundtruth = []
        
        path_to_models = "./pytorch_models/new_models/training/"
        models = os.listdir(path_to_models)
        
        
        recommender_encoder.load_state_dict(torch.load('./pytorch_models/new_models/training/encoder_LAST'+str(epoch)+'.txt'))
        recommender_decoder.load_state_dict(torch.load('./pytorch_models/new_models/training/decoder_LAST'+str(epoch)+'.txt'))
        
        bar = Bar("Training",max=round(len(train_dataset.text_reviews)/batch_size+1))
              
        #iterate all the batches
        for index in range(0,len(train_dataset.text_reviews),batch_size):
        #for index, (user_reviews, product_reviews, neighbourhood_reviews, product_ratings, text_reviews, rating_reviews, decoder_inputs) in enumerate([first]):
                
                #XXX make the batch
                user_reviews = train_dataset.user_reviews[index:index+batch_size]
                product_reviews = train_dataset.product_reviews[index:index+batch_size]
                neighbourhood_reviews = train_dataset.neighbourhood_reviews[index:index+batch_size]
                product_ratings = train_dataset.product_ratings[index:index+batch_size]
                text_reviews = train_dataset.text_reviews[index:index+batch_size]
                decoder_inputs = format_decoder_input(text_reviews,language_input)
                rating_reviews = train_dataset.rating_reviews[index:index+batch_size]

                #XXX make batch tensors
                user_reviews = torch.nn.utils.rnn.pad_sequence(user_reviews, batch_first=True, padding_value=0)
                product_reviews = torch.nn.utils.rnn.pad_sequence(product_reviews, batch_first=True, padding_value=0)
                text_reviews = torch.nn.utils.rnn.pad_sequence(text_reviews, batch_first=True, padding_value=0)
                decoder_inputs = torch.nn.utils.rnn.pad_sequence(decoder_inputs, batch_first=True, padding_value=0)
                product_ratings = torch.stack(product_ratings).float()
                rating_reviews = torch.stack(rating_reviews).double()
                
                #take the output size, this is different for every batch
                output_length =  text_reviews.shape[1]
                
                #initialize mask to use it in the prediction of rating
                text_reviews_mask = text_reviews.view(-1).to(device1)
                mask = initialize_mask(text_reviews_mask)
                
                
                user_reviews = user_reviews.to(device1)
                product_reviews = product_reviews.to(device1)
                product_ratings = product_ratings.to(device1)
                
                #run the encoder
                overall_context_vector_encoder, overall_attention_weights, decoder_hidden = recommender_encoder(user_reviews, product_reviews)
                #overall_attention_weights = overall_attention_weights.to(self.device1)
                decoder_inputs = decoder_inputs.to(device1)
                #run the decoder
                activations_loss, rating_prediction, repetitiveness_loss_back, target_length = recommender_decoder(overall_context_vector_encoder, decoder_hidden, decoder_inputs, output_length, overall_attention_weights, product_ratings, mask, teacher_forcing_ratio)

                #----------------------------------------------------------------------------------------> TEXT LOSSs
                #print('overall',(nan!=0).nonzero())
                #print(activations_loss[activations_loss==float('inf')])
                activations = activations_loss.view(-1, output_space)
                text_reviews_loss = text_reviews.view(-1).to(device1)
                review_loss = criterion_review(activations, text_reviews_loss)
                overall_review_loss.append(review_loss.item())
                #----------------------------------------------------------------------------------------> RATING LOSS
                #move target to GPU, and change it's type
                rating_reviews = rating_reviews.to(device1)
                rating_prediction = rating_prediction.view(-1)
                rating_prediction = rating_prediction.type(torch.DoubleTensor).to(device1)
                #keep record of the rating prediction loss
                rating_loss = criterion_rating(rating_reviews.squeeze(1), rating_prediction).type(torch.FloatTensor).to(device1)
                overall_rating_loss.append(rating_loss.item())                
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

                        predicted_str, target_str, rouge_score = make_sentence_one_by_one(predicted_sentence, ground_truth, target_reviews_length, language_output, role, count_predictions, rating_prediction_item, ground_truth_rating,print_sentence,max_review_length)
                        n_gram_overlap_predictions.append(predicted_str)# keep the predictions sentences to calculate the n-gram overlap
                        n_gram_overal_groundtruth.append(target_str)# keep the gound truth sentences to calculate the n-gram overlap
                        overall_rouge_score.append(rouge_score)
                #print('-------------------- '+str(index)+' -------------------- \n\n')
                bar.next()
        bar.finish()
        print('\n Validation \n')

        print("Model ----> ",epoch,'overall_rouge: ',sum(overall_rouge_score)/len(overall_rouge_score),'overall_rating_loss: ',sum(overall_rating_loss)/len(overall_rating_loss))
        
        
        
        
        
        
