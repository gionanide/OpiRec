#!usr/bin/python
import torch
from pytorch_models import recommender_v2_encoder
from pytorch_models import decoder_DMN
import preprocessing
import word2vec
import evaluation
import random

'''

Here we are build the Encoder-Decoder model all in one

'''
class Seq2Seq(torch.nn.Module):

        def __init__(self, hidden_units_encoder, input_dropout, hidden_units_decoder, output_space, dropout_after_linear, dropout_lstm_output, episodes, normalize_product_reviews, review_weight, criterion_review, coverage_weight, teacher_forcing_ratio, teacher_forcing, device1, tokenizer, eos):
        
                super(Seq2Seq, self).__init__()
        
                #define the Encoder
                self.encoder = recommender_v2_encoder.Recommender_v2_encoder(hidden_units_encoder, input_dropout)
                
                #define the Attention Decoder with Dynamic Memory
                self.decoder = decoder_DMN.DMN(hidden_units_decoder, output_space, dropout_after_linear, dropout_lstm_output, episodes, normalize_product_reviews)
        
                #assign some properties
                self.review_weight = review_weight
                self.criterion_review = criterion_review
                self.coverage_weight = coverage_weight
                self.teacher_forcing_ratio = teacher_forcing_ratio
                self.teacher_forcing = teacher_forcing
                self.device = device1
                self.tokenizer = tokenizer
                self.eos = eos
                
                
        def forward(self, user_inputs, product_inputs, neighbourhood_inputs, decoder_inputs, training_ground_truth):
        
                #encode the input reviews
                user_lstm_output, user_h, product_lstm_output, product_h = self.encoder(user_inputs, product_inputs, neighbourhood_inputs)                
                #assign the first hidden state of the decoder, and make them 3D tensors
                decoder_hidden = (product_h[0].unsqueeze(0), product_h[1].unsqueeze(0))
                #initialize it as False every time we are start predicting
                decoder_predict_eos = False
                
                #initialize the loss for this sample
                gradient_loss = 0
                
                #initialize the coverage vector
                coverage_vector = torch.zeros([1, 1, product_inputs.shape[0]], dtype=torch.float).to(self.device)

                #iterate all the samples from the ground truth
                for target_index, target_word_id in enumerate(training_ground_truth[0]):

                        # ------------------------------------------------------------------- DECODER -----------------------------------------------------------
                        #activations, decoder_hidden = recommender_decoder(user_lstm_output, user_h, neighbourhood_lstm_output, neighbourhood_h, product_lstm_output, product_h, decoder_hidden, decoder_inputs)
                        activations, decoder_hidden, coverage_vector, overall_attention_weights = self.decoder(user_lstm_output, user_h, product_lstm_output, product_h, decoder_hidden, decoder_inputs, coverage_vector)
                        
                        
                        if (self.teacher_forcing):
                        
                                #generate a random number between 0-1, the bigger the ratio the more probable the number to be smaller that it
                                random_number = random.uniform(0, 1)
                                #print('\n Random number:',random_number,'\n')
                                
                                if ( random_number < self.teacher_forcing_ratio ):
                                
                                        #print('True target')
                                        #counter_true+=1

                                        #because the teacher forcing ratio in close to one, it is more probable that the number wil be lower that the ratio
                                        decoder_inputs = target_word_id
                                        #make the appropriate format in order to feed the previous word to the decoder
                                        decoder_inputs = preprocessing.index_to_word_mapping(decoder_inputs, self.tokenizer)
                                        #print(decoder_inputs)
                                        
                                        #print('\n Decoder input:',decoder_inputs,'\n')                                        
                                        decoder_inputs = word2vec.word2vec(decoder_inputs, target_word_id)
                                        decoder_inputs = torch.tensor(decoder_inputs, dtype=torch.torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                                                                                
                                else:
                                
                                        #print('Predicted target')
                                        #counter_predicted+=1
                                
                                        #otherwise feed the model with it's previous output, this becomes more probable as tha training goes on
                                        decoder_inputs_activation = torch.argmax(activations).item() + 1
                                        #print('Decodeeeer inputs',decoder_inputs)
                                        
                                        #if the decoder feed itself with eos token end the training procedure for this sample. We consider it as an end of predictions
                                        if (decoder_inputs_activation==self.eos):
                                        
                                                decoder_predict_eos = True                                        
                                        
                                        #make the appropriate format in order to feed the previous word to the decoder
                                        decoder_inputs = preprocessing.index_to_word_mapping(decoder_inputs_activation, self.tokenizer)
                                        
                                        #maybe in the early stages of training procedure the model outputs None values, it means no word at all, so we have to feed the model with the actual groudn truth, besides teacher forcing rate we are using this as well
                                        if (decoder_inputs==None):
                                        
                                                #make the appropriate format in order to feed the previous word to the decoder
                                                decoder_inputs = preprocessing.index_to_word_mapping(target_word_id, self.tokenizer)
                                        
                                        
                                        
                                        #print(decoder_inputs)
                                        decoder_inputs = word2vec.word2vec(decoder_inputs, decoder_inputs_activation)
                                        decoder_inputs = torch.tensor(decoder_inputs, dtype=torch.torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                                        



                        #iterating the ground thruth
                        target_word = torch.tensor(target_word_id, dtype=torch.long, device=self.device).unsqueeze(0)
                        #print('Target:',target_word)
                        
                        activations = activations.squeeze(0)
                        #print(activations)
                        #print('Prediction:',activations.shape)
                
                        #calculate the loss
                        gradient_loss += self.review_weight*self.criterion_review(activations, target_word) + self.coverage_weight*evaluation.coverage_loss_func(overall_attention_weights,coverage_vector)
                        
                        #if decoder's own prediction if the EOS token, we assume that this is the end of the sentence, we caluclate the erros and we stop predicting
                        if (decoder_predict_eos):
                        
                                #define the predition to False again
                                decoder_predict_eos = False
                        
                                break
                          
                #return the loss of just one sample
                return gradient_loss
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                
