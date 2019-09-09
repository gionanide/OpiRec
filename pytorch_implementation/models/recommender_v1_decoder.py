#!usr/bin/python
from pytorch_models import bahdanau_attention_pytorch as bahdanau_attention
import torch


class Recommender_v1_decoder(torch.nn.Module):


        def __init__(self, hidden_units, output_vocabulary_size):
        
                #create a child class of Recommender_v1
                super(Recommender_v1_decoder, self).__init__()
                
                #initializa some properties
                self.hidden_units = hidden_units
                self.output_space = output_vocabulary_size
                
                #define custom attention layers
                self.user_profile_attention = bahdanau_attention.BahdanauAttention(hidden_units, multiple=False)
                self.neighbourhood_profile_attention = bahdanau_attention.BahdanauAttention(hidden_units, multiple=False)
                self.overall_attention = bahdanau_attention.BahdanauAttention(hidden_units, multiple=True)
                
                #an lstm before the output to take the previous output as well
                self.lstm_decode = torch.nn.LSTM(hidden_units, hidden_units)
                
                #build the classification procedure
                self.output = torch.nn.Linear(hidden_units, output_vocabulary_size)
                
                #and the activation function after the Dense layer
                #self.softmax = torch.softmax(dim=2)
                
                
                
        def forward(self, user_lstm_output, user_h, neighbourhood_lstm_output, neighbourhood_h, product_lstm_output, product_h, decoder_hidden):
               
                
                #------------------------------------------------------------------------------------------------------------------> ATTENTION
                #print('------ Attention ------')
                #make attention for user and neighbourhood profile first
                
                user_context_vector, user_attention_weights = self.user_profile_attention(user_lstm_output, user_lstm_output, user_lstm_output)
                #print('User attention:',user_context_vector.shape, user_context_vector.type())
                #print('User attention weights:',user_attention_weights.shape, user_attention_weights.type())
                
                
                
                neighbourhood_context_vector, neighbourhood_attention_weights = self.neighbourhood_profile_attention(neighbourhood_lstm_output, neighbourhood_lstm_output, neighbourhood_lstm_output)
                #print('Neighbourhood attention:',neighbourhood_context_vector.shape, neighbourhood_context_vector.type())
                #print('Neighbourhood attention weights:',neighbourhood_attention_weights.shape, neighbourhood_attention_weights.type())
                
                #after this we are ready to take the outputs from the profiles that we conclude, user and neighbourhood and find the overall attention
                overall_context_vector, overall_attention_weights = self.overall_attention(product_lstm_output, user_context_vector, neighbourhood_context_vector)
                #print('Overall attention:',overall_context_vector.shape,overall_context_vector.type())
                #print('Overall attention weights:',overall_attention_weights.shape,overall_attention_weights.type())
                #print('\n')
                
                
                #use LSTM for the decoding
                lstm_decode_output, decoder_hidden = self.lstm_decode(overall_context_vector, decoder_hidden)
                #print('LSTM decoding output',lstm_decode_output.shape)
                
                
                #print('------ Classification ------')
                #after we found the overall attention we are now going for the classification stage
                classify = self.output(lstm_decode_output)
                #print('Linear output:',classify.shape,classify.type())
                
                #print('Classify')
                #print(classify)
                #print('Classify',classify.shape)
                                
                #and check the activations
                activations = torch.softmax(classify, dim=2)
                #print('Activations:',activations.shape,activations.type())
                
                #print('Activations')
                #print(activations)
                #print(activations.shape)
                        
                return activations, decoder_hidden
                
                
                
        def initHidden(self):
        
                return torch.zeros(1, 1, self.hidden_size, device=device)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
