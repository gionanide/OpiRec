#!usr/bin/python
from pytorch_models import bahdanau_attention_pytorch as bahdanau_attention
import torch


class Recommender_v1_encoder(torch.nn.Module):


        def __init__(self, hidden_units):
        
                #create a child class of Recommender_v1
                super(Recommender_v1_encoder, self).__init__()
                
                #initializa some properties
                self.hidden_units = hidden_units
                
                
                #start build the model, when we feed our model the embeddings are already ready
                self.user_lstm = torch.nn.LSTM(hidden_units, hidden_units)
                self.product_lstm = torch.nn.LSTM(hidden_units, hidden_units)
                self.neighbourhood_lstm = torch.nn.LSTM(hidden_units, hidden_units)

                
        def forward(self, user_inputs, product_inputs, neighbourhood_inputs):
        
                #print('------ Inputs ------')
                #print('User input:',user_inputs.shape,user_inputs.type())
                #print('Product input:',product_inputs.shape,product_inputs.type())
                #print('Neighbourhood input:',neighbourhood_inputs.shape,neighbourhood_inputs.type())
                #print('\n')
        
        
        
                #----------------------------------------------------------------------------------------------------------------> LSTM
                #print('------ LSTMs ------')
                #feed the user/product LSTMs to make user's profile
                user_lstm_output, user_h = self.user_lstm(user_inputs)
                #print('User lstm:',user_lstm_output.shape,user_lstm_output.type())
                
                neighbourhood_lstm_output, neighbourhood_h = self.neighbourhood_lstm(neighbourhood_inputs)
                #print('Neighbourhood lstm:',neighbourhood_lstm_output.shape,neighbourhood_lstm_output.type())
                
                product_lstm_output, product_h = self.product_lstm(product_inputs)
                #print('Product lstm:',product_lstm_output.shape,product_lstm_output.type())
                #print('\n')
                
                
                return user_lstm_output, user_h, neighbourhood_lstm_output, neighbourhood_h, product_lstm_output, product_h
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
