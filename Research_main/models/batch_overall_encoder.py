#!usr/bin/python
from pytorch_models import batch_bahdanau_attention_pytorch as bahdanau_attention
import torch


class Overall_Encoder(torch.nn.Module):


        def __init__(self, hidden_units, input_dropout, episodes, normalize_product_reviews, number_of_layers, parallel):
        
                #create a child class of Recommender_v1
                super(Overall_Encoder, self).__init__()
                #initializa some properties
                self.hidden_units = hidden_units
                self.similarity_threshold = 0.9
                #define the number of episodes for the DMN
                self.episodes = episodes
                #define dropout layer to apss all the inputs
                self.input_dropout_layer = torch.nn.Dropout(input_dropout)
                #define the number of layers, depth of the Encoder
                self.number_of_layers = number_of_layers
                #define if we use parallel models, if so we have to reshape our hidden states accordingly
                self.parallel = parallel                
                #start build the model, when we feed our model the embeddings are already ready
                self.user_lstm = torch.nn.LSTM(hidden_units, hidden_units, bidirectional=True, num_layers=self.number_of_layers, dropout=0.2, bias=True, batch_first=True)
                self.product_lstm = torch.nn.LSTM(hidden_units, hidden_units, bidirectional=True, num_layers=self.number_of_layers, dropout=0.2, bias=True, batch_first=True)
                #define custom attention layers
                self.user_profile_attention = bahdanau_attention.BahdanauAttention(hidden_units*2, multiple=1)
                self.overall_attention = bahdanau_attention.BahdanauAttention(hidden_units*2, multiple=3)
                #we are not using neighbour
                #self.neighbourhood_lstm = torch.nn.LSTM(hidden_units, hidden_units, bidirectional=False, num_layers=1)

        def calculate_cosine(self, base, span):
                return torch.nn.functional.cosine_similarity(base,span,dim=1).mean()#cosine similarity between the rows of the two matrices
        
        def forward(self, user_inputs, product_inputs, deep_memory_calculations):              
                #pass them from the input dropout layer after the embedding
                user_inputs = self.input_dropout_layer(user_inputs)
                product_inputs = self.input_dropout_layer(product_inputs)
                #neighbourhood_inputs = self.input_dropout_layer(neighbourhood_inputs)        
                #-------> LSTM
                #print('------ LSTMs ------')
                '''
                #feed the user/product LSTMs to make user's profile
                if (self.parallel):
                        self.user_lstm.flatten_parameters()
                '''        
                user_lstm_output, user_h = self.user_lstm(user_inputs)
                user_h_forward = user_h[0].view(self.number_of_layers, user_lstm_output.shape[0], 600)
                user_h_backrward =   user_h[1].view(self.number_of_layers, user_lstm_output.shape[0], 600)            
                user_h = (user_h_forward, user_h_backrward)
                '''
                if (self.parallel):
                        #RESHAPE hidden states to (b, seq, f) in order for the parallel to split them correctly (split based on the dim=0)
                        user_h = list(user_h)
                        for i in range(len(user_h)):
                                user_h[i] = user_h[i].permute(1,0,2).contiguous()
                        user_h = tuple(user_h)
                
                if (self.parallel):
                        self.product_lstm.flatten_parameters()
                '''       
                product_lstm_output, product_h = self.product_lstm(product_inputs)                   
                product_h_forward = product_h[0].view(self.number_of_layers, product_lstm_output.shape[0], 600)
                product_h_backrward =   product_h[1].view(self.number_of_layers, product_lstm_output.shape[0], 600)               
                product_h = (product_h_forward, product_h_backrward)
                #assign the first hidden state of the decoder, and make them 3D tensors
                decoder_hidden = (product_h[0], product_h[1])
                #returns: lstm sequence: (user_reviews, 1, 600), hidden: (2, 1, 600)
                '''
                if (self.parallel):
                        #RESHAPE hidden states to (b, seq, f) in order for the parallel to split them correctly (split based on the dim=0)
                        decoder_hidden = list(decoder_hidden)
                        for i in range(len(decoder_hidden)):
                                decoder_hidden[i] = decoder_hidden[i].permute(1,0,2).contiguous()
                        decoder_hidden = tuple(decoder_hidden)
                '''
                #USER SELF ATTENTION
                user_context_vector, user_attention_weights = self.user_profile_attention(user_lstm_output, user_lstm_output, user_lstm_output, user_lstm_output)
                #initialize the first concept of the memeory context as just the mean of the hidden states
                overall_context_vector = (torch.sum(product_lstm_output, dim=1)/product_lstm_output.shape[1]).unsqueeze(1)
                previous_overall_context_vector = overall_context_vector.detach()
                #print('Going into memory')
                #--------> Dynamic Memory Network
                #pass the memory multiple times
                for episode in range(self.episodes):
                        #-----------> Here I must apply selection criterion
                        #overall attention
                        #previous_overall_context_vector = overall_context_vector.detach()
                        overall_context_vector, overall_attention_weights = self.overall_attention(product_lstm_output, user_context_vector, overall_context_vector, overall_context_vector)
                        #print('previous_overall_context_vector',previous_overall_context_vector)
                        #print('overall_context_vector',overall_context_vector)
                        deep_memory_calculations+=1                
                return previous_overall_context_vector, overall_attention_weights, decoder_hidden, deep_memory_calculations
  
        #INITIALIZE HIDDEN STATES
        def initHidden(self):
        
                for value in self.user_lstm.state_dict():
                
                        #format values
                        param = self.user_lstm.state_dict()[value]
                        if 'weight_ih' in value:
                                #print(value,param.shape,'Orthogonal')
                                torch.nn.init.orthogonal_(self.user_lstm.state_dict()[value])#input TO hidden ORTHOGONALLY || Wii, Wif, Wic, Wio
                        elif 'weight_hh' in value:
                                #INITIALIZE SEPERATELY EVERY MATRIX TO BE THE IDENTITY AND THE STACK THEM                        
                                weight_hh_data_ii = torch.eye(self.hidden_units,self.hidden_units)#H_Wii
                                weight_hh_data_if = torch.eye(self.hidden_units,self.hidden_units)#H_Wif
                                weight_hh_data_ic = torch.eye(self.hidden_units,self.hidden_units)#H_Wic
                                weight_hh_data_io = torch.eye(self.hidden_units,self.hidden_units)#H_Wio
                                weight_hh_data = torch.stack([weight_hh_data_ii,weight_hh_data_if,weight_hh_data_ic,weight_hh_data_io], dim=0)
                                weight_hh_data = weight_hh_data.view(self.hidden_units*4,self.hidden_units)
                                #print(value,param.shape,weight_hh_data.shape,self.number_of_layers,self.hidden_units,'Identity')
                                self.user_lstm.state_dict()[value].data.copy_(weight_hh_data)#hidden TO hidden IDENTITY.state_dict()[value].data.copy_(weight_hh_data)#hidden TO hidden IDENTITY
                        elif 'bias' in value:
                                #print(value,param.shape,'Zeros')
                                #torch.nn.init.constant_(self.user_lstm.state_dict()[value], val=0)
                                self.user_lstm.state_dict()[value].data[self.hidden_units:self.hidden_units*2].fill_(1)#set the forget gate | (b_ii|b_if|b_ig|b_io)
                        
                        
                for value in self.product_lstm.state_dict():
                
                        #format values
                        param = self.product_lstm.state_dict()[value]
                        if 'weight_ih' in value:
                                #print(value,param.shape,'Orthogonal')
                                torch.nn.init.orthogonal_(self.product_lstm.state_dict()[value])#input TO hidden ORTHOGONALLY || Wii, Wif, Wic, Wio
                        elif 'weight_hh' in value:
                                #INITIALIZE SEPERATELY EVERY MATRIX TO BE THE IDENTITY AND THE STACK THEM                        
                                weight_hh_data_ii = torch.eye(self.hidden_units,self.hidden_units)#H_Wii
                                weight_hh_data_if = torch.eye(self.hidden_units,self.hidden_units)#H_Wif
                                weight_hh_data_ic = torch.eye(self.hidden_units,self.hidden_units)#H_Wic
                                weight_hh_data_io = torch.eye(self.hidden_units,self.hidden_units)#H_Wio
                                weight_hh_data = torch.stack([weight_hh_data_ii,weight_hh_data_if,weight_hh_data_ic,weight_hh_data_io], dim=0)
                                weight_hh_data = weight_hh_data.view(self.hidden_units*4,self.hidden_units)
                                #print(value,param.shape,weight_hh_data.shape,self.number_of_layers,self.hidden_units,'Identity')
                                self.product_lstm.state_dict()[value].data.copy_(weight_hh_data)#hidden TO hidden IDENTITY
                        elif 'bias' in value:
                                #print(value,param.shape,'Zeros')
                                #torch.nn.init.constant_(self.product_lstm.state_dict()[value], val=0)
                                self.product_lstm.state_dict()[value].data[self.hidden_units:self.hidden_units*2].fill_(1)#se he forget gate | ( ----- b_ii 0-hidden_units | ----- b_if hidden_units-2*hidden_units | ----- b_ig 2*hidden_units-3*hidden_units| ----- b_io 3*hidden_units-4*hidden_units)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
