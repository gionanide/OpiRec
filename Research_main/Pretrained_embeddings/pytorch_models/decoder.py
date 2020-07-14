#!usr/bin/python
import torch
from pytorch_models import Attention as bahdanau_attention
from pytorch_models import coverage_loss
from classes import word2vec
import random

class DMN(torch.nn.Module):

        def __init__(self, hidden_units, output_space, dropout_after_linear, dropout_lstm_output, number_of_layers, parallel, tokenizer, device1, pad_emb, eos_emb, sos_emb, pad_token, eos_token, sos_token, beam_decode):
        
                super(DMN, self).__init__()
                
                #initializa some properties
                self.hidden_units = hidden_units
                self.output_space = output_space
                self.tokenizer = tokenizer
                self.device1 = device1
                self.pad_emb = pad_emb
                self.eos_emb = eos_emb
                self.sos_emb = sos_emb
                self.pad_token = pad_token
                self.eos_token = eos_token
                self.sos_token = sos_token
                self.beam_decode = beam_decode
                
                #define a dropout to the teacher forcing input
                self.decoder_input_dropout = torch.nn.Dropout(0.2)
                
                #define the number of layers, depth of the Decoder
                self.number_of_layers = number_of_layers
                
                #the model has to know when I use parallel, in order to apply the permutations
                self.parallel = parallel
                
                #after the concatenation of teacher forcing input
                self.attention_combine = torch.nn.Linear(900, hidden_units)
                #use Xavier and He Normal (He-et-al) Initialization
                torch.nn.init.xavier_normal_(self.attention_combine.state_dict()['weight'])
                
                #apply dropout after relu activation
                self.relu_dropout = torch.nn.Dropout(0.2)
                                
                #an lstm before the output to take the previous output as well
                self.lstm_decode = torch.nn.LSTM(hidden_units, hidden_units, bidirectional=False, num_layers=self.number_of_layers, bias=True, batch_first=True)
                
                #define a dropout layer after the lstm
                self.lstm_dropout = torch.nn.Dropout(dropout_lstm_output)
                
                #build the classification procedure
                self.output = torch.nn.Linear(hidden_units, output_space)
                #use Xavier and He Normal (He-et-al) Initialization
                torch.nn.init.xavier_normal_(self.output.state_dict()['weight'])
                
                #use this to make an overall value of the context vector to feed the rating prediction procedure
                self.rating_prediction = torch.nn.Linear(hidden_units*2, 1)
                torch.nn.init.xavier_normal_(self.rating_prediction.state_dict()['weight'])
                self.rating_prediction_dropout = torch.nn.Dropout(0.2)
                
                
                #define a dropout layer after linear activations
                self.dropout = torch.nn.Dropout(dropout_after_linear)
                
                
        #because in this Decoder architecture we want to use teacher forcing so we feed the output of our decoder(the real one before we insert the encoder output to our LSTM)
        def forward(self, overall_context_vector_encoder, decoder_hidden, decoder_inputs, target_length,  overall_attention_weights, product_rating, mask, teacher_forcing_ratio):   
        
                '''
                if (self.parallel):
                        #RESHAPE hidden states to (seq, b, f) because they were in shape (b, seq, f)
                        decoder_hidden = list(decoder_hidden)
                        for i in range(len(decoder_hidden)):
                                decoder_hidden[i] = decoder_hidden[i].permute(1,0,2).contiguous()
                        decoder_hidden = tuple(decoder_hidden)
                '''
                           
                overall_context_vector = overall_context_vector_encoder

                #---------------------------------------------------------------------------------------------------------------------> REVIEW
                activations_list = []
                for timestep in range(target_length):
                
                        '''
                        #use LSTM for the decoding
                        if (self.parallel):
                                self.lstm_decode.flatten_parameters() #|| Also because the dataparallel split in dimension 0, we have to reshape the hidden states dimensions
                        '''
                        
                        #generate a random number between 0-1, the bigger the ratio the more probable the number to be smaller that it
                        random_number = random.uniform(0, 1)
                        if ( (random_number <= teacher_forcing_ratio) or (timestep==0) ):
                                #feed with the real output
                                teacher_forcing_input = decoder_inputs[:,timestep,:].unsqueeze(1) # [b, 1, 300]
                                
                                #print('ready feed')    
                                
                        else:
                        
                                #print('feed tiself')
                        
                                teacher_forcing_input = torch.empty(classify.shape[0], 300)
                                #otherwise feed the model with it's previous output, this becomes more probable as tha training goes on
                                for index_activation, activation in enumerate(classify):
                                        word_id = torch.argmax(activation).item()
                                        if (word_id==self.sos_token):
                                                teacher_forcing_input[index_activation] = self.sos_emb.squeeze(0).squeeze(0)
                                        elif (word_id==self.eos_token):
                                                teacher_forcing_input[index_activation] = self.eos_emb.squeeze(0).squeeze(0)
                                        elif (word_id==self.pad_token):
                                                teacher_forcing_input[index_activation] = self.pad_emb.squeeze(0).squeeze(0)
                                        else:
                                                word = word2vec.index_to_word_mapping(word_id, self.tokenizer)
                                                word = word2vec.word2vec(word, word_id)
                                                word = torch.tensor(word, dtype=torch.torch.float32)
                                                teacher_forcing_input[index_activation] = word #[1, 300]
                                        
                                teacher_forcing_input = teacher_forcing_input.unsqueeze(1).to(self.device1)# [b, 1, 300]
                                
                                #nan = torch.isnan(teacher_forcing_input)
                                #print('teacher_forcing_input_inpuuuuuuuuuuuuuuut',(nan!=0).nonzero())
                                
                                #teacher_forcing_input[teacher_forcing_input!=teacher_forcing_input] = 0 #replace nan values with zeros
                                
                        print("teacher_forcing_input",teacher_forcing_input.shape)
                        
                        #apply Dropout to the decoder's input
                        teacher_forcing_input = self.decoder_input_dropout(teacher_forcing_input)
                        
                        #nan = torch.isnan(teacher_forcing_input)
                        #print('teacher_forcing_input',(nan!=0).nonzero())
                           
                        #combinet eacher forcing input with the output hidden from lstm
                        teacher_forcing_result = torch.cat((overall_context_vector, teacher_forcing_input),dim=2) # [b, 1, 900]
                        
                        #nan = torch.isnan(teacher_forcing_result)
                        #print('teacher_forcing_result',(nan!=0).nonzero())
                        
                        #combine them
                        overall_context_vector_tf = torch.nn.functional.relu(self.attention_combine(teacher_forcing_result)) # [b, 1, 600] from [b, 1, 900]
                        
                        #nan = torch.isnan(overall_context_vector_tf)
                        #print('overall_context_vector_tf',(nan!=0).nonzero())
                        
                        overall_context_vector_tf = self.relu_dropout(overall_context_vector_tf)# apply a dropout after rely activation
                        
                        #nan = torch.isnan(overall_context_vector_tf)
                        #print('overall_context_vector_tf1',(nan!=0).nonzero())
                                                       
                        overall_context_vector, decoder_hidden = self.lstm_decode(overall_context_vector_tf, decoder_hidden) # [b, 1, 600]
                        
                        #nan = torch.isnan(overall_context_vector)
                        #print('overall_context_vector',(nan!=0).nonzero())
                        
                        #using the dropout after the lstm
                        decoder_hidden_states = self.lstm_dropout(overall_context_vector)
                        
                        #nan = torch.isnan(decoder_hidden_states)
                        #print('decoder_hidden_states',(nan!=0).nonzero())

                        #after we found the overall attention we are now going for the classification stage
                        classify = self.output(decoder_hidden_states)
                        
                        #nan = torch.isnan(classify)
                        #print('classify',(nan!=0).nonzero())
                        
                        #apply dropout
                        classify_dropout = self.dropout(classify)
                        
                        #nan = torch.isnan(classify_dropout)
                        #print('classify_dropout',(nan!=0).nonzero())
                        
                        #print('Classify',classify.shape)
                        activations_list.append(classify_dropout)
 
                #print(activations_list[0].shape)
 
                #concatenate all the decoder states
                activations_list = torch.cat(activations_list, dim=1)
                
                #print(activations_list.shape)
                
                #---------------------------------------------------------------------------------------------------------------------------> RATING
                product_rating = product_rating.unsqueeze(2)
                rating_overall = torch.cat((overall_context_vector_encoder, overall_context_vector), 2)
                
                #format the loss function inputs, flatten the output with respect to the batch size
                repetitiveness_loss_back = coverage_loss.coverage_loss(mask, activations_list,self.device1,self.beam_decode)
                #print('repetitiveness_loss_back',repetitiveness_loss_back)
                
                # multiply (1,normalize_reviews)x(normalize_reviews,1) + (1,hidden_units*2)x(hidden_units*2,1) (1-repetitiveness_loss_back)*
                rating_prediction = (torch.bmm(overall_attention_weights,product_rating) + (1-repetitiveness_loss_back)*torch.tanh(self.rating_prediction(rating_overall)) )              
                #rating_prediction = ( torch.bmm(overall_attention_weights,product_rating) + self.rating_prediction(rating_overall) )
                '''
                
                ----- CHEKING PROPERTIES -----
                
                print("repetitiveness_loss_back---->",repetitiveness_loss_back.item())
                print("1-repetitiveness_loss_back---->",(1-repetitiveness_loss_back).item())
                print("previous ratings---->",torch.bmm(overall_attention_weights,product_rating).item())
                print("model influence---->",torch.tanh(self.rating_prediction(rating_overall)).item())
                
                if(repetitiveness_loss_back>0.9):
                    print("===========================")
                    print("===========================")
                    print("===========================")
                    print("===========================")
                    print("===========================")
                    print("===========================")
                    print("===========================")
                    print("===========================")
                    print("===========================")
                    print("===========================")
                    
                    '''
                
                        
                return activations_list, rating_prediction, repetitiveness_loss_back, target_length
                
                
                
        #INITIALIZE HIDDEN STATES
        def initHidden(self):  
              
                for value in self.lstm_decode.state_dict():
                
                        #format values
                        param = self.lstm_decode.state_dict()[value]
                        if 'weight_ih' in value:
                                #print(value,param.shape,'Orthogonal')
                                torch.nn.init.orthogonal_(self.lstm_decode.state_dict()[value])#input TO hidden ORTHOGONALLY || Wii, Wif, Wic, Wio
                        elif 'weight_hh' in value:
                                #INITIALIZE SEPERATELY EVERY MATRIX TO BE THE IDENTITY AND THE STACK THEM                        
                                weight_hh_data_ii = torch.eye(self.hidden_units,self.hidden_units)#H_Wii
                                weight_hh_data_if = torch.eye(self.hidden_units,self.hidden_units)#H_Wif
                                weight_hh_data_ic = torch.eye(self.hidden_units,self.hidden_units)#H_Wic
                                weight_hh_data_io = torch.eye(self.hidden_units,self.hidden_units)#H_Wio
                                weight_hh_data = torch.stack([weight_hh_data_ii,weight_hh_data_if,weight_hh_data_ic,weight_hh_data_io], dim=0)
                                weight_hh_data = weight_hh_data.view(self.hidden_units*4,self.hidden_units)
                                #print(value,param.shape,weight_hh_data.shape,self.number_of_layers,self.hidden_units,'Identity')
                                self.lstm_decode.state_dict()[value].data.copy_(weight_hh_data)#hidden TO hidden IDENTITY.state_dict()[value].data.copy_(weight_hh_data)#hidden TO hidden IDENTITY
                        elif 'bias' in value:
                                #print(value,param.shape,'Zeros')
                                #torch.nn.init.constant_(self.lstm_decode.state_dict()[value], val=0)
                                self.lstm_decode.state_dict()[value].data[self.hidden_units:self.hidden_units*2].fill_(1)#set the forget gate to 1| (b_ii|b_if|b_ig|b_io)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
