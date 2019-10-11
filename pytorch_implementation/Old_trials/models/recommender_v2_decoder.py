
#!usr/bin/python
from pytorch_models import bahdanau_attention_pytorch as bahdanau_attention
import torch


class Recommender_v2_decoder(torch.nn.Module):


        def __init__(self, hidden_units, output_vocabulary_size, dropout_after_linear, dropout_lstm_output):
        
                #create a child class of Recommender_v1
                super(Recommender_v2_decoder, self).__init__()
                
                #initializa some properties
                self.hidden_units = hidden_units
                self.output_space = output_vocabulary_size
                
                #define custom attention layers
                self.user_profile_attention = bahdanau_attention.BahdanauAttention(hidden_units, multiple=1)
                self.neighbourhood_profile_attention = bahdanau_attention.BahdanauAttention(hidden_units, multiple=1)
                self.overall_attention = bahdanau_attention.BahdanauAttention(hidden_units, multiple=3)
                
                
                #choose where to look more, overall attention or the previous word
                self.combine_attention = bahdanau_attention.BahdanauAttention(hidden_units, multiple=2)
                
                
                
                #an lstm before the output to take the previous output as well
                self.lstm_decode = torch.nn.LSTM(hidden_units, hidden_units)
                
                #define a dropout layer after the lstm
                self.lstm_dropout = torch.nn.Dropout(dropout_lstm_output)
                
                #build the classification procedure
                self.output = torch.nn.Linear(hidden_units, output_vocabulary_size)
                
                #define a dropout layer after linear activations
                self.dropout = torch.nn.Dropout(dropout_after_linear)
                
                #and the activation function after the Dense layer
                #self.softmax = torch.softmax(dim=2)
                
                
        #because in this Decoder architecture we want to use teacher forcing so we feed the output of our decoder(the real one before we insert the encoder output to our LSTM)
        def forward(self, user_lstm_output, user_h, neighbourhood_lstm_output, neighbourhood_h, product_lstm_output, product_h, decoder_hidden, decoder_inputs):
               
                
                #------------------------------------------------------------------------------------------------------------------> ATTENTION
                #print('------ Attention ------')
                #make attention for user and neighbourhood profile first
                #print('User lstm output:',user_lstm_output.shape)
                #print('Neighbourhood lstm output:',neighbourhood_lstm_output.shape)
                #print('Product lstm output:',product_lstm_output.shape)
                #print('Decoder inputs:',decoder_inputs.shape)
                
                user_context_vector, user_attention_weights = self.user_profile_attention(user_lstm_output, user_lstm_output, user_lstm_output)
                #print('User attention:',user_context_vector.shape, user_context_vector.type())
                #print('User attention weights:',user_attention_weights.shape, user_attention_weights.type())
                
                
                
                neighbourhood_context_vector, neighbourhood_attention_weights = self.neighbourhood_profile_attention(neighbourhood_lstm_output, neighbourhood_lstm_output, neighbourhood_lstm_output)
                #print('Neighbourhood attention:',neighbourhood_context_vector.shape, neighbourhood_context_vector.type())
                #print('Neighbourhood attention weights:',neighbourhood_attention_weights.shape, neighbourhood_attention_weights.type())
                
                
                #----------------------------------------------------------------------------------------------------------------------------------------------------------> Here I must apply DMN
                #after this we are ready to take the outputs from the profiles that we conclude, user and neighbourhood and find the overall attention
                overall_context_vector, overall_attention_weights = self.overall_attention(product_lstm_output, user_context_vector, neighbourhood_context_vector)
                #print('Overall attention:',overall_context_vector.shape,overall_context_vector.type())
                #print('Overall attention weights:',overall_attention_weights.shape,overall_attention_weights.type())
                #print('\n')
                
                
                
                
                
                
                #----------------------------------------------------------------------------------------------------------------------------> Going for decoding the context vector
                
                #USE this if you want to cancatenate the previous output of the decoder with the lstm output
                #concatenate previous output of the model with the new attention
                #concat_with_previous_output = torch.cat((lstm_decode_output, decoder_inputs), 2)
                #print('Concatenation output',concat_with_previous_output.shape)
                
                
                #choose where to look more, overall attention or the previous word
                combine_attention_context_vector, combine_attention_attention_weights = self.combine_attention(overall_context_vector, decoder_inputs, decoder_inputs)
                #print('User attention:',combine_attention_context_vector.shape, combine_attention_context_vector.type())
                #print('User attention weights:',combine_attention_attention_weights.shape, combine_attention_attention_weights.type())
                
                
                
                
                
                
                #use LSTM for the decoding
                lstm_decode_output, decoder_hidden = self.lstm_decode(combine_attention_context_vector, decoder_hidden)
                #print('LSTM decoding output',lstm_decode_output.shape)
                
                
                
                
                #using the dropout after the lstm
                lstm_dropout_output = self.lstm_dropout(lstm_decode_output)
                #print('LSTM dropout output:',lstm_dropout_output.shape,lstm_dropout_output.type())
                
                
                

                #print('------ Classification ------')
                #after we found the overall attention we are now going for the classification stage
                classify = self.output(lstm_dropout_output)
                #print('Linear output:',classify.shape,classify.type())
                
                #print('Classify')
                #print(classify)
                #print('Classify',classify.shape)
                
                
                #apply the dropout layer
                overall_output = self.dropout(classify)
                #print('overall_output',overall_output.shape)
                
                
                                
                #and check the activations
                activations = torch.softmax(overall_output, dim=2)
                #print('Activations:',activations.shape,activations.type())
                
                #print('Activations')
                #print(activations)
                #print(activations.shape)
                        
                return activations, decoder_hidden
                
                
                
        def initHidden(self):
        
                return torch.zeros(1, 1, self.hidden_size, device=device)
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
