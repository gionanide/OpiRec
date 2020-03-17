#!usr/bin/python
import torch



#make a seperate class for Bahdanau attention layer
class BahdanauAttention(torch.nn.Module):

        #initialize the class calling the default function
        def __init__(self, units, multiple):
                #because we need subclass model
                super(BahdanauAttention, self).__init__()
                #check for multiple inputs
                self.multiple = multiple
                #define differente value for the output units
                output_units = units
                #make two dense layers because need a weighted sum
                self.W_1 = torch.nn.Linear(units, output_units)
                #use Xavier and He Normal (He-et-al) Initialization
                torch.nn.init.xavier_normal_(self.W_1.state_dict()['weight'])
                self.W_2 = torch.nn.Linear(units, output_units) 
                torch.nn.init.xavier_normal_(self.W_2.state_dict()['weight'])
                self.W_3 = torch.nn.Linear(units, output_units)                   
                torch.nn.init.xavier_normal_(self.W_3.state_dict()['weight'])
                self.W_4 = torch.nn.Linear(units, output_units)                   
                torch.nn.init.xavier_normal_(self.W_4.state_dict()['weight'])
                #self.maching_score = torch.nn.Linear(units, units) 
                #define the output 
                self.V = torch.nn.Linear(output_units, 1)
                torch.nn.init.xavier_normal_(self.V.state_dict()['weight'])

        def forward(self, encoder_inputs, encoder_inputs1, encoder_inputs2, encoder_inputs3):
                if(self.multiple==3):
                        #print('product_lstm_output',encoder_inputs.shape)
                        #print('user_context_vector',encoder_inputs1.shape)
                        #print('overall context',encoder_inputs2.shape)
                        #print('encoder_inputs',encoder_inputs)
                        #print('encoder_inputs1',encoder_inputs1)
                        #print('encoder_inputs2',encoder_inputs2)
                        score = self.V(torch.tanh( self.W_2(encoder_inputs) + self.W_1(encoder_inputs1) + self.W_3(encoder_inputs2) ) )
                        #print('Scores 3:',score.shape)
                        #print('Scores 3:',score)
                elif(self.multiple==2):
                        #find the score for every hidden state, shape ----> (batch_size, max_length, hidden_size)
                        #in our case the batch_size is the reviews so we assign weight to every review
                        score = self.V(torch.tanh( self.W_2(encoder_inputs) + self.W_1(encoder_inputs1) ) )            
                        #print('Scores 2:',score.shape)
                        #print('Scores 2:',score)
                elif(self.multiple==1):
                        #find the score for every hidden state, shape ----> (batch_size, max_length, hidden_size)
                        #in our case the batch_size is the reviews so we assign weight to every review
                        score = self.V(torch.tanh(self.W_2(encoder_inputs) ))            
                        #print('Scores 1:',score.shape)
                        #print('Scores 1:',score) 
                elif(self.multiple==4):
                        #use the 4 inputs if you want to use the Coverage model
                        score = self.V(torch.tanh( self.W_2(encoder_inputs) + self.W_1(encoder_inputs1) + self.W_3(encoder_inputs2) + self.W_4(encoder_inputs3) ) )
                        #print('Scores 3:',score.shape)
                        #print('Scores 3:',score)      
                #reshape score
                score = score.squeeze(2)
                #attention weights shape ----> (batch_size, max_length, 1), we conclude with 1 because we got the score back
                #-----> axis=0, iterate along the rows
                #-----> axis=1, iterate along the columns
                attention_weights = torch.softmax(score, dim=1).unsqueeze(1)
                #encoder_inputs = encoder_inputs.squeeze()
                #print(encoder_inputs.shape)
                #print(attention_weights.shape)
                #print(encoder_inputs.shape)
                context_vector = torch.bmm(attention_weights,encoder_inputs)
                #the outputs the model returns, RECALL: the outputs must be returned as a list [output1, output2, ...... ,outputN]
                return context_vector, attention_weights
                
                
        


























