import torch
import operator
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
from Preprocessing import preprocessing
from Preprocessing import word2vec
from pytorch_models import coverage_loss

device1 = torch.device('cuda:0')
'''
Implement the Dynamic Memory Network
- Ask Me Anything: Dynamic Memory Networks for Natural Language Processing
- https://arxiv.org/abs/1506.07285
'''
# https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, Prob, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.prob = Prob
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.prob / float(self.leng - 1 + 1e-6) + alpha * reward   
        
    def __lt__(self, other):
        return self.prob < other.prob             
          
class DMN_beam_search(torch.nn.Module):
        def __init__(self, hidden_units, output_space, dropout_after_linear, dropout_lstm_output, number_of_layers, parallel, tokenizer, device1, pad_emb, eos_emb, sos_emb, pad_token, eos_token, sos_token, max_queue):
        
                super(DMN_beam_search, self).__init__()
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
                self.max_queue = max_queue
                #define a dropout to the teacher forcing input
                self.decoder_input = torch.nn.Dropout(0.0)
                #define the number of layers, depth of the Decoder
                self.number_of_layers = number_of_layers
                #the model has to know when I use parallel, in order to apply the permutations
                self.parallel = parallel
                #after the concatenation of teacher forcing input
                self.attention_combine = torch.nn.Linear(900, hidden_units)
                #use Xavier and He Normal (He-et-al) Initialization
                torch.nn.init.xavier_normal_(self.attention_combine.state_dict()['weight'])               
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
                
                #make rating prediction
                self.rating_prediction = torch.nn.Linear(hidden_units*2,1)
                torch.nn.init.xavier_normal_(self.rating_prediction.state_dict()['weight'])
                self.rating_prediction_dropout = torch.nn.Dropout(0.2)
                
                #define a dropout layer after linear activations
                self.dropout = torch.nn.Dropout(dropout_after_linear)
                
        #Calculate repetitiveness_loss
        def repetiviness(self, mask, activations):
                #count the repetitiveness between the predictions
                #print('Activations',activations.shape) ---> (b*seq, vocab)
                #print('Mask shape',mask.shape) ---> (b*seq)
                repetitiveness_loss = 0
                for index in range(len(activations)-1):
                        repetitiveness_loss += torch.nn.functional.cosine_similarity(activations[index],activations[index+1],0)
                repetitiveness_loss_back = repetitiveness_loss/len(activations)
                
                return repetitiveness_loss_back
                
                
        #because in this Decoder architecture we want to use teacher forcing so we feed the output of our decoder(the real one before we insert the encoder output to our LSTM)
        def forward(self, overall_context_vector_encoder, decoder_hidden, decoder_inputs, target_length,  overall_attention_weights, product_rating, mask, beam_decode):          
                #BEAM SEARCH DECODER
                beam_width = 10 # depth of the search
                topk = 1  # how many sentence do you want to generate
                decoded_batch = []         
                overall_context_vector = overall_context_vector_encoder
                #feed with the real output
                decoder_input = torch.LongTensor([[self.sos_token]]) #SOS TOKEN
                #print('sos token id',sos_token)
                #----> REVIEW
                activations_list = []
                for timestep in range(target_length):
                        
                        # Number of sentence to generate
                        endnodes = []
                        number_required = min((topk + 1), topk - len(endnodes))

                        # starting node -  hidden vector, previous node, word id, logp, length
                        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
                        nodes = PriorityQueue()

                        # start the queue
                        nodes.put((-node.eval(), node))
                        qsize = 1

                        # XXX --> START BEAM SEARCH <----
                        while True:
                                #break if decoding is going to take to long to finish
                                if qsize > self.max_queue: break
                                #take the best node
                                score, n = nodes.get()
                                decoder_input = n.wordid
                                decoder_hidden = n.h
                                #check if I have an end node, a node which is the END OF SEQUENCE TOKEN with a previous node
                                if ( (n.wordid.item()==self.eos_token) and (n.prevNode!=None) ):
                                        endnodes.append((score,n))
                                        #check if I reach the maximum number of sentences
                                        if (len(endnodes)>=number_required):
                                                break #if yes stop
                                        else:
                                                continue #else continue, but with new sentence because for this one I met the EOS token               
                                # XXX: Decode for one step
                                if (decoder_input==self.sos_token):
                                        teacher_forcing_input = self.sos_emb
                                elif (decoder_input==self.eos_token):
                                        teacher_forcing_input = self.eos_emb
                                elif (decoder_input==self.pad_token):
                                        teacher_forcing_input = self.pad_emb
                                else:
                                        teacher_forcing_input = preprocessing.index_to_word_mapping(decoder_input, self.tokenizer)
                                        teacher_forcing_input = word2vec.word2vec(teacher_forcing_input, decoder_input)
                                        teacher_forcing_input = torch.tensor(teacher_forcing_input, dtype=torch.torch.float32).unsqueeze(0).unsqueeze(0)#[1, 300]    
                                #apply Dropout to the decoder's input
                                teacher_forcing_input = self.decoder_input(teacher_forcing_input).to(device1)                         
                                #combine teacher forcing input with the output hidden from lstm
                                teacher_forcing_result = torch.cat((overall_context_vector, teacher_forcing_input),dim=2) # [b, 1, 900]
                                #combine them
                                overall_context_vector_tf = torch.nn.functional.relu(self.attention_combine(teacher_forcing_result)) # [b, 1, 600] from [b, 1, 900]
                                overall_context_vector, decoder_hidden = self.lstm_decode(overall_context_vector_tf, decoder_hidden) # [b, 1, 600]
                                #using the dropout after the lstm
                                decoder_hidden_states = self.lstm_dropout(overall_context_vector)
                                #after we found the overall attention we are now going for the classification stage
                                classify = self.output(decoder_hidden_states)
                                # XXX: Decode for one step
                                activations_list.append(classify)
                                # SET THE TOP BEAM SEARCH
                                probs, indices = torch.topk(classify, beam_width)
                                probs = probs.squeeze(1).squeeze(1)
                                indices = indices.squeeze(1).squeeze(1)
                                nextnodes = []
                                for new_k in range(beam_width):
                                        decoded_t = indices[0][new_k].view(1, -1)
                                        prob = probs[0][new_k].item()
                                        #initialize a new Node
                                        node = BeamSearchNode(decoder_hidden, n, decoded_t, n.prob+prob, n.leng+1)
                                        score = -node.eval()
                                        nextnodes.append((score,node)) 
                                #put them into the queue
                                for i in range(len(nextnodes)):
                                        score, nn = nextnodes[i]
                                        nodes.put((score,nn))
                                #increase the queu size
                                qsize += len(nextnodes) - 1
                        #choose n best paths
                        if (len(endnodes)==0):
                                endnodes = [nodes.get() for _ in range(topk)]
                        utterances = []
                        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                                utterance = []
                                utterance.append(n.wordid.item())
                                #back track
                                while (n.prevNode!=None):
                                        n = n.prevNode
                                        utterance.append(n.wordid.item())  
                                utterance = utterance[::-1] #reverse
                                utterances.append(utterance)
                        decoded_batch.append(utterances) 
                #concatenate all the decoder states
                activations_list = torch.cat(activations_list, dim=1)
                #-------------------> RATING
                product_rating = product_rating.unsqueeze(2)
                rating_overall = torch.cat((overall_context_vector_encoder, overall_context_vector), 2)
                #format the loss function inputs, flatten the output with respect to the batch size
                repetitiveness_loss_back = coverage_loss.coverage_loss(mask, decoded_batch,self.device1,beam_decode)
                # multiply (1,normalize_reviews)x(normalize_reviews,1) + (1,hidden_units*2)x(hidden_units*2,1)
                rating_prediction = ( torch.bmm(overall_attention_weights,product_rating) + (1-repetitiveness_loss_back)*self.rating_prediction_dropout(torch.tanh(self.rating_prediction(rating_overall))) )
                      
                return activations_list, rating_prediction, repetitiveness_loss_back, decoded_batch
                
        #INITIALIZE HIDDEN STATES
        def initHidden(self):  
                for value in self.lstm_decode.state_dict():
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
                        if 'bias' in value:
                                #print(value,param.shape,'Zeros')
                                #torch.nn.init.constant_(self.lstm_decode.state_dict()[value], val=0)
                                self.lstm_decode.state_dict()[value].data[self.hidden_units:self.hidden_units*2].fill_(1)#set the forget gate to 1| (b
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
