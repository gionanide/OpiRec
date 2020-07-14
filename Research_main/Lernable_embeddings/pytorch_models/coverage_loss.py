#!usr/bin/python
import torch

'''

Get To The Point: Summarization with Pointer-Generator Networks: https://arxiv.org/pdf/1704.04368.pdf

'''
def coverage_loss(mask,activations_mask,device,beam_decode): #loss is bounded between [0,1]
        #activations_mask = activations.copy()
        if (beam_decode):
                #find the maximum sequence length
                max_length = 0
                for index, x in enumerate(activations_mask):
                        if (len(x[0])>max_length):
                                max_length=len(x[0])#find the max length for padding
                        activations_mask[index] = x[0]
                for index, x in enumerate(activations_mask):
                        activations_mask[index] += [0] * (max_length - len(x))# zero pad all the lists to match the same size
                activations_mask = torch.FloatTensor(activations_mask)
                activations_mask = activations_mask.unsqueeze(0)#give the first dimension 1 because batch_Size=1
                mask = (activations_mask == 0)#find where is zero and tehn replace with False/0s and True/1s accordingly
                mask = mask.squeeze(0)#we do not need the first dimension
        else:
                mask = mask.view(activations_mask.shape[0],activations_mask.shape[1])# reshape the mask

        activations_distribution = torch.nn.functional.softmax(activations_mask,dim=2)# pass the activation through a softmax || Recall that this stage is happening again inside the crossentropy function, we do this again here only to calcualte the overlapping distributions||
        overall_coverage = 0#initialize the overall loss for the batch
        
        for index, activation in enumerate(activations_distribution): #iterate every batch
        
                loss = 0 #initialize the coverage loss for a sentence in the batch
                overall_weights = torch.zeros(activation.shape[1]).to(device)#the first coverage vector equals to zero
                
                for time, timestep in enumerate(activation): #iterate every single timestep           
                        
                        if (mask[index][time]==0):#it means that we foudn the first padding symbol, so there are only padding symbols from this point till the end of sequence, se we stop
                                break
                     
                        loss = torch.sum(torch.min(overall_weights,timestep)) #find the min element of the current dimensions, compare elementwise
                        overall_weights = ((torch.sum(activation[:time+1,:],dim=0))/activation.shape[0]).to(device) #after timestep = 0 || now we are taking the (unormalized)sum of the previous activations                        
                overall_coverage = overall_coverage + loss # keep track of the general conerage loss of the batch
                
        cov_loss = overall_coverage/activations_distribution.shape[0] #divide by the batch size
        #print(index,'loss',cov_loss)
        
        
        return cov_loss
        
        
        
        
