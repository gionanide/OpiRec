#!usr/bin/python
import torch

'''

Implement Maxout network:

- Maxout Networks

- https://arxiv.org/pdf/1302.4389.pdf

'''

device1 = torch.device('cuda:0')

class Maxout_network(torch.nn.Module):

        def __init__(self, input_space, linear_nodes, num_layers, maxout_units):
        
                #call this function to inheritate from torch.nn.Module
                super(Maxout_network, self).__init__()
                
                #define the number of hidden units in Linear layers
                self.linear_nodes = linear_nodes
                
                #define the deapth of each layers
                self.num_layers = num_layers
                
                #define the number of maxout units
                self.maxout_units = maxout_units
                
                #make a list with each layer
                self.layers_forward = []
                
                #the first layer must handle the input, handle an input: (inputs_shape, linear_nodes)
                self.layers_forward.append(torch.nn.ModuleList([torch.nn.Linear(input_space,linear_nodes) for maxout_unit in range(maxout_units)]))
                
                #here we are defining the depth of our Maxout network
                for layer in range(1,num_layers):             
                        
                        #make a list with the number of layers, every layer contains number of Linear networks that equals the nnumber of maxout units, handles an input (linear_nodes, linear_nodes)
                        self.layers_forward.append(torch.nn.ModuleList([torch.nn.Linear(linear_nodes,linear_nodes) for maxout_unit in range(maxout_units)]))
                        
                        
                #print('Linear nodes in each Maxout unit:',linear_nodes)
                #print('Number of layers(depth):',num_layers)
                #print('Maxout units:',maxout_units)                
                
                
        def forward(self, input_tensor):
        
                #iterate all the layers
                for layer in range(self.num_layers):
                
                        #for every layer we call the maxout
                        input_tensor = self.maxout(input_tensor, self.layers_forward[layer])    
                
                
                return input_tensor         
                
                
        #iterate every maxout unit in a pecific layer and conclude to a Tensor with dimensions: (1, maxout_units)
        def maxout(self, input_tensor, layer):
        
                #initalize a list to keep the maximum from every maxout unit
                maxout_outputs = []
                
                #move the Linear layers to CUDA
                layer = layer.to(device1)
        
                #iterate all maxout units
                for maxout_unit in range(self.maxout_units):

                        #pass the input in the first layer
                        output = layer[maxout_unit](input_tensor)
                        
                        #take the max from each network(maxout units) 
                        max_output = torch.max(output)
                        
                        #and append every max to a list
                        maxout_outputs.append(max_output)
                        
                        
                #convert this list to a Pytorch tensor and reshape it
                maxout_output_tensor = torch.FloatTensor(maxout_outputs)
                maxout_output_tensor = maxout_output_tensor.unsqueeze(0).unsqueeze(0)
                maxout_output_tensor = maxout_output_tensor.to(device1)
                
                
                return maxout_output_tensor       
                
                
                
        
'''
#-------------------------------------------------------------- TEST THE NETWORK -------------------------------------------------

input_tensor = torch.randn(1, 1, 300, dtype=torch.float)

print(input_tensor.shape)

input_space = input_tensor.shape[2]
linear_nodes = 20
num_layers = 3
maxout_units = 20

        
output_layer = Maxout_network(input_space, linear_nodes, num_layers, maxout_units)

maxout_output_tensor = output_layer(input_tensor)

print(maxout_output_tensor.shape)
'''









