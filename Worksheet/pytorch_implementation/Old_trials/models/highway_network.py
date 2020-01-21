#!usr/bin/python
import torch
'''

Implement the Highway network: 

- Highway Networks

- https://arxiv.org/pdf/1505.00387.pdf

'''
class Highway_network(torch.nn.Module):

        def __init__(self, neurons, num_layers, function):
        
                super(Highway_network, self).__init__()
                
                #define the number of hidden units in Linear layers
                self.neurons = neurons
                
                #define the deapth of each layers
                self.num_layers = num_layers
                
                #define the activation function in the normal layer H
                self.function = function
                
                '''
                
                We want to implement: T_gate * H_gate + (1 - T_gate) * C_gate
                
                '''
                
                #T-gate, define which parts and in which volume to change from input, DECISION: pass output of H_gate or output of C_gate(output of this gate is the input)
                self.T_gate = torch.nn.ModuleList([torch.nn.Linear(neurons,neurons) for layers in range(num_layers)])
                
                #H-gate, change the input
                self.H_gate = torch.nn.ModuleList([torch.nn.Linear(neurons,neurons) for layers in range(num_layers)])
                
                #C-gate, keep the input as follows
                self.C_gate = torch.nn.ModuleList([torch.nn.Linear(neurons,neurons) for layers in range(num_layers)])
                
                
                
                
                
        def forward(self, input_tensor):
        
                #iterate all layers
                for layer in range(self.num_layers):
        
                        #first go to the T gate
                        T_gate_output = torch.nn.functional.sigmoid(self.T_gate[layer](input_tensor))
                        
                        #following the H gate
                        H_gate_output = self.function(self.H_gate[layer](input_tensor))
                        
                        #just passing the input
                        C_gate_output = self.C_gate[layer](input_tensor)
                        
                        #conclude
                        output_tensor = T_gate_output * H_gate_output + (1 - T_gate_output) * C_gate_output
                        
                        
                return output_tensor
                        
                        
