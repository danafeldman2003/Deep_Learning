# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 00:26:31 2020

@author: Dana Feldman
"""
import numpy as np
import matplotlib.pyplot as plt 

'''
    Dana Feldman
    YB4
    This neural network predicts if a given person is a male or a female.
    Males and females have 2 features: height and weight
    Each person is represented by a row in the inputs array
    The outputs:
    male = 0
    female = 1
    
'''

#data
inputs = np.array([
[-2, -1],  # Alice
[25, 6],   # Bob
[17, 4],   # Charlie
[-15, -6], # Diana
])
    
outputs = np.array([
[1], # Alice
[0], # Bob
[0], # Charlie
[1], # Diana
])


def sigmoid(x):    
    """
    Activation Function
    input: x value
    output: sigmoind(x) value
    """
    return 1 / (1 + np.exp(-x))



def sigmoid_derivative(x):
    """
    derivative of sigmoid activation function
    input: x value
    output: sigmoid_derivative(x) = x*(x-1)
    """
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, inputs, outputs):
        '''initialize variables in class'''
       #weights used when passing forward from the input layer to the hidden layer
        self.weights_i_to_h = 2 * np.random.random((len(inputs[1,:]), 2)) - 1
       #neurons of the hidden layer
        self.hidden=np.random.random((len(inputs),len(self.weights_i_to_h[1,:])))
       #weights used when passing forward from the hidden layer to the output layer
        self.weights_h_to_o=2 * np.random.random((len(self.hidden[1,:]), 1)) - 1
        #bias used when passing forward from the input layer to the hidden layer
        self.bias_i_to_h=2 * np.random.random((1, len(self.hidden[1,:]))) - 1
        #bias used when passing forward from the hidden layer to the output layer
        self.bias_h_to_o=np.random.random((1,1))
        #list to keep track of error history for the graph
        self.error_history = []
        #list to keep track of epoch for the graph
        self.epoch_list = []
    
    
    def train(self,inputs,outputs,epochs=1000):
        '''
        inputs: array of variables, array of the outputs that match the inputs
        
        this function controls the training process of the neural network.
        the functions calls the feed forward and back propagation functions
        and passes them the correct variables
        '''
        for epoch in range(epochs):
            #forward pass
            self.hidden=self.feed_forward(inputs, self.weights_i_to_h,self.bias_i_to_h)
            #result of the forward pass:
            output=self.feed_forward(self.hidden,self.weights_h_to_o,self.bias_h_to_o)
            
            #back propagation
            self.backpropagation(inputs,outputs, output)
            self.epoch_list.append(epoch) #keep track of epochs
            
            
        
    
    def feed_forward(self,inputs_forward,weights_forward,bias_forward):
        '''
        inputs:array of variables, array of weights, array of bias
        output: array of values after doing a forward pass in one layer:
                the variables array is multiplyed by the weights array (dot product)
                then the bias is added
                the result is passed through the sidmoid activation function
        '''
        return (sigmoid(np.dot(inputs_forward, weights_forward)+bias_forward))
     
        
    
        
    def backpropagation(self,inputs,outputs,pred_output,lr=0.1):
        '''
        inputs:array of variables, array of outputs that match the inputs,
                array of outputs that the neural network predicted
        this function uses gradient descent in order to minimize the error 
        the function updates the weights and the biases to minimize the error 
        '''
        #updating the hidden-to-output-layer weights and bias (weights_h_to_o and bias_h_to_o)
        output_layer_error  = outputs - pred_output #output error        
        self.error_history.append(((output_layer_error) ** 2).mean())#keep track of error
        #calculating the impact of each weight on the error(delta) by using the chain rule and derivatives
        output_layer_delta = output_layer_error * sigmoid_derivative(pred_output) #hidden-to-output-layer weights delta 
        self.weights_h_to_o += lr * np.dot(self.hidden.T, output_layer_delta)#updating weights
       #output layer bias
        output_layer_bias_delta=np.sum(output_layer_delta,axis=0,keepdims=True)#output layer bias delta
        self.bias_h_to_o+=lr*output_layer_bias_delta#updating bias
        
                          
        #updating the inputs-to-hidden-layer weights and bias (weights_i_to_h and bias_i_to_h)
        hidden_layer_error=np.dot(output_layer_delta ,self.weights_h_to_o.T)#hidden layer error  
         #calculating the impact of each weight on the error(delta) by using the chain rule and derivatives
        hidden_layer_delta= hidden_layer_error * sigmoid_derivative(self.hidden)#inputs-to-hidden-layer weights delta
        self.weights_i_to_h+=lr * np.dot(inputs.T,hidden_layer_delta)#updating weights
        #hidden layer bias
        hidden_layer_bias_delta=np.sum(hidden_layer_delta,axis=0,keepdims=True)#hidden layer bias delta
        self.bias_i_to_h+=lr*hidden_layer_bias_delta#updating bias
        
    
    def test(self,test_inputs):
        '''
        inputs:list of variables
        output: list with one variable that represents male or female
                the variable is between 0 and 1 (can't be 0 or 1)
                0 is male
                1 is female
                
        '''
        result=self.feed_forward(test_inputs,self.weights_i_to_h,self.bias_i_to_h)
        return self.feed_forward(result,self.weights_h_to_o,self.bias_h_to_o)
    




def main():
    
    #create neural network
    NN = NeuralNetwork(inputs, outputs)
    # train neural network
    NN.train(inputs,outputs)
    
    #print(NN.test([1,10]))
    
    
    #making the loss/epoch graph
    plt.figure(figsize=(15,5))
    plt.plot(NN.epoch_list, NN.error_history)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show() 


    
    
    
    
if __name__ == '__main__':
    main()