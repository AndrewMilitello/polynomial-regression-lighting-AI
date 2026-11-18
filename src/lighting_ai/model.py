import numpy as np
from lighting_ai.config import *
class Linear():
    def __init__(self, fan_in, fan_out, lr=0.01, is_output=False, activation=None):
        limit = np.sqrt(6 / (fan_in + fan_out))
        self.weights = np.random.uniform(-limit, limit, (fan_in, fan_out))
        self.bias = np.full((1, fan_out), 0.1)
        self.vw = np.zeros_like(self.weights)
        self.vb = np.zeros_like(self.bias)

    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.bias
        return self.output
#init neur net architecture

class Network():
    def __init__(self, lr=0.01, ptcount=bsize):
        self.lr = lr
        self.input_count = (sample_size ** 2) * 3 + 3
        self.hidden_layer = Linear(fan_in=self.input_count, fan_out=hidden_layer_1_size, lr=lr)
        self.hidden_layer_2 = Linear(fan_in=hidden_layer_1_size, fan_out=hidden_layer_2_size, lr=lr)
        self.output_layer = Linear(fan_in=hidden_layer_2_size, fan_out=3 * output_degree, lr=lr)
    def leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.01)
    def __call__(self, input):
        
        self.last_input = input
        h1 = self.hidden_layer.forward(input)
        self.hidden_output = self.leaky_relu(h1)
        h2 = self.hidden_layer_2.forward(self.hidden_output)
        self.hidden2_output = self.leaky_relu(h2)
        self.final_output = self.output_layer.forward(self.hidden2_output)
        return self.final_output
    
    def parameters(self):
        return [
            (self.hidden_layer.weights, self.hidden_layer.bias),
            (self.hidden_layer_2.weights, self.hidden_layer_2.bias),
            (self.output_layer.weights, self.output_layer.bias)
        ]
