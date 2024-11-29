class Layer:
    def __init__(self, input_size, output_size, activation_function, learning_rate):
        
        self.weights = [[0.0 for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [0.0 for _ in range(output_size)]
        
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        
        self.z = []
    
    
    def forward(self, inputs):
        outputs = []
        num_inputs = len(inputs)
        
        self.z = []
        for index in range(len(self.weights)):
            weight = self.weights[index]
            weighted_sum = sum([weight[x] * inputs[x] for x in range(num_inputs)])
            
            preactivation_value = weighted_sum + self.biases[index]
            self.z.append(preactivation_value)
            
            activation_output = self.activation_function(preactivation_value)
            outputs.append(activation_output)
        
        return outputs
    
    
    def derivative(self, value):
        return self.activation_function(value) * (1 - self.activation_function(value))
        
    
    def backward(self, inputs, target, next_layer=None):
        
        for index, value in enumerate(inputs):
            derivative_z = value * self.derivative(self.z[index])

    
        

class Network:
    
    def __init__(self, layer_info):
        self.layers = [Layer(output_size) for output_size in layer_info]

    
    
    def train(self):
        pass
    
    
    def predict(self):
        pass