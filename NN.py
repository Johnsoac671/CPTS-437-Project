class Layer:
    def __init__(self, input_size, output_size, activation_function, learning_rate):
        
        self.weights = [[0.0 for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [0.0 for _ in range(output_size)]
        
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        
        self.z = []
        self.outputs = []
        self.errors = []
    
    
    def forward(self, inputs):
        self.outputs = []
        num_inputs = len(inputs)
        
        self.z = []
        for index in range(len(self.weights)):
            weight = self.weights[index]
            weighted_sum = sum([weight[x] * inputs[x] for x in range(num_inputs)])
            
            preactivation_value = weighted_sum + self.biases[index]
            self.z.append(preactivation_value)
            
            activation_output = self.activation_function(preactivation_value)
            self.outputs.append(activation_output)
        
        return self.outputs
    
    
    def derivative(self, value):
        return self.activation_function(value) * (1 - self.activation_function(value))
        
    
    def backward(self, inputs, target, next_layer=None):
        
        errors = []
        
        if not next_layer:
            for value in self.z:
                output = self.activation_function(value)
                error = (output - target) * self.derivative(value)
                errors.append(error)
        
        else:
            
            for index, value in enumerate(self.z):
                error = 0
                
                for next_index, weight in enumerate(next_layer.weights):
                    error += next_layer.errors[next_index] * weight[index]
                
                errors.append(error * self.derivative(value))
        
        
        self.errors = errors
        
        for weight_index, weight in self.weights:
            for input_index, input_value in inputs:
                weight[input_index] -= self.learning_rate * self.error[weight_index] * input_value
        
        
        for bias_index, bias in self.biases:
            bias -= self.learning_rate * self.errors[bias_index]
                
            

    
        

class Network:
    
    def __init__(self, layer_info):
        self.layers = [Layer(output_size) for output_size in layer_info]

    
    
    def train(self):
        pass
    
    
    def predict(self):
        pass