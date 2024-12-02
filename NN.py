import random

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

    
    
    def train(self, training_data, epochs):
        # For the specified epochs, run back propogation on the network
        for epoch in range(epochs):
            random.shuffle(training_data)
            for inputs, target, in training_data:
                
                # Forward Pass of the input
                result = self.predict(inputs)

                # Backward Pass of the input
                for i in range(len(self.layers) - 1, -1, -1):
                    next_layer = self.layers[i + 1] if i + 1 < len(self.layers) else None
                    self.layers[i].backward(inputs, target, next_layer)
    
    def predict(self, inputs):
        # Pass inputs forward through each layer
        for layer in self.layers:
            inputs = layer.forward(inputs)
        
        # Return the final product
        return inputs

