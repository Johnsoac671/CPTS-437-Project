class Layer:
    def __init__(self, input_size, output_size):
        
        self.weights = [[0.0 for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [0.0 for _ in range(output_size)]
    
    def forward(self):
        pass
    
    def backward(self):
        pass
        
    
        

class Network:
    
    def __init__(self, layer_info):
        self.layers = [Layer(output_size) for output_size in layer_info]

    
    
    def train(self):
        pass
    
    
    def predict(self):
        pass