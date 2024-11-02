from processing import Vector, vectorize
import math

class KNN:
    
    def __init__(self, vectorset, k=5):
        self.vectorset = vectorset
        self.k = k
        
    
    def predict(self, data):
        vectorized_data = vectorize(data)
        
        distances = []
        
        for vector in self.vectorset:
            distance = self.get_distance(vector, vectorized_data)
            distances.append((distance, vector.label))
            
        distances.sort(key=lambda x: x[0])
        
        nearest = distances[:self.k]
        nearest_labels = [x[1] for x in nearest]
        
        prediction = round(sum(nearest_labels) / self.k)
        
        return prediction
    
    
    def get_distance(self, vector, data):
        
        dot_product = sum(
            [x[0] * x[1] for x in zip(vector, data)]
            )
        
        vec_mag = self.get_magnitude(vector)
        data_mag = self.get_magnitude(data)
        
        return dot_product / (vec_mag * data_mag)
        

    def get_magnitude(self, vector):
        return math.sqrt(
            sum(
                [x*x for x in vector]
                ))