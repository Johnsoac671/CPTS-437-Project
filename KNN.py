from processing import Vector, vectorize

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