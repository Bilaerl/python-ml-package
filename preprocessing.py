import numpy as np

def add_bias(X):
    np.array(X) # convert to numpy array
    X = X.reshape(len(X),-1) # reshape X for compatability

    bias = np.ones((len(X),1)) # create bias
    X = np.hstack((bias,X)) # add bias to data

    return X


class FeatureMapper:

    def __init__(self, degree):
        self.degree = degree
    

    def __repr__(self):
        return f"FeatureMapper(degree={self.degree})"
    

    def map(self, X1,X2):
        out = np.ones((len(X1),1))
        for i in range(1,self.degree+1):
            for j in range(i+1):
                temp = (X1**(i-j)) * (X2**j)
                temp = temp.reshape(len(temp),1)
                out = np.hstack((out,temp))
        return out


class Normalizer:

    def __repr__(self):
        return f"Normalizer()"
    
    def fit(self, X):
        X = np.array(X) # convert to numpy array
        X = X.reshape(len(X),-1) # reshape X for compatability

        self.mu = X.mean(axis=0) # mean by column
        self.sigma = X.std(axis=0) # standard deviation by column
        
        X = (X-self.mu)/self.sigma # normalize X
        return X # return normalized X
    

    def transform(self, X):
        X = np.array(X) # convert to numpy array
        X = X.reshape(len(X),-1) # reshape X for compatability

        X = (X-self.mu)/self.sigma # normalize X
        return X # return normalized X