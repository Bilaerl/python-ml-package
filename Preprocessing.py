import numpy as np

def add_bias(X):
    np.array(X) # convert to numpy array
    bias = np.ones((len(X),1)) # create bias
    X = np.hstack([bias,X]) # add bias to data

    return X