import numpy as np
import scipy.optimize as op

class LinearReg():

    def __init__ (self):
        pass


    def train(self, X,y):
        pass


    def predict(self, X,y):
        pass


class LogisticReg():

    def __init__(self, threshold=0.5, reg_term=0):
        self.threshold = threshold
        self.reg_term = reg_term
    

    def __repr__(self):
        return f"LogisticReg(threshold={self.threshold},reg_term={self.reg_term})"
    

    @staticmethod
    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    

    def cost_func(self, theta, X,y):
        m = len(y)
        
        # predictions
        h = self.sigmoid(X.dot(theta))

        # cost
        term_1 = ((-y).T).dot(np.log(h))
        term_2 = ((1-y).T).dot(np.log(1-h))
        cost = (1/m) * (term_1 - term_2)

        # regularizing cost
        cost_reg = (self.reg_term/(2*m)) * (theta[1:] ** 2).sum()
        cost += cost_reg

        # gradients
        grad = (1/m) * ( (X.T).dot(h - y) )

        # regularizing gradients
        grad_reg = np.zeros_like(grad)
        grad_reg[1:] = (self.reg_term/m) * theta[1:,:]
        grad += grad_reg

        return cost, grad


    def train(self, X,y, theta=None):
        # convert input to numpy array
        X, y  = np.array(X), np.array(y)

        m, n = X.shape
        y = y.reshape(m,1) # reshape y for compatibility

        if theta is None: # if theta is not given
            # set theta to zeros
            theta = np.zeros((n,1))
        else:
            theta = np.array(theta) # convert to numpy array
            theta = theta.reshape(n,1) # reshape for compatability
        
        # optimize
        result = op.minimize(fun=self.cost_func, x0=theta, args=(X, y), method='TNC', jac=True)

        # set optimal theta and cost
        self.theta, self.cost = result.x, result.fun


    def predict(self, X):
        # raw prediction
        h = self.sigmoid(X.dot(self.theta))
        # compare raw prediction to threshold
        predictions = h > self.threshold

        # return predictions as int
        return predictions.astype('int')