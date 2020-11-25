from abc import ABC, abstractmethod
import numpy as np
import scipy.optimize as op


class Supervised(ABC):

    @abstractmethod
    def cost_func(self, theta, X,y): pass

    @abstractmethod
    def predict(self, X): pass


    def train(self, X,y, theta=None, optimizer='TNC'):
        """Train a model, fit the model to given data.

        Input:
            X (array_like): m examples having n features
            y (array_like): Labels for the examples in X
            theta (array_like): Gradients to start optimization from,
                default is an n size vector of zeros
            optimizer: Optimization algorithm to use, default is 'TNC'
        
        Output:
            None

        """
        # convert input to numpy array
        X, y  = np.array(X), np.array(y)

        if theta is None: # if theta is not given
            # set theta to zeros of shape nx1
            theta = np.zeros((X.shape[1],1))
        else:
            # convert to numpy array
            theta = np.array(theta)
        
        # optimize
        result = op.minimize(fun=self.cost_func, x0=theta, args=(X, y), method=optimizer, jac=True)

        # set optimal theta and cost
        self.theta, self.cost = result.x, result.fun




class LinearReg(Supervised):
    """Create a Linear Regression object

    Input:
        reg_term (int): A number by which the model is regularize to stop it from
            overfitting

            Note: Regularization has not been tried yet
    
    Output:
        None
    """

    def __init__ (self, reg_term=0):
        self.reg_term = reg_term
    

    def __repr__(self):
        return f"LinearReg(reg_term={self.reg_term})"


    def cost_func(self, theta, X,y):
        """Find the cost of a prediction and gradients to minimize that cost

        Input:
            theta (array_like): Current gradients to be used in making prediction
            X (array_like): m examples of n features on which the predictions will be made
            y (array_like): Labels for examples of X. These are the targets of the prediction
                and will be used to measure accuracy of predictions made

        Output:
            cost (float): Cost of the current prediction
            grad (array_like): Corrected gradients base on the cost of current prediction
        """
        m,n = X.shape
        # reshape for compatibility
        y, theta = y.reshape(m,1), theta.reshape(n,1)
        
        # predictions
        h = X.dot(theta)

        # cost
        cost = (1/(2*m)) * sum((h-y) ** 2)

        # regularizing cost
        cost_reg = (self.reg_term/(2*m)) * (theta[1:] ** 2).sum()
        cost += cost_reg

        # gradients
        grad = (1/m) * ((X.T).dot(h-y))

        # regularizing gradients
        grad_reg = np.zeros_like(grad)
        grad_reg[1:] = (self.reg_term/m) * theta[1:]
        grad += grad_reg

        return cost, grad


    def predict(self, X):
        """Predict X base on the models training

        Input:
            X (array_like): m examples of n features
        
        Output:
            Predictions (int|array_like): Value predicted  
        """
        # return prediction
        return X.dot(self.theta)


class LogisticReg(Supervised):
    """Create a Logistic Regression object

    Input:
        threshold (float): A number between 0 and 1 used as classification boundary
            above which everything is of that particular class (1), and below which
            everything is not of the class (0)
        reg_term (int): A number by which the model is regularize to stop it from
            overfitting
    
    Output:
        None
    """

    def __init__(self, threshold=0.5, reg_term=0):
        self.threshold = threshold
        self.reg_term = reg_term
    

    def __repr__(self):
        return f"LogisticReg(threshold={self.threshold},reg_term={self.reg_term})"
    

    @staticmethod
    def sigmoid(z):
        """Squashes any number to a number between 0 and 1

        Input:
            z (int|float|array_like): Number or series of numbers to be squashed
        
        Output:
            (float|array_like): Input squashed to number or series of numbers between 0 and 1
        """
        return 1/(1 + np.exp(-z))
    

    def cost_func(self, theta, X,y):
        """Find the cost of a prediction and gradients to minimize that cost

        Input:
            theta (array_like): Current gradients to be used in making prediction
            X (array_like): m examples of n features on which the predictions will be made
            y (array_like): Labels for examples of X. These are the targets of the prediction
                and will be used to measure accuracy of predictions made

        Output:
            cost (float): Cost of the current prediction
            grad (array_like): Corrected gradients base on the cost of current prediction
        """
        m,n = X.shape
        # reshape for compatibility
        y, theta = y.reshape(m,1), theta.reshape(n,1)
        
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
        grad = (1/m) * ((X.T).dot(h-y))

        # regularizing gradients
        grad_reg = np.zeros_like(grad)
        grad_reg[1:] = (self.reg_term/m) * theta[1:]
        grad += grad_reg

        return cost, grad


    def predict(self, X):
        """Predict X base on the models training

        Input:
            X (array_like): m examples of n features
        
        Output:
            Predictions (int|array_like): 0s and 1s showing if a given example is classified as
                member of a class (1) or not (0)
            
        """
        # raw prediction
        h = self.sigmoid(X.dot(self.theta))
        # compare raw prediction to threshold
        predictions = h > self.threshold

        # return predictions as int
        return predictions.astype('int')


class SVM:

    def __init__(self, kernel="linear", reg_term=0):
        self.reg_term = reg_term
        self.kernel = kernel
    

    def __repr__(self):
        return f"SVM(kernel={self.kernel}, reg_term={self.reg_term})"
    

    def linear_kernel(self):
        pass


    def guassian_kernel(self):
        pass


    def train(self, theta, X, y):
        pass


    def predict(self, X):
        pass