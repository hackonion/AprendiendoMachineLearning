
import numpy as np 

class Perceptron(object):
    """Percepton clasifier
    Parameters
    ----------
    eta: float
        Learning rate (between 0.0 and 1.0)
    n_iter: int 
        Passes over the training dataset.
    random_state: int
        Random number generator seed for randoms weight
        initialization


    Attributes
    -----------
    w_: id-array
        Weight after fittings
    errors_: list
        Numbers of misclasification in each epoch
    """
    #Constructor
    def __init__(self,eta=0.01,n_iter=50,random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self,X,Y):
        """Fit training data set

        Parameters:
            X ([array-like]), shape: [n_samples, n_feactures]
                Training vectors, where n_samples is the number of 
                smples and
                n_feactures is the number of features.
            Y ([array-like]), shape: [n_samples]
                Target values
            Returns
            -------
            self: object
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi in zip(X,Y):
                updates = self.eta * (target - self.predict(xi))
                self.w_[1:] += updates * xi
                self.w_[0] += updates
                errors += int(updates != 0.0)
                self.errors_.append(errors)
            return self
        
    def net_input(self,X):
        """Calculate net input
        """
        return np.dot(X,self.w_[1:]) + self.w_[0]
        
    def predict(self, X):            
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
        



    


