import numpy as np

def zca_process(train_X):
    X_mean, W = zca_whiten(train_X.values)
    return pd.DataFrame(data=np.dot(train_X.values-X_mean,W),columns=train_X.columns)

def zca_whiten(X):
    """
    Applies ZCA whitening to the data (X) Data ,
    
    X^T * X = CovX = U *d* U^T where the columns of E are the normalized eigenvectors
    D=d^(-1/2), the whitening matrix is W = E * D * E^T: Xw = X*W
    Cov Xw = Xw^T*Xw = E*D*E^T*X^T*X*E*D*E^T = E*D*E^T*CovX*E*D*E^T = E*D*E^T*E *d* E^T*E*D*E^T = I
    The inverse is W^-1: W^-1 = E * D^(-1) * E^T
    The un-whitened matrix is X = (X*W)*W^-1

    Input:
    -X (numpy array): input data, rows are data points, columns are features
    
    Output:
    -X_mean (array): mean of X
    -W (matrix): dewhitening matrix
    """

    EPS = 10e-18
    upperBound=10e14
    meanX = np.mean(X, axis=0)
    Xc = X - meanX

    cov = np.dot(Xc.T, X)/float(Xc.shape[0]-1.0)

    U,d, _ = np.linalg.svd(cov)
    d = np.sqrt(np.clip(a=d, a_min=EPS, a_max=upperBound))
    D_whiten = np.diag(1. / d)
    D_dewhiten = np.diag(d)    
    W = np.dot(np.dot(U, D_whiten), U.T)
    W_dewhiten = np.dot(np.dot(U, D_dewhiten), U.T)
    X_white = np.dot(Xc, W)

    # test
    test1 = np.std(X_white, axis=0)
    test2 = np.add(np.dot(X_white, W_dewhiten),meanX)
    #print("white reconstruction error: "+str(np.max( X-test2 )))
    #print("white standardization error: " + str(np.max(test1 - np.ones(test1.shape))))
    
    return meanX, W
    
def scale_to_0_1(volatility_model_parameter):    
    """Scale parameter from model parameter bounderies to [0, 1].
    
    Input:
    -volatility_model_parameter_recaled (dict): parameter on [0, 1] interval
    -id_volatility_model (string): id of volatility model
    -volatility_model_parameter (dict): parameter in model bounderies
    
    Output:
    -volatility_model_def (dict): number of train-test datasets
    """
    parameter_lower_bounds = [0.0001,-0.95,0.01,0.01,1]
    parameter_upper_bounds= [0.04,-0.1,1.0,0.2,10.0]
    
    parameter_names= ['v0', 'rho', 'sigma', 'theta', 'kappa']
    volatility_model_parameter_recaled=volatility_model_parameter.copy()

    for i_parameter, parameter_name in enumerate(parameter_names):
        volatility_model_parameter_recaled[parameter_name]=(volatility_model_parameter[parameter_name]\
        -parameter_lower_bounds[i_parameter])/(parameter_upper_bounds[i_parameter]\
        -parameter_lower_bounds[i_parameter])
    return volatility_model_parameter_recaled.reset_index(drop=True)

class Scaler:
    def _init__(self,data):
        #self.fit(data)
        pass
    
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.var = X.std(axis=0)
    
    def transform(self, X):
        return (X - self.mean)/self.var
    
    def untransform(self, X):
        return (X + self.mean)*self.var