import numpy as np
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from scipy.stats import bernoulli
from sklearn.linear_model import Lasso, LassoCV

class GL(object):
    def __init__(self, p , delta=0, prob=0.1):
        self.p = p
        self._delta = delta
        self.prob = prob
    
    @property
    def delta(self):
        return self._delta
    @delta.setter
    def delta(self, value):
        if value<0:
            raise ValueError
        self._delta = value
        
    def Matrix_generation(self, p=None, delta=None, prob=None):
        if p is None:
            p = self.p
        if delta is None:
            delta = self.delta
        if prob is None:
            prob = self.prob
            
        a = np.random.binomial(1,prob, (p,p))
        b = ((a+a.T)/2)
        c = np.where(b>0.5, 0.5, b)
        I = np.eye(p) * delta
        Theta = c + I
        sd = np.sqrt(np.diag(Theta))
        sd_mat = np.outer(sd, sd)
        Theta_cor = Theta/sd_mat
        eigenvalues = np.linalg.eig(Theta_cor)[0]
        if all(eigenvalues>0):
            return Theta_cor
        return self.Matrix_generation(p, delta+0.1)
    
class Graph_Lasso(object):
    def __init__(self, X, lambd=0):
        self._X = X
        self._lambd = lambd
    
    def NWL(self, lambd=None):
        if lambd is None:
            lambd = self._lambd
        P = self._X.shape[1]
        matrix = np.eye(P)
        for i in np.arange(P):
            obj = Lasso(lambd, fit_intercept=False)
            y = self._X[:,i]
            x = np.concatenate((self._X[:,:i], self._X[:,i+1:]), axis=1)
            obj.fit(x, y)
            coef = np.insert(obj.coef_, i, 1)
            matrix[i] = coef
        return matrix
    
    def CV(self, X, Y, fold, lambd, optimizer):
        x, y = np.array_split(X, fold), np.array_split(Y, fold) 
        error = 0. 
        for i in range(fold):
            x_ = x.copy()
            y_ = y.copy()
            x_val = x_[i] 
            y_val = y_[i]
            x_.pop(i)
            y_.pop(i)

            x_train = np.concatenate(x_) if self.folds !=1 else x_val
            y_train = np.concatenate(y_) if self.folds !=1 else y_val 

            obj = optimizer(lambd)
            obj.fit(x_train, y_train)
            pred = obj.predict(x_val)
            error += np.sum(np.square(pred-y_val))
        return error / fold
            
    def NWLCV_diy(self, lambds):
        P = self._X.shape[1]
        matrix = np.eye(P)
        for i in np.arange(P):
            b_cverror = float("+Inf")
            b_lambd = 0
            y = self._X[:,i]
            x = np.concatenate((self._X[:,:i], self._X[:,i+1:]), axis=1)
            for lambd in lambds:
                cverror = self.CV(x,y,10,lambd, Lasso)
                if cverror < b_cverror:
                    b_lambd = lambd
                    
            obj = Lasso(b_lambd, fit_intercept=False)   
            obj.fit(x, y)
            coef = np.insert(obj.coef_, i, 1)
            matrix[i] = coef
        return matrix
            
    def NWLCV(self):
        P = self._X.shape[1]
        matrix = np.eye(P)
        for i in np.arange(P):
            obj = LassoCV(fit_intercept=False)
            y = self._X[:,i]
            x = np.concatenate((self._X[:,:i], self._X[:,i+1:]), axis=1)
            obj.fit(x, y)
            coef = np.insert(obj.coef_, i, 1)
            matrix[i] = coef
        return matrix
    
    def GRL(self, max_iter=300,lambd=None):
        if lambd is None:
            lambd = self._lambd
        mle = GraphicalLasso(lambd, max_iter=max_iter).fit(self._X)
        return mle.precision_
        
    def GRLCV(self):
        mle = GraphicalLassoCV().fit(self._X)
        return mle.precision_
        
def X_gen(Cov, size=200, random_state=0):
    p = Cov.shape[0]
    np.random.seed(random_state)
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=np.linalg.pinv(Cov), size=size)
    return X
    
def metric(A, B, mode=None): #order matters! A should be true matrix, B is the predictted one.
    metric = {}
    a = A.copy()
    b = B.copy()
    p = np.diag(a).size
    
    #a[np.abs(a)<0.05]=0
    a[a!=0]=1
    #b[np.abs(b)<0.05]=0
    b[b!=0]=1
    
    if mode == "joint":
        b = (b + b.T) / 2
        b[b!=1] = 0
    if mode == "or":
        b = (b + b.T) / 2
        b[b==0.5] = 1
    
    FP = (b-a==1).sum() / 2
    FN = (a-b==1).sum() / 2
    TN = (np.where(a==b, a, -1)==0).sum() / 2
    TP = ((a==b).sum() - p) / 2 - TN
    n_pp = ((b==1).sum() - p) / 2
    n_np = ((b==0).sum()) / 2
    n_pt = ((a==1).sum() - p) / 2
    n_nt = ((a==0).sum()) / 2
    metric["FPR"] = FP/n_nt
    metric["TPR"] = TP/n_pt
    metric["F1"] = 2.*TP/(2.*TP+FP+FN)
    metric["area"] = (TP/n_pt) * (1-FP/n_nt)
    metric["FP"] = FP
    metric["TP"] = TP
    metric["FN"] = FN
    metric["TN"] = TN
    if n_pp:
        metric["precision"] = TP/n_pp
        metric["FDR"] = 1. - TP/n_pp
    return metric

def curve_auc(Yaxis, Xaxis, mode="ROC"):
    Y = Yaxis.copy()
    X = Xaxis.copy()
    if mode == "ROC":
        Y.insert(0, 1)
        X.insert(0, 1)
        Y.append(0)
        X.append(0)
    elif mode == "PR":
        Y.insert(0, 0)
        X.insert(0, 1)
        Y.append(1)
        X.append(0)
    yaxis = np.insert(Y, np.arange(1,len(Y)+1), np.roll(Y,-1))
    yaxis[-1] = yaxis[-2]
    xaxis = np.insert(X, np.arange(len(X)), X)
    AUC = (np.abs(np.diff(xaxis)) * yaxis[:len(yaxis)-1]).sum()
    
    return yaxis,xaxis,AUC

def sample_cov(X):
    Xb = np.mean(X, axis=0)
    n = X.shape[0]
    p = X.shape[1]
    matrix = np.zeros((p,p))
    for i in range(n):
        matrix += np.outer(X[i]-Xb, X[i]-Xb)
    return matrix / n

def BIC(X, lambd):
    scov = sample_cov(X)
    n = X.shape[0]
    lmap = {"GRL":[0, float("+Inf")], "NWL1":[0, float("+Inf")], "NWL2":[0, float("+Inf")]}
    
    gl = Graph_Lasso(X)
    for i in lambd:
        inv_covn = gl.NWL(lambd=i)
        inv_covg = gl.GRL(lambd=i)

        a = inv_covn.copy()
        a[a!=0]=1
        a1 = (a + a.T) / 2 # joint
        a1[a1!=1] = 0
        a2 = (a + a.T) / 2 # or
        a2[a2==0.5] = 1

        BIC_g = -n * np.log(np.linalg.det(inv_covg)) + n * np.trace(np.dot(inv_covg, scov)) \
        + np.log(n) * (np.triu(inv_covg, 1)!=0).sum()

        BIC_n1 = -n * np.log(np.linalg.det(a1)) + n * np.trace(np.dot(a1, scov)) \
        + np.log(n) * (np.triu(a1, 1)!=0).sum()

        BIC_n2 = -n * np.log(np.linalg.det(a2)) + n * np.trace(np.dot(a2, scov)) \
        + np.log(n) * (np.triu(a2, 1)!=0).sum()

        if BIC_g < lmap["GRL"][1]:
            lmap["GRL"][0] = i
            lmap["GRL"][1] = BIC_g
        if BIC_n1 < lmap["NWL1"][1]:
            lmap["NWL1"][0] = i
            lmap["NWL1"][1] = BIC_n1
        if BIC_n2 < lmap["NWL2"][1]:
            lmap["NWL2"][0] = i
            lmap["NWL2"][1] = BIC_n2
            
    l_GRL = lmap["GRL"][0]
    l_NWL1 = lmap["NWL1"][0]
    l_NWL2 = lmap["NWL2"][0]
    
    mp = {"matrix":[gl.GRL(lambd=l_GRL), gl.NWL(lambd=l_NWL1), gl.NWL(lambd=l_NWL2)], "lambda":lmap}
    return mp








