import numpy as np

def logistic(w,xTr,yTr,lmbda):
    """
    INPUT:
    w     : d   dimensional weight vector
    xTr   : nxd dimensional matrix (each row is an input vector)
    yTr   : n   dimensional vector (each entry is a label)
    
    OUTPUTS:
    loss     : the total loss obtained with w on xTr and yTr (scalar)
    gradient : d dimensional gradient at w
    """
    n, d = xTr.shape
    
    loss = np.sum(np.log(1+(np.exp(np.dot(-xTr,w.T)*yTr))))+lmbda*np.dot(w.T,w)    
    gradient = -np.sum((yTr.reshape(n,1)*xTr)/(1+(np.exp(np.dot(xTr,w.T)*yTr))).reshape(n,1),axis=0)+2*lmbda*w
    
    return loss, gradient

def linclassify(w,xTr):
    w = w.reshape(-1)
    preds = np.array(np.sign(np.inner(w,xTr)),dtype=int)
    return preds