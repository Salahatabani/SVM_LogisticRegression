import numpy as np

def hinge(w,xTr,yTr,lmbda):
    """
    INPUT:
    w     : d   dimensional weight vector
    xTr   : nxd dimensional matrix (each row is an input vector)
    yTr   : n   dimensional vector (each entry is a label)
    lmbda : regression constant (scalar)
    
    OUTPUTS:
    loss     : the total loss obtained with w on xTr and yTr (scalar)
    gradient : d dimensional gradient at w
    """
    
    n, d = xTr.shape
    v = 1-np.dot(xTr,w)*yTr
    loss = np.sum(v[v>0])+lmbda*np.dot(w.T,w)
    gradient = -np.sum(np.array(v>0,dtype=int).reshape(n,1)*(yTr.reshape(n,1)*xTr),axis=0)+2*lmbda*w
    return loss, gradient

def linclassify(w,xTr):
    w = w.reshape(-1)
    preds = np.array(np.sign(np.inner(w,xTr)),dtype=int)
    return preds