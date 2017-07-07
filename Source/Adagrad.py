import numpy as np

def adagrad(func,w,alpha,maxiter,delta=1e-02):
    """
    INPUT:
    func    : function to minimize
              (loss, gradient = func(w))
    w       : d dimensional initial weight vector 
    alpha   : initial gradient descent stepsize (scalar)
    maxiter : maximum amount of iterations (scalar)
    delta   : if norm(gradient)<delta, it quits (scalar)
    
    OUTPUTS:
     
    w      : d dimensional final weight vector
    losses : vector containing loss at each iteration
    """
    
    losses = np.zeros(maxiter)
    eps = 1e-06
    
    
    # initialize loss and gradient from func()
    loss, gradient = func(w)
    
    # intialize z vector
    z = np.zeros(w.shape)
    
    # adagrad updating w
    for i in range(maxiter):
        loss, gradient = func(w)
        #z = z + np.square(gradient)
        z = z + gradient*gradient
        update = (alpha*gradient)/(np.sqrt(z+eps))
        w = w - update
        losses[i] = loss
        if np.linalg.norm(gradient) < delta:
            print ('total interation: %s' %i)
            print ('final loss: %s' %loss)
            break
    print ('total interation: %s' %i)
    return w, losses