import numpy as np

class SGD:
    """
    This can be used as mini-batch GD or pure stochastic gradient descent by changing the batch size
    For pure SGD, batch size = 1
    """
    def __init__(self, lr=0.01):
        self.lr =lr
    
    def update(self, model, dW_list, db_list):
        for i in range(len(model.layers)):
            W,b = model.layers[i]
            W-=self.lr*dW_list[i]
            b-=self.lr*db_list[i]
            model.layers[i] = (W,b)


class Momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.vW=[]
        self.vb=[]
        self.initialized = False

    def update(self, model, dW_list, db_list):
        if not self.initialized:
            for(W,b) in model.layers:
                self.vW.append(np.zeros_like(W))
                self.vb.append(np.zeros_like(b))
            self.initialized = True

        for i in range(len(model.layers)):
            W, b = model.layers[i]
            self.vW[i] = self.beta*self.vW[i] + self.lr*dW_list[i]
            self.vb[i] = self.beta*self.vb[i] + self.lr*db_list[i]

            W-=self.vW[i]
            b-=self.vb[i]
            model.layers[i]=(W,b)

class NAG:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.vW = []
        self.vb = []
        self.initialized = False
    
    def update(self, model, dW_list, db_list):
        if not self.initialized:
            for(W,b) in model.layers:
                self.vW.append(np.zeros_like(W))
                self.vb.append(np.zeros_like(b))
            self.initialized = True

        for i in range(len(model.layers)):
            W, b = model.layers[i]

            # Save Old velocity
            vW_prev = self.vW[i].copy()
            vb_prev = self.vb[i].copy()

            # Update velocity
            self.vW[i] = self.beta*self.vW[i] + self.lr*dW_list[i]
            self.vb[i] = self.beta*self.vb[i] + self.lr*db_list[i]

            # The Nesterov Lookahead step
            W-=(self.beta*vW_prev + (1+self.beta)*self.vW[i])
            b-=(self.beta*vb_prev + (1+self.beta)*self.vb[i])

            model.layers[i] = (W,b)

class RMSProp:
    # Initial lr is a parmeter and by default and is kept at 0.001 as we saw in class
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.vW = []
        self.vb = []
        self.initialized = False

    def update(self, model, dW_list, db_list):
        if not self.initialized:
            for(W,b) in model.layers:
                self.vW.append(np.zeros_like(W))
                self.vb.append(np.zeros_like(b))
            self.initialized = True
        
        for i in range(len(model.layers)):
            W, b = model.layers[i]

            # Updating the running average of the squared grasients
            self.vW[i] = self.beta*self.vW[i] + (1-self.beta)*(dW_list[i]**2)
            self.vb[i] = self.beta*self.vb[i] + (1-self.beta)*(db_list[i]**2)

            W-=self.lr*dW_list[i]/(np.sqrt(self.vW[i]) + self.eps)
            b-=self.lr*db_list[i]/(np.sqrt(self.vb[i]) + self.eps)

            model.layers[i] = (W, b)

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mW = []
        self.vW = []
        self.mb = []
        self.vb = []
        self.t = 0
        self.initialized = False

    def update(self, model, dW_list, db_list):
        if not self.initialized:
            for(W,b) in model.layers:
                self.mW.append(np.zeros_like(W))
                self.vW.append(np.zeros_like(W))
                self.mb.append(np.zeros_like(b))
                self.vb.append(np.zeros_like(b))
            self.initialized = True

        self.t+=1 # Time step
        
        for i in range(len(model.layers)):
            W, b = model.layers[i]

            # Update biased first moment estimate
            self.mW[i] = self.beta1*self.mW[i] + (1-self.beta1)*dW_list[i]
            self.mb[i] = self.beta1*self.mb[i] + (1-self.beta1)*db_list[i]

            # Update biased second moment estimate
            self.vW[i] = self.beta2*self.vW[i] + (1-self.beta2)*(dW_list[i]**2)
            self.vb[i] = self.beta2*self.vb[i] + (1-self.beta2)*(db_list[i]**2)

            # Bias-corrected first moment
            mW_hat = self.mW[i]/(1-self.beta1**self.t)
            mb_hat = self.mb[i]/(1-self.beta1**self.t)

            # Bias-corrected second raw moment
            vW_hat = self.vW[i]/(1-self.beta2**self.t)
            vb_hat = self.vb[i]/(1-self.beta2**self.t)

            # Updating parameters
            W-=self.lr*mW_hat/(np.sqrt(vW_hat) + self.eps)
            b-=self.lr*mb_hat/(np.sqrt(vb_hat) + self.eps)

            model.layers[i] = (W, b)


class Nadam:
    """
    Nadam = Adam + Nesterov
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mW = []
        self.vW = []
        self.mb = []
        self.vb = []
        self.t = 0
        self.initialized = False       

    def update(self, model, dW_list, db_list):
        if not self.initialized:
            for(W,b) in model.layers:
                self.mW.append(np.zeros_like(W))
                self.vW.append(np.zeros_like(W))
                self.mb.append(np.zeros_like(b))
                self.vb.append(np.zeros_like(b))
            self.initialized = True

        self.t+=1 # Time step
        alpha_t = self.lr*np.sqrt(1 - self.beta2**self.t)/(1-self.beta1**self.t)

        for i in range(len(model.layers)):
            W, b = model.layers[i]

            # Update biased first moment estimate
            self.mW[i] = self.beta1*self.mW[i] + (1-self.beta1)*dW_list[i]
            self.mb[i] = self.beta1*self.mb[i] + (1-self.beta1)*db_list[i]

            # Update biased second moment estimate
            self.vW[i] = self.beta2*self.vW[i] + (1-self.beta2)*(dW_list[i]**2)
            self.vb[i] = self.beta2*self.vb[i] + (1-self.beta2)*(db_list[i]**2)

            # The Nesterov Lookahead step for the first moment
            mW_hat = (self.beta1*self.mW[i]/(1-self.beta1**self.t)) + ((1-self.beta1)*dW_list[i]/(1-self.beta1**self.t))
            mb_hat = (self.beta1*self.mb[i]/(1-self.beta1**self.t)) + ((1-self.beta1)*db_list[i]/(1-self.beta1**self.t))

            # Correcting the second moment
            vW_hat = self.vW[i]/(1-self.beta2**self.t)
            vb_hat = self.vb[i]/(1-self.beta2**self.t)

            # Updating parameters
            W-=alpha_t*mW_hat/(np.sqrt(vW_hat) + self.eps)
            b-=alpha_t*mb_hat/(np.sqrt(vb_hat) + self.eps)

            model.layers[i] = (W, b)
