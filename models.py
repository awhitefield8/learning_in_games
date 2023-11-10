import numpy as np

class Mw:
    """
    multiplicative weights function
    """
    def __init__(self, n, nu):
        """
        Args,
            n (int): number of experts
            nu (float): learning rate
        """
        self.n = n
        self.nu = nu
        self.weights = np.full(n, 1/n)
    
    def update(self, c):
        """
        Args,
            c (numpy array): cost vector
        """
        self.weights *= 1 - self.nu * c
        self.weights /= np.sum(self.weights)


class Firm:
    def __init__(self,id,beta):
        self.id = id
        self.prices = []
        self.shares = [] #market share
        self.beta = beta

    def setPrice(self,price_ext=None):
        if price_ext is None:
            price = 1
        else:
            price = price_ext
        self.prices.append(price)
        return(price)

    def updateShares(self,share):
        self.shares.append(share)

    def dpv(self):
        DPV = 0
        for i in range(len(self.prices)):
            DPV += (self.beta**i)*(self.prices[i])*(self.shares[i])
        return(DPV)





