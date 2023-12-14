import numpy as np

class Mw:
    """
    multiplicative weights object
    """
    def __init__(self, n=2, nu=0.2,weights=None):
        """
        Args,
            n (int): number of experts
            nu (float): learning rate
            weights (np.array): array of weights
        """
        self.n = n
        self.nu = nu
        if weights is None:
            self.weights = np.full(n, 1/n)
        else:
            self.weights = weights
    
    def update(self, c):
        """
        Args,
            c (list): cost vector
        """
        cost_array = np.array(c)
        self.weights *= 1 - self.nu * cost_array
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





