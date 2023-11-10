class Firms:
    """
    a firm that will set prices
    """
    def __init__(self,n):
        """
        Args,
            n (int): number of firms
        """
        self.n = n
        self.profit=0
    
    def prices(self):
        prices = np.full(self.n,1)
        return(prices)


    def updateProfit(self,flowprofit):
        """
        Args,
            flowprofit (numpy array): vector of flow profits
        """
        self.profit += flowprofit

#class Firms:
