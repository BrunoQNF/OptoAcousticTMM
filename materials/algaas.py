class AlGaAsx:

    def __init__(self, x):
        self.n = 2.96*x + 3.54*(1-x)
        self.rho = 3770*x + 5350*(1-x)
        self.c = 5660*x + 4780*(1-x)
        if x > 0.5:
            self.name = "AlAs"
            self.color = "#91cbfa"
        
        else:
            self.name = "GaAs"
            self.color = "#aea6f7"
        self.Z = self.rho * self.c
