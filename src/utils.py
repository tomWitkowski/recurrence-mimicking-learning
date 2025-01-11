import numpy as np
import pandas as pd
import sys, os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from config import Config as cfg

# @njit
def fast_rate_of_return(df_bad_values_list,
            lewar = cfg.lewar,
            wallet = 1,
            multiplier = 0,
            position = 0,
            opened_for = 0,
            bid = None,
            ask = None,
            bid_open = 0,
            ask_open = 0
           ):

    df_bad_values_list += [df_bad_values_list[-1][:2]+[0]]
    
    for bid,ask,d in iter(df_bad_values_list):
        if d != position:
            if position == 1 and d == -1:
                opened_for = 0

                if position == 1:
                    multiplier = (bid - ask_open)/ask_open
                elif position == -1:
                    multiplier = (bid_open - ask)/bid_open

                wallet *= (multiplier*lewar+1)

                position = 0        
                bid_open, ask_open = 0, 0


                bid_open, ask_open = bid, ask
                position = -1

            elif position == -1 and d == 1:

                opened_for = 0

                if position == 1:
                    multiplier = (bid - ask_open)/ask_open
                elif position == -1:
                    multiplier = (bid_open - ask)/bid_open

                wallet *= (multiplier*lewar+1)

                position = 0        
                bid_open, ask_open = 0, 0

                bid_open, ask_open = bid, ask
                position = 1
                
                
            elif d == 0:
                opened_for = 0
                bid, ask = bid, ask
                
                if position == 1:
                    multiplier = (bid - ask_open)/ask_open
                elif position == -1:
                    multiplier = (bid_open - ask)/bid_open
                
                wallet *= (multiplier*lewar+1)
                position = 0        
                bid_open, ask_open = 0, 0
                
            elif d == 1:
                bid_open, ask_open = bid, ask
                position = 1
                
            elif d == -1:
                bid_open, ask_open = bid, ask
                position = -1

        elif bid is not None:
            if position != 0:
                opened_for += 1/(60*24)
                
            if position == 1:
                multiplier = (bid - ask_open)/ask_open
            elif position == -1:
                multiplier = (bid_open - ask)/bid_open

        if wallet < 0.00001:
            return 0
            
    return wallet


class Line(object):

    def __init__(self,coor1,coor2):
        self.coor1 = coor1
        self.coor2 = coor2
    
    @property
    def slope(self):
        x2,y2 = self.coor2
        x1,y1 = self.coor1
        return (float(y2-y1))/(x2-x1)
    
    @property
    def inter(self):
        x2,y2 = self.coor1
        x1,y1 = self.coor2
        return y1 - x1*self.slope
    
    def predict(self, X):
        if isinstance(X, list):
            X = np.array(X)
        return (X*self.slope + self.inter).tolist()
    
   
    
class LightStrategy:
    
    __slots__ = ['lewar','wallet','multiplier','opened_for','position','bid','ask','bid_open','ask_open','result']
    
    def __init__(self):
        """
        Position in [-1,0,1]
            > -1 - position sold 
            >  0 - none position
            >  1 - position buy
        """
        
        self.lewar = cfg.lewar
        self.wallet = 1
        self.multiplier = 0
        self.position = 0
        self.opened_for = 0
        self.result = []
        self.bid = None
        self.ask = None
        self.bid_open = 0
        self.ask_open = 0
    
    def calc_multi(self):
        if self.position == 1:
            self.multiplier = (self.bid - self.ask_open)/self.ask_open
        elif self.position == -1:
            self.multiplier = (self.bid_open - self.ask)/self.bid_open
    
    def update_wallet(self):
        self.calc_multi()
        self.wallet *= (self.multiplier*self.lewar+1)
    
    def close_position(self, bid, ask):
        self.opened_for = 0
        self.bid, self.ask = bid, ask
        self.update_wallet()   
        self.position = 0        
        self.bid_open, self.ask_open = 0, 0
        
    def buy(self, bid, ask):
        self.bid_open, self.ask_open = bid, ask
        self.position = 1
        
        
    def sell(self, bid, ask):
        self.bid_open, self.ask_open = bid, ask
        self.position = -1
                    
    def make_decision(self, d, bid, ask):
        if d != self.position:
            if self.position == 1 and d == -1:
                self.close_position(bid, ask)
                self.sell(bid, ask)
            elif self.position == -1 and d == 1:
                self.close_position(bid, ask)
                self.buy(bid, ask)
            elif d == 0:
                self.close_position(bid, ask)
            elif d == 1:
                self.buy(bid, ask)
            elif d == -1:
                self.sell(bid, ask)
            else:
                raise ValueError(f'Co do kurwy? d = {d} and position = {self.position} (make_decision)')
        elif self.bid is not None:
            if self.position != 0:
                self.opened_for += 1/(60*24)
            self.calc_multi()

    def evaluate(self, df: object, reset_wallet: bool = True, collect_result: bool = False):
        # reset wallet
        if reset_wallet:
            self.wallet = 1
            self.result = []

        # iterate fastly
        for b,a,d in iter(df[['bid','ask','d']].values.tolist()):
            self.make_decision(d,b,a)
            
            if self.wallet < 0.00001:
                return 0

            if collect_result:
                self.result.append(self.wallet)

        # close at the very end
        self.make_decision(0,b,a)
        self.result.append(self.wallet)

        return self.wallet