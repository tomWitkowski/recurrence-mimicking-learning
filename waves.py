import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import deque
import traceback
import tensorflow as tf
import math


class WaveGrasper:
    """
    
    >>>>>>> OLD VERSION, DEPRECATED <<<<<<<<
    
    A class to extract waves from Series
    
    Input:
    > series - price series
    > periods - many periods means big observations' window where local extremas are to be found in -> rather more long-term waves
    
    Returns:
    > a class with waves objects:
        > Series - a series containing waves' ends
        > Waves - a DataFrame, where each row is one wave 
        > ABCD, y - frames with ABCD formations and the price finishing 
        subsequent wave (scaled the best way, but in feed-forward networks 
        y should be divided by 10 in order to provide NN better data)
    
    Exemplary usage:
    
    lts = get_it('lts')          # or any another sequence of price
    wg = WaveGrasper(lts, 30)    # init and basically doing all the job
    wg.Waves                     # get waves Dataframe
    wg.Series                    # get sequence of waves
    wg.ABCD, wg.y                # exo and endo variables of waves formations? Go ahead!
    
    """
    @staticmethod
    def get_property(series, function, periods):
        return series.rolling(periods, center=False).agg(function).dropna()
    
    @staticmethod
    def unique(array):
        return list(set(array.tolist()))
    
    @staticmethod
    def get_values(series, inds, name=None):
        df_ = pd.DataFrame( series.iloc[inds].rename('price') )
        if name is not None:
            df_['name'] = name
        df_.index = inds
        return df_
    
    @staticmethod
    def reduce_extrema(df):
        looker = [0]
        k=0
        for i in range(1, len(df)):
            if df.name.values[i] != df.name.values[i-1]:
                k+=1
            looker.append(k)
        df['look'] = looker
        df['index'] = df.index

        df = pd.concat( [ df[df.name=='max'].loc[df[df.name=='max'].groupby('look')['price'].idxmax()],
                         df[df.name=='min'].loc[df[df.name=='min'].groupby('look')['price'].idxmin()]  ], axis = 0 )
        
        return df.set_index('index').sort_index()
    
    @staticmethod
    def extract_waves(df):
        start_price = []
        end_price = []
        start_ind = []
        end_ind = []
        for i in range(1,len(df)):
            start_price.append( df.price.values[i-1] )
            end_price.append( df.price.values[i] )
            start_ind.append( df.index.values[i-1] )
            end_ind.append( df.index.values[i] )
        return pd.DataFrame({'start_price':start_price ,
                            'end_price':end_price , 
                            'start_ind':start_ind ,
                            'end_ind':end_ind })
    
    @staticmethod
    def wave_min_max(x: list):
        min_ = min(x[:-1])
        max_ = max(x[:-1])
        return (x-min_)/(max_-min_), [min_,max_]
      
    @staticmethod
    def rescale(y, scalers):
      # (x-min_)/(max_-min_) reversal
        min_, max_ = np.array(scalers)[:,0].reshape(-1,1), np.array(scalers)[:,1].reshape(-1,1)
        y = y.reshape(-1,1)
        return y*(max_-min_)+min_
    
    
    def __init__(self, series, periods = 30, length = 6):
        self.minima_inds = self.unique( self.get_property(series, lambda x: x.idxmin(), periods).values )
        self.maxima_inds = self.unique( self.get_property(series, lambda x: x.idxmax(), periods).values )
        self.minima, self.maxima = self.get_values(series, self.minima_inds, 'min'), \
                                   self.get_values(series, self.maxima_inds, 'max')
        self.Series = pd.concat([self.minima, self.maxima],axis=0).sort_index()
        self.Series = self.reduce_extrema(self.Series)
        self.Waves = self.extract_waves(self.Series) # ind means indexes
        self.get_ABCD_y(length)
        
    def get_ABCD_y(self, length = 6):
        """ 
        A function which transforms waves sequence to ABCD formation and scales it 
        
        Input:
        > length - how many extremas including y
        """
        X = self.Series.price.values
        ABCD = []
        y = []
        scalers = []
        
        # get scaled waves
        for i in range(length, len(X)):
            wave = X[i-length:i]
            wave, extrema = self.wave_min_max(wave)
            scalers.append(extrema)
            ABCD.append( wave[:-1].tolist() )
            y.append( wave[-1] )
        
        self.ABCD = np.array(ABCD).reshape(-1, length-1)
        self.y = np.array(y).reshape(-1,)
        self.scalers = scalers
    
        return self.ABCD , self.y, self.scalers
    
    

class BallTurner:

    """
    Finds local extremas
    """
    
    def __init__(self, data, early_detector: object = None):
        self.EarlyDetector = early_detector
        self.data = data.reset_index(drop=True).dropna()
        self.pct_change = self.data.pct_change()
        self.df = pd.DataFrame({'price':self.data})

    def calc_change(self, ball):
        price_now = ball[1]
        price_ass = self.assumption[1]
        self.change = (price_now-price_ass)/price_ass
        
    
    def calc_change_ahead(self, ball):
        price_now = ball
        price_ass = self.assumption if isinstance(self.assumption, float) else self.assumption.price
        self.change = (price_now-price_ass)/price_ass
        
    def check_change(self):
        return abs(self.change) >= self.min_change

    def turn(self, min_change = 0.03, gather_point_data: bool = False):

        self.min_change = min_change
        self.gather = []
        self.assumption_dir = None
        self.start = None

        self.point_data = []
        
        for self.i, self.ball in enumerate(self.df.itertuples()):
            if self.ball[0]==1:
                self.assumption_dir='go_up' # be an optimist
                self.assumption = self.ball
                
            elif self.ball[0]>1:
            
                self.calc_change(self.ball)
                
                if self.assumption_dir == 'go_up':
                    if self.change>0:
                        if self.assumption[1] < self.ball[1]:
                            self.assumption = self.ball
                    elif self.change<0:
                        if self.check_change():
                            self.assumption_dir = 'go_down'
                            self.gather.append(self.assumption)
                            if self.assumption[1] > self.ball[1]:
                                self.assumption = self.ball
                if self.assumption_dir == 'go_down':
                    if self.change<0:
                        self.assumption = self.ball
                    elif self.change>0:
                        if self.check_change():
                            self.assumption_dir = 'go_up'
                            self.gather.append(self.assumption)
                            if self.assumption[1] < self.ball[1]:
                                self.assumption = self.ball
            
            if self.ball.Index > self.i:
                raise ValueError(self.ball.Index, self.assumption.Index, self.i)
            
            
            if gather_point_data == 'just_ai':
                if len(self.gather)>self._length:
                    self.point_data.append([self.i, (self.i-self.assumption.Index)*(1 if self.ball.price >= self.assumption.price else -1) ])
                   
            elif gather_point_data:
                if len(self.gather)>self._length:
                    self.point_data.append([self.i,
                                            self.assumption.Index, 
                                            [x.price for x in self.gather[-self._length:]], 
                                            [x.Index for x in self.gather[-self._length:]],
                                            self.assumption.price, 
                                            self.ball.price] )
                    
#         self.gather.append(self.assumption) # to be done with NN classifier
        self.df = pd.DataFrame(self.gather).set_index('Index')
        self.df['name'] = self.df.price.pct_change().map(lambda x: 'max' if x>0 else 'min')
        return self.df
    
    def turn_ahead(self, close: float,  min_change = None, bar_data: object = None):

        self.i += 1
        self.ball = close
        
        if self.min_change is None:
            self.min_change = min_change
        
        if not isinstance(self.assumption, float):
            self.assumption = self.assumption[1]
        
        self.calc_change_ahead(self.ball)

        if self.assumption_dir == 'go_up':
            if self.change>0:
                if self.assumption < self.ball:
                    self.assumption = self.ball
            elif self.change<0:
                if self.check_change():
                    self.assumption_dir = 'go_down'
                    self.gather.append(self.assumption)
                    self.df = self.df.append({'price':self.assumption}, ignore_index=True)
                    if self.assumption > self.ball:
                        self.assumption = self.ball
        if self.assumption_dir == 'go_down':
            if self.change<0:
                self.assumption = self.ball
            elif self.change>0:
                if self.check_change():
                    self.assumption_dir = 'go_up'
                    self.gather.append(self.assumption)
                    self.df = self.df.append({'price':self.assumption}, ignore_index=True)
                    if self.assumption < self.ball:
                        self.assumption = self.ball
 
        if self.EarlyDetector is not None:
        
            self.input_dict = {
                'assumpt':self.assumption,
                'dir':1 if self.assumption_dir == 'go_up' else 0,
                'change':self.change,
                'last_ext': self.gather[-1] if isinstance(self.gather[-1], float) else self.gather[-1].price}

            if self.EarlyDetector.predict(self.input_dict, bar_data):
                self.assumption_dir = 'go_up' if self.assumption_dir == 'go_down' else 'go_down'
                self.gather.append(self.assumption)
                self.df = self.df.append({'price':self.assumption}, ignore_index=True)
            
        self.df['name'] = self.df.price.pct_change().map(lambda x: 'max' if x>0 else 'min')
        return self.df
                
        
class EarlyDetector:
    """
    Class predicting if assumption is an extremum
    Uses neural network 
    """
    def __init__(self, nn, cols: list = [], n_before: dict = {},
                 scaler: object = None, threshold: float = 0.5):
        self.nn = nn
        self.cols = cols
        self.n_before = n_before
        self.scaler = scaler
        self.threshold = threshold
    
    def prepare_pred_data(self, input_dict, input_bars):
        input = []
        for col in self.cols:
            if col in input_dict.keys():
                input.append( input_dict[col] )
            elif col in input_bars.keys():
                if col == 'close':
                    close = input_bars['close'].values[-self.n_before['close']:][::-1]
                    X_close = (close-min(close))/(max(close)-min(close)) 
                else:
                    input.extend( input_bars[col].values[-self.n_before[col]:][::-1] )
            else:
                raise ValueError(f"col {col} not in data")
                
        return [self.scaler.transform([input]).tolist()[0] + X_close.tolist()]
        
        
    def predict(self, input_dict, input_bars):
        input = self.prepare_pred_data(input_dict, input_bars)
        self._pred = self.nn.predict(input)
        return self._pred[0][0] > self.threshold
    
        
class WaveScaler:
    """ 
    It's MinMaxScaler with ignoring last element option. 
    We don't want to take the last element (y) into consideration, 
    because than scaling can contain some information from the future.
    """
    
    @staticmethod
    def full_list(wave): 
        return np.array([x.reshape(-1,1) if isinstance(x, np.ndarray) else x for x in wave]).reshape(-1,1)
    
    def __init__(self):
        pass
    
    def fit(self, wave):
        wave = self.full_list(wave)
        self.min_ = min(wave[:-1])
        self.max_ = max(wave[:-1])
    
    def full_fit(self, wave):
        wave = self.full_list(wave)
        self.min_ = min(wave)
        self.max_ = max(wave)
    
    def transform(self, wave):
        if isinstance(wave, (float,int)):
            wave = [wave]
        if isinstance(wave[0], (np.ndarray, pd.Series, tf.Tensor)):
            return [(x_-self.min_)/(self.max_-self.min_) for x_ in wave]
        else:
            return (wave-self.min_)/(self.max_-self.min_)
            
    def inverse_transform(self, wave: object):
        return wave*(self.max_-self.min_)+self.min_
    
    def fit_transform(self, wave, full_fit: bool = False):
        if full_fit:
            self.full_fit(wave)
        else:
            self.fit(wave)
        return self.transform(wave)
        
        
class WaveGrasper(BallTurner):
    """
    Version 0.03
    """
    @staticmethod
    def wave_min_max(x: list):
        min_ = min(x[:-1])
        max_ = max(x[:-1])
        return (x-min_)/(max_-min_), [min_,max_]
    
    @staticmethod
    def full_min_max(x: list):
        min_ = min(x)
        max_ = max(x)
        return (x-min_)/(max_-min_), [min_,max_]
      
    @staticmethod
    def rescale(y, scalers):
      # (x-min_)/(max_-min_) reversal
        min_, max_ = np.array(scalers)[:,0].reshape(-1,1), np.array(scalers)[:,1].reshape(-1,1)
        y = y.reshape(-1,1)
        return y*(max_-min_)+min_
    
    
    def __init__(self, series, min_change = 0.02, length = 6):
        super().__init__(series)
        self.Series = self.turn(min_change)
        if self.feat_eng:
            self.Series = self.extract_aggs(self.Series, series)
        self.get_ABCD_y(length)
        
    def get_ABCD_y(self, length = 6):
        """ 
        A function which transforms waves sequence to ABCD formation and scales it 
        
        Input:
        > length - how many extremas including y
        """
        X = self.Series.price.values
        inds = self.Series.index.values
        self.__inds__ = inds
        ABCD = []
        y = []
        scalers = []
        ind = []
        
        # get scaled waves
        for i in range(length, len(X)):
            wave = X[i-length:i]
            ind.append( inds[i-1] )
            wave, extrema = self.wave_min_max(wave)
            scalers.append(extrema)
            ABCD.append( wave[:-1].tolist() )
            y.append( wave[-1] )
        
        self.ABCD = np.array(ABCD).reshape(-1, length-1)
        self.y = np.array(y).reshape(-1,)
        self.scalers = scalers
        self.inds = ind
    
        return self.ABCD , self.y, self.scalers
    
    def get_transformer_data(self, length = 50):
        
        X = self.Series.price.values
        inds = self.Series.index.values
        self.__inds__ = inds
        
        input_encoder = []
        input_decoder = []
        output_decoder = []
    
        scalers = []
        ind = []
        
        # get scaled waves
        for i in range(length, len(X)):
            T = X[i-length:i]
            ind.append( inds[i-1] )
            scaler = WaveScaler()
            T = scaler.fit_transform(T)
            scalers.append(scaler)
            
            input_encoder.append(T[:-2].tolist() )
            input_decoder.append(T[-3:-1].tolist())
            output_decoder.append(T[-2:].tolist())
            
        self.input_encoder = np.array(input_encoder).reshape(-1,length-2)
        self.input_decoder = np.array(input_decoder).reshape(-1,2)
        self.output_decoder = np.array(output_decoder).reshape(-1,2)
        
        self.scalers = scalers
        self.inds = ind
        
        return self.input_encoder, self.input_decoder, self.output_decoder, self.scalers
    
    
    def get_categorical_transformer_data(self, length = 50, prediction: str = False, continous: bool = False):
        
        X = self.Series.price.values
        inds = self.Series.index.values
        self.__inds__ = inds
        
        input_encoder = []
        input_decoder = []
        output_decoder = []
        scalers = []
        ind = []
        
        if prediction:
            # get only the last set of waves
            T = X[-length+1:]
            scaler = WaveScaler()
            T = scaler.fit_transform(T, full_fit=True)
            return [T[:-1]], [T[-2:]]
        else:
            # get scaled waves
            for i in range(length, len(X)):
                T = X[i-length:i]
                ind.append( inds[i-1] )
                scaler = WaveScaler()
                T = scaler.fit_transform(T)
                scalers.append(scaler)

                input_encoder.append(T[:-2].tolist() )
                input_decoder.append(T[-3:-1].tolist())            
                if continous:
                    output_decoder.append([float(T[-1] - T[-3])])
                    
                output_decoder.append([float(T[-1] - T[-3])])

        self.input_encoder = np.array(input_encoder).reshape(-1,length-2)
        self.input_decoder = np.array(input_decoder).reshape(-1,2)
        self.output_decoder = np.array(output_decoder).reshape(-1,1)
        
        self.scalers = scalers
        self.inds = ind
        
        return self.input_encoder, self.input_decoder,self.output_decoder, self.scalers
    
    

class WaveGrasper(BallTurner):
    """
    Version 0.04
    """

    @staticmethod
    def extract_aggs(Series, price):
        
        price = pd.concat([price.reset_index(), Series],1)

        price['inter'] = price.price.interpolate('linear')
        price['residuals'] = price.close - price.inter

        i = 0
        num = []
        for v in price.price.values.tolist():
            num.append(i)
            if not math.isnan(v):
                i+=1

        price['wave'] = num
#         print(price)
        agg = price.groupby('wave')[['residuals','price']].agg(['mean','std','count'])
        agg = agg.iloc[:-1,:]['residuals'].rename(columns = {'mean':'bias','count':'time'})
        agg.index = Series.index
        agg['std'] = agg['std'].fillna(0)
        
        Series = pd.concat([Series, agg], 1)
        return Series, None
    
    
    @staticmethod
    def wave_min_max(x: list):
        min_ = min(x[:-1])
        max_ = max(x[:-1])
        return (x-min_)/(max_-min_), [min_,max_]
    
    @staticmethod
    def full_min_max(x: list):
        min_ = min(x)
        max_ = max(x)
        return (x-min_)/(max_-min_), [min_,max_]
      
    @staticmethod
    def rescale(y, scalers):
      # (x-min_)/(max_-min_) reversal
        min_, max_ = np.array(scalers)[:,0].reshape(-1,1), np.array(scalers)[:,1].reshape(-1,1)
        y = y.reshape(-1,1)
        return y*(max_-min_)+min_
    
    
    def __init__(self, series, min_change = 0.02, length = 6, gather_point_data: bool = False, feature_engineering: bool = False):
        self.feat_eng = feature_engineering
        self._min_change = min_change
        self._length = length
        super().__init__(series)
        self.Series = self.turn(min_change, gather_point_data)
        if self.feat_eng:
            self.Series, self.feature_scaler = self.extract_aggs(self.Series, series)
            self.df = self.Series.copy()
#         self.Waves = self.extract_waves(self.Series) # ind means indexes
#         self.get_ABCD_y(length)
        
    def get_ABCD_y(self, length = 6):
        """ 
        A function which transforms waves sequence to ABCD formation and scales it 
        
        Input:
        > length - how many extremas including y
        """
        X = self.Series.price.values
        inds = self.Series.index.values
        self.__inds__ = inds
        ABCD = []
        y = []
        scalers = []
        ind = []
        
        # get scaled waves
        for i in range(length, len(X)):
            wave = X[i-length:i]
            ind.append( inds[i-1] )
            wave, extrema = self.wave_min_max(wave)
            scalers.append(extrema)
            ABCD.append( wave[:-1].tolist() )
            y.append( wave[-1] )
        
        self.ABCD = np.array(ABCD).reshape(-1, length-1)
        self.y = np.array(y).reshape(-1,)
        self.scalers = scalers
        self.inds = ind
    
        return self.ABCD , self.y, self.scalers
    
    def get_transformer_data(self, length = 50, prediction: bool = False, categorical: bool = False, trend_grasper: bool = False, decision: bool = False):
        
        if self.feat_eng:
            X = self.Series[['price','bias','std','time']].iloc[1:].values
#             print('nans sum in cols \n',np.sum(np.isnan(X),0))
        else:
            X = self.Series.price.values
            
        if 'decision' in self.Series:
            decisions = self.Series.decision.values.tolist()
            
        inds = self.Series.index.values
        self.__inds__ = inds
        
        input_encoder = []
        input_decoder = []
        output_decoder = []
    
        scalers = []
        ind = []
        
        if prediction:
            # get only the last set of waves
            if self.feat_eng:
                T = X[-length+1:]            
                self._scaler = WaveScaler()
                T = np.concatenate((self._scaler.fit_transform(T[:,0], full_fit=True).reshape(-1,1),T[:,1:]),1)
                return [T[:-1]], [T[-2:]]
                
            else:
                T = X[-length+1:]            
                self._scaler = WaveScaler()
                T = self._scaler.fit_transform(T, full_fit=True)
                return [T[:-1]], [T[-2:]]
        # get scaled waves
        print('decisions')
        for i in range(length, len(X)-1):
            T = X[i-length:i]
                
            ind.append( inds[i] )
            scaler = WaveScaler()
            if self.feat_eng:
                T = np.concatenate((scaler.fit_transform(T[:,0]).reshape(-1,1),T[:,1:]),1)
            else:
                T = scaler.fit_transform(T)
            if trend_grasper:
                T_add1 = X[i]
                T_add1 = scaler.transform(T_add1)
            scalers.append(scaler)
            
            if self.feat_eng:
                input_encoder.append(T[:-2,:])
                input_decoder.append(T[-3:-1,:])
                
                if decision:
                    output_decoder.append(decisions[i-1])
                else:
                    if categorical:
                        output_decoder.append(float((T_add1[0] if trend_grasper else T[-1,0]) > T[-3,0]))
                    else:
                        output_decoder.append(T[-2:,0])
            else:
                input_encoder.append(T[:-2].tolist() )
                input_decoder.append(T[-3:-1].tolist())
                
                if decision:
                    output_decoder.append(decisions[i-1])
                else:
                    if categorical:
                        output_decoder.append(float((T_add1 if trend_grasper else T[-1]) > T[-3]))
                    else:
                        output_decoder.append(T[-2:].tolist())
            
        if self.feat_eng:
            self.input_encoder = np.array(input_encoder)
            self.input_decoder = np.array(input_decoder)
            self.output_decoder = np.array(output_decoder)
        else:
            self.input_encoder = np.array(input_encoder).reshape(-1,length-2)
            self.input_decoder = np.array(input_decoder).reshape(-1,2)
            self.output_decoder = np.array(output_decoder) if categorical else np.array(output_decoder).reshape(-1,2)
        
        self.scalers = scalers
        self.inds = ind
        
        return self.input_encoder, self.input_decoder, self.output_decoder, self.scalers
    
    
    def get_categorical_transformer_data(self, length = 50, prediction: bool = False):
        
        X = self.Series.price.values
        inds = self.Series.index.values
        self.__inds__ = inds
        
        input_encoder = []
        input_decoder = []
        output_decoder = []
        scalers = []
        ind = []
        
        if prediction:
            # get only the last set of waves
            T = X[-length+1:]
            scaler = WaveScaler()
            T = scaler.fit_transform(T, full_fit=True)
            return [T[:-1]], [T[-2:]]
        else:
            # get scaled waves
            for i in range(length, len(X)):
                T = X[i-length:i]
                ind.append( inds[i-1] )
                scaler = WaveScaler()
                T = scaler.fit_transform(T)
                scalers.append(scaler)

                input_encoder.append(T[:-2].tolist() )
                input_decoder.append(T[-3:-1].tolist())
                output_decoder.append([float(T[-1] > T[-3])])

        self.input_encoder = np.array(input_encoder).reshape(-1,length-2)
        self.input_decoder = np.array(input_decoder).reshape(-1,2)
        self.output_decoder = np.array(output_decoder).reshape(-1,1)
        
        self.scalers = scalers
        self.inds = ind
        
        return self.input_encoder, self.input_decoder,self.output_decoder, self.scalers
    
    
def interpolate_waves(series, n_between: int = 4):
    """ 
        DEPRECATED
        
        Acces only for extrema is convenient for training
        Unfortunately, using onl extremas causes biases in trading process
        Therefore, linear interpolation simulates real price changes
        
    """
    series.index = series.index*n_between
    series = pd.concat([series, pd.Series(np.arange(0,len(series)*n_between))],1)[series.name]
    return series.interpolate('linear').iloc[:-n_between+1]
    

def rescale(scalers, waves):
    return np.array(list(map(lambda s,p: s.inverse_transform(p), scalers, waves)))


class DeepWaveGrasper(WaveGrasper):
    """
    WaveGrasper version 0.04
    
    Implementation contains implementatio of:
    > additive extremum finding
    > deep method of early extremum identification
    
    """
    
    def __init__(self, EarlyDetector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.EarlyDetector = EarlyDetector
        
    def get_data(self, price, bar_data, use_ed: bool = False):
        """
        Args:
        > price - new close price
        > bar_data - dataframe
        """
        if use_ed:
#             self.data = bar_data.close.reset_index(drop=True).dropna()
#             self.pct_change = self.data.pct_change()
#             self.df = pd.DataFrame({'price':self.data})
#             self.Series = self.turn(self.min_change)
            self.turn_ahead(price, bar_data = bar_data)
            if self.feat_eng:
                self.Series, self.feature_scaler = self.extract_aggs(self.Series, bar_data.close)
                self.df = self.Series.copy()
        else:
            self.data = bar_data.close.reset_index(drop=True).dropna()
            self.pct_change = self.data.pct_change()
            self.df = pd.DataFrame({'price':self.data})
            self.Series = self.turn(self.min_change)
#             print(self.Series)
            if self.feat_eng:
                self.Series, self.feature_scaler = self.extract_aggs(self.Series,  bar_data.close)
                self.df = self.Series.copy()