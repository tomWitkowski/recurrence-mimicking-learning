import os
from config import Config as cfg


os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1


import tensorflow as tf

from numba import njit

# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(5)

def tricky_argmax(x, epsilon: float = 0.00001):
    x = tf.keras.activations.relu(x-tf.reduce_max(x,1, keepdims=True)+epsilon)
    return x/epsilon


double_tanh = lambda x, a=5, b=2.5: (tf.tanh(x*a+b) + tf.tanh(x*a-b))/2. # 10

from utils import fast_rate_of_return

# class ReturnLayer(tf.keras.layers.Layer):
#     """
#     It's simplified version of the main ReturnLayer
#     """
#     def __init__(self, lewar: int = cfg.lewar, with_swap: bool = False):
#         super(ReturnLayer, self).__init__()
#         self.lewar = lewar
#         self.with_swap = with_swap
        
#     @tf.function
#     def call(self, inp, cost: float = 0.005):
#         trans_cost =   tf.math.pow(1-cost,  tf.reduce_sum( tf.abs(inp[1:,-1] - inp[:-1,-1]) ) ) 
#         dB = (inp[1:,1] - inp[:-1,1]) / inp[:-1,1]

#         additive_return = tf.reduce_sum(dB*inp[:-1,2] )
#         general_return =  (1+additive_return)*trans_cost
    
#         return general_return
    
    
class ReturnLayer(tf.keras.layers.Layer):
    def __init__(self, lewar: int = cfg.lewar, with_swap: bool = False):
        super(ReturnLayer, self).__init__()
        self.lewar = lewar
        self.with_swap = with_swap
        
        
    @tf.function
    def call(self, inp):
        spreads_perc =  tf.reduce_mean( (inp[:,1] - inp[:,0])/inp[:,0], 0 )*2
        trans_cost =   tf.math.pow(1-spreads_perc*self.lewar,  tf.reduce_sum( tf.abs(inp[1:,-1] - inp[:-1,-1]) )/2 ) 
        dA = inp[1:,1] - inp[:-1,1]

        additive_return = tf.reduce_sum(dA*inp[:-1,2] / inp[:-1,1])*self.lewar
        general_return =  (1+additive_return)*trans_cost
    
        if self.with_swap:
            point_perc = 0.00001 / 1.1
            minutes_per_day = 60*24
            long_swap_per_day = 7 * point_perc

            n_minutes_opened_position = tf.reduce_sum( tf.abs(inp[:,2]) )
            swap_multiplier = ((1-long_swap_per_day)**(1/minutes_per_day))**(n_minutes_opened_position*self.lewar) 
            
            # swap_multiplier = swap_multiplier**(1/3)
            
            general_return = general_return * swap_multiplier
    
#         general_return = tf.py_function(fast_rate_of_return, [inp], tf.float32)
    
        return general_return
    
class DiffLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DiffLayer, self).__init__()
        self._flatten = tf.keras.layers.Flatten()

#     @tf.function
    def __call__(self, x):
        x = tf.expand_dims(x, 0) 
        x = x - tf.transpose(x)
        x = tf.transpose(x, [1,0,2])
        return self._flatten(x)
    

def rsin(x, r, pi = 3.141592653589793):
    return tf.math.sin(x*pi*r*2)/r  


# def min_max(t):
#     ma = tf.math.reduce_max(t,axis=1)
#     mi = tf.math.reduce_min(t,axis=1)

#     mi = tf.transpose( tf.reshape(tf.concat([mi]*t.shape[1],0),(t.shape[1],-1)) )
#     ma = tf.transpose( tf.reshape(tf.concat([ma]*t.shape[1],0),(t.shape[1],-1)) )

#     return (t-mi)/(ma-mi)

@njit
def get_decs(lines):
    dec = 2 if sum(lines[0]) >1 else 0
    decs_ = [dec]

    # deterministic decisions only
    for l in lines:
        dec = l[dec]
        decs_.append(dec)
        
    return decs_
   
    
class Deviser(tf.keras.Model):
    """
    Model taking input and returning latent market state
    """
    @staticmethod
    def min_max(t):
        
        
        
        ma = tf.math.reduce_max(t,axis=1)
        mi = tf.math.reduce_min(t,axis=1)

        mi = tf.transpose( tf.reshape(tf.concat([mi]*t.shape[1],0),(t.shape[1],-1)) )
        ma = tf.transpose( tf.reshape(tf.concat([ma]*t.shape[1],0),(t.shape[1],-1)) )

        return (t-mi)#/(ma-mi)
    
    @staticmethod
    def conv(x, filters, length, expand: bool = False, flatten: bool = False, 
             convl: bool = True, drout: bool = True, strides: int = 1, att:bool =False, padding='same'):
        if expand:
            compat_input = x = tf.expand_dims(x,2)
        else:
            compat_input = x

        if convl:
            x = tf.keras.layers.Conv1D(filters, length, strides=strides, padding=padding)(x)

        x = tf.keras.layers.BatchNormalization()(x)

        if att:
            x = x + tf.keras.layers.Attention()([x,x])

        if flatten:
            compat_input = x = tf.keras.layers.Flatten()(x)
        if drout:
            x = tf.keras.layers.Dropout(0.2)(x)
        return x
    
    def __init__(self, x_shape: int):
        
        history_input = tf.keras.Input(shape=(x_shape))
        
        layer = self.min_max(history_input)
        layer = tf.keras.layers.BatchNormalization()(layer)
    
        layer = self.conv(layer,16,5,strides=1,expand=True,flatten=False,att=False,convl=True)
    
        # layer = self.conv(layer,50,6,strides=1,expand=False,flatten=False,att=False,convl=True)
        # layer += self.conv(layer,50,3,strides=1,expand=False,flatten=False,att=False,convl=True)
        layer = tf.keras.layers.MaxPooling1D(2,2)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
#         layer = tf.random.normal(tf.shape(layer), layer, 0.001)
        
        # for i in range(2):
        #     layer = self.conv(layer,8*2**(i+1),5,strides=1,expand=False,flatten=False,att=False,convl=True)
        #     layer += self.conv(layer,8*2**(i+1),3,strides=1,expand=False,flatten=False,att=False,convl=True)
        #     layer = tf.keras.layers.MaxPooling1D(2,2)(layer)
        #     layer = tf.keras.layers.BatchNormalization()(layer)
    
        # layer = self.conv(layer,50,4,strides=1,expand=False,flatten=False,att=False,convl=True)
        # layer = self.conv(layer,12,4,strides=1,expand=False,flatten=False,att=False,convl=True)
        
        layer = tf.keras.layers.Flatten()(layer)

        # -----------------------------------------------

        KD = 1
        NH = 20
        
        
#         layer = tf.keras.layers.Dense(40)(layer)
#         layer = tf.keras.layers.Dropout(0.1)(layer)
#         layer = tf.keras.layers.BatchNormalization()(layer)
#         layer = tf.keras.layers.LeakyReLU(0.1)(layer)
        
#         layer = tf.keras.layers.Dense(20)(layer)
#         layer = tf.keras.layers.Dropout(0.1)(layer)
#         layer = tf.keras.layers.BatchNormalization()(layer)
#         layer = tf.keras.layers.LeakyReLU(0.1)(layer)
        
#         att = tf.keras.layers.MultiHeadAttention(num_heads=NH, 
#                                                  key_dim=KD
#                                                 )(layer[..., None],
#                                                   layer[..., None])
#         layer += tf.keras.layers.Flatten()(att)
#         layer = tf.keras.layers.BatchNormalization()(layer)
        
        
#         layer = tf.keras.layers.Dense(30)(layer)
#         layer = tf.keras.layers.Dropout(0.1)(layer)
#         layer = tf.keras.layers.BatchNormalization()(layer)
#         layer = tf.keras.layers.LeakyReLU(0.1)(layer)
        
        
#         att = tf.keras.layers.MultiHeadAttention(num_heads=NH, 
#                                                  key_dim=KD
#                                                 )(layer[..., None],
#                                                   layer[..., None])
#         layer += tf.keras.layers.Flatten()(att)
#         layer = tf.keras.layers.BatchNormalization()(layer)

#         layer = tf.keras.layers.Dense(10)(layer)
#         layer = tf.keras.layers.Dropout(0.1)(layer)
#         layer = tf.keras.layers.BatchNormalization()(layer)
#         layer = tf.keras.layers.LeakyReLU(0.1)(layer)
        
#         att = tf.keras.layers.MultiHeadAttention(num_heads=NH, 
#                                                  key_dim=KD
#                                                 )(layer[..., None],
#                                                   layer[..., None])
#         layer += tf.keras.layers.Flatten()(att)
#         layer = tf.keras.layers.BatchNormalization()(layer)
    
#         layer = DiffLayer()(layer)
#         layer = tf.keras.layers.BatchNormalization()(layer)
# #         layer = tf.random.normal(tf.shape(layer), layer, 0.01)
        
#         # -----------------------------------------------
        
#         layer = tf.keras.layers.Dense(25)(layer)
#         layer = tf.keras.layers.Dropout(0.1)(layer)
#         layer = tf.keras.layers.BatchNormalization()(layer)
#         layer = tf.keras.layers.LeakyReLU(0.1)(layer)
        
#         att = tf.keras.layers.MultiHeadAttention(num_heads=NH, 
#                                                  key_dim=KD
#                                                 )(layer[..., None],
#                                                   layer[..., None])
#         layer += tf.keras.layers.Flatten()(att)
#         layer = tf.keras.layers.BatchNormalization()(layer)
        
        # layer = tf.keras.layers.Dense(50)(layer)

        # layer = tf.keras.layers.Dropout(0.1)(layer)
        # layer = tf.keras.layers.BatchNormalization()(layer)
        # layer = tf.keras.layers.LeakyReLU(0.1)(layer)

        layer = tf.keras.layers.Dense(20)(layer)

        layer = tf.keras.layers.Dropout(0.1)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.LeakyReLU(0.1)(layer)

        layer1 = tf.keras.layers.Dense(15)(layer)
        
        layer2 = tf.keras.layers.Dense(15)(layer)
        
        layer = layer1 - layer2
        layer = tf.keras.layers.BatchNormalization()(layer)
        
#         layer = layer1
        
#         KD = 1
#         NH = 10
        
#         att = tf.keras.layers.MultiHeadAttention(num_heads=NH, 
#                                                  key_dim=KD
#                                                 )(layer[..., None],
#                                                   layer[..., None])
#         layer += tf.keras.layers.Flatten()(att)
#         layer = tf.keras.layers.BatchNormalization()(layer)
        
        super().__init__(inputs=[history_input], outputs=[layer])
        
        self.compile(
            optimizer='Adam', 
                    loss ='mse',
                    metrics = ['mse'])
        
        
class Decider(tf.keras.Model):

    @staticmethod
    def double_tanh(x, a=5, b=2.5): 
        return (tf.tanh(x*a+b) + tf.tanh(x*a-b))/2. # 10
    
    @staticmethod
    def dense(layer, n_neurons, activ, rest: bool = True):
        
        input_layer = layer
        
        layer = tf.keras.layers.Dense(n_neurons)(layer) # 50  tf.keras.layers.LeakyReLU(0.2) 
        layer = tf.keras.layers.Dropout(0.1)(layer)

        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = activ(layer)

        if rest:
            layer_rest = tf.keras.layers.Dense(n_neurons)(tf.concat([layer, input_layer], 1)) # 50  tf.keras.layers.LeakyReLU(0.2) 
            layer_rest = tf.keras.layers.Dropout(0.1)(layer_rest)

            layer += layer_rest

            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = activ(layer)

        return layer
    

    @staticmethod
    @tf.function
    def monitor_loss(maximum_reward_output, model_reward_output):
        return model_reward_output
    
    
    def __init__(self, 
                 layer_shape: int, 
                 activation: object = tf.keras.layers.LeakyReLU(0.1),
                 rest_connection: bool = False
                ):
        
        layer_input = tf.keras.Input(shape=(layer_shape))
        current_state_input = tf.keras.Input((3))
        
        layer = tf.concat([layer_input, current_state_input],1)
        
        # layer = self.dense(layer,12, activation, rest_connection)
#         layer = tf.concat([layer, current_state_input],1)
        layer = self.dense(layer, 10, activation, rest_connection)
#         layer = tf.concat([layer, current_state_input],1)
        layer = self.dense(layer, 5, activation, rest_connection)

        layer = tf.keras.layers.Dense(1)(layer)
        decisions = double_tanh(layer)
            
        super().__init__(inputs=[layer_input, current_state_input], outputs=[decisions])
        
        self.compile(
                optimizer='Adam', 
                loss ='mse',
                metrics = [self.monitor_loss])
        
        
class Financier(tf.keras.Model):
    def __init__(self):
        bid_ask_input = tf.keras.Input((2))
        decision_vector = tf.keras.Input((1))
        
        bid_ask_input_estimated = tf.expand_dims(tf.concat([bid_ask_input, decision_vector[:,-1:]],1),0)
        bid_ask_input_estimated = tf.reshape(bid_ask_input_estimated, (-1,3))

        # x_vector = bid_ask_input[0,:]
        # naive_best_reward = tf.math.abs(x_vector[-1] - x_vector[0])/x_vector[0] * cfg.lewar * 100 + 1


        estimated_reward = ReturnLayer()(bid_ask_input_estimated) #/ naive_best_reward

        # avoid_transaction_regularization = 1-tf.reduce_mean(decision_vector**2)**3
        # estimated_reward *=  (1-(1-avoid_transaction_regularization)/10)

        super().__init__(inputs = [bid_ask_input, decision_vector], outputs = [estimated_reward])
        
        
class Trader:
    def __init__(self, deviser: Deviser = None, 
                 decider: Decider = None,
                 financier: Financier = None,
                 input_len: int = None
                ):
        
        if None in [deviser, decider, financier]:
            
            self.deviser = Deviser(input_len)
            self.decider = Decider(self.deviser.output.shape[1])
            self.financier = Financier()
            
        else:
            self.deviser = deviser
            self.decider = decider
            self.financier = financier
        
        self._all_dec = []
        self._xvs = []
        
        
    def multiply_decisions(self, XV):
        """
        XVs = input of the decider
        """
        if len(XV) != int(len(self._xvs)/3):
            zeros = tf.zeros((len(XV),1))
            ones = tf.ones((len(XV),1))
            self._all_dec = tf.concat( [tf.concat([ones, zeros, zeros],1), 
                                  tf.concat([zeros, ones, zeros],1), 
                                  tf.concat([zeros, zeros, ones],1) ],0)
            self._xvs = tf.concat([XV,XV,XV],0)
            
        return self._xvs, self._all_dec
    
    
    def solve_decision_path(self, pred):
        lines = tf.concat( tf.split( tf.round(pred)+1, 3), 1)
        lines = lines.numpy().astype(int)

        decs_ = get_decs(lines)

        return tf.one_hot(decs_[:-1],3)

    
    def train_iteration(self, XV, BA):
        
        with tf.GradientTape() as tape:
            devise_pred = self.deviser(XV)
            pred = self.decider(self.multiply_decisions(devise_pred))
            dec_known = self.solve_decision_path(pred)
            reward = self.financier( [BA, self.decider([devise_pred, dec_known]) ])

#             dec = 0.
#             container = tf.constant([[0.]])
#             for xv in XV:
#                 latent = self.deviser(xv)
#                 dec = self.decider([latent, dec])
#             reward = self.financier( [BA, self.decider([devise_pred, dec_known]) ])

            # loss_value = 1/reward
            loss_value = -reward

        grad_dev, grad_dec = tape.gradient(loss_value,
                                           [self.deviser.trainable_weights, 
                                            self.decider.trainable_weights])

        grads = [tf.clip_by_value(g, -1000, 1000) for g in grad_dev]
        self.deviser.optimizer.apply_gradients(zip(grads, self.deviser.trainable_weights))
        
        grads = [tf.clip_by_value(g, -1000, 1000) for g in grad_dec]
        self.decider.optimizer.apply_gradients(zip(grads, self.decider.trainable_weights))
        
        decisions = tf.argmax(dec_known,1)-1
        
        return grad_dev+grad_dec, decisions, reward, loss_value
    
    
    def test_iteration(self, XV, BA, batch_size: int = 10_000, 
                       step_by_step: bool = False,
                       just_historical_path: bool = False
                      ):
        
        if step_by_step:
            dec_known = tf.constant([1.])
            for xv in XV:
                dec = tf.one_hot(tf.cast(tf.round(dec_known[-1:,None]), tf.int32),3)[0,...]
                pred = self.decider([self.deviser(xv[None,:]), 
                                     dec])
                
                dec_known = tf.concat([dec_known, tf.cast(tf.argmax(pred,1), tf.float32)], axis=0)

            if just_historical_path:
                return dec_known
            
        else:
            devise_pred = self.deviser.predict(XV, batch_size = batch_size)
            pred = self.decider(self.multiply_decisions(devise_pred))
            dec_known = self.solve_decision_path(pred)
            
            if just_historical_path:
                return dec_known[:-1]
            
        reward = self.financier( [BA, self.decider.predict([devise_pred, dec_known],
                                                           batch_size = batch_size)])
        loss_value = 1/reward
        decisions = tf.argmax(dec_known,1)-1
        
        return [], decisions, reward, loss_value
    
    
    
    
    def set_lr(self, lr: float):
        self.decider.optimizer.lr.assign(lr)
        self.deviser.optimizer.lr.assign(lr)
    
    
    def __call__(XV, BA):
        return 
    