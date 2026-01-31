import os


os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import tensorflow as tf

from numba import njit
import numpy as np
import random
import os


# print('SEED: ', seed_value)
# import time;time.sleep(10)
# os.environ['PYTHONHASHSEED']=str(seed_value)
# random.seed(seed_value)
# np.random.seed(seed_value)
# tf.random.set_seed(seed_value)

# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(5)

def tricky_argmax(x, epsilon: float = 0.00001):
    x = tf.keras.activations.relu(x-tf.reduce_max(x,1, keepdims=True)+epsilon)
    return x/epsilon

double_tanh = lambda x, a=5, b=2.5: (tf.tanh(x*a+b) + tf.tanh(x*a-b))/2. # 10

from utils import fast_rate_of_return

class ReturnLayer(tf.keras.layers.Layer):
    def __init__(self, lewar: int = 1, with_swap: bool = False):
        super(ReturnLayer, self).__init__()
        self.lewar = lewar
        self.with_swap = with_swap
        self.type='sharpe'
        
    @tf.function
    def call(self, y, pred):
        return -tf.reduce_mean((pred-y)**2)
        # return -tf.keras.losses.MSE(y,pred)

    
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
    dec = 1 # 2 if sum(lines[0]) >1 else 0
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
        
        history_input = layer = tf.keras.Input(shape=(x_shape))
        
        layer = store_init = tf.expand_dims(history_input,2)

        layer = tf.keras.layers.Flatten()(layer)

        # layer = self.min_max(history_input)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Dense(10)(layer)
        layer = tf.keras.activations.tanh(layer)
            
        super().__init__(inputs=[history_input], outputs=[layer])
        
        self.compile(
            optimizer='Adam', 
                    loss ='mse',
                    metrics = ['mse'])
        
        
# ten = tf.constant(10., dtype=tf.float32)

class Decider(tf.keras.Model):

    @staticmethod
    def double_tanh(x, a=5, b=2.5): 
        # a = tf.math.multiply(a,10.)
        # b = tf.math.multiply(b,10.)
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
        
        adaptive_dtanh = False

        if adaptive_dtanh:
            self.a = tf.Variable(5., trainable=True, dtype=tf.float32)
            self.b = tf.Variable(2.5, trainable=True, dtype=tf.float32)
        else:
            self.a,self.b = 0.5, 0.25

        layer_input = tf.keras.Input(shape=(layer_shape))
        current_state_input = tf.keras.Input((3))
        
        layer = tf.concat([layer_input, current_state_input],1)
        
        layer = tf.keras.layers.Dense(5)(layer)
        decisions = tf.keras.activations.tanh(layer)
        layer = tf.keras.layers.Dense(1)(layer)
        decisions = tf.nn.tanh(layer) # double_tanh(layer)
            
        super().__init__(inputs=[layer_input, current_state_input], outputs=[decisions])
        
        self.compile(
                optimizer='Adam', 
                loss ='mse',
                metrics = [self.monitor_loss])
        
class NoDecider(tf.keras.Model):
    @staticmethod
    @tf.function
    def monitor_loss(maximum_reward_output, model_reward_output):
        return model_reward_output
    
    def __init__(self, 
                 layer_shape: int, 
                 activation: object = tf.keras.layers.LeakyReLU(0.1),
                 rest_connection: bool = False
                ):
        
        adaptive_dtanh = False

        if adaptive_dtanh:
            self.a = tf.Variable(5., trainable=True, dtype=tf.float32)
            self.b = tf.Variable(2.5, trainable=True, dtype=tf.float32)
        else:
            self.a,self.b = 0.5, 0.25

        layer_input = tf.keras.Input(shape=(layer_shape))
        layer=layer_input
        # current_state_input = tf.keras.Input((3))
        
        # layer = tf.concat([layer_input, current_state_input],1)
        
        layer = tf.keras.layers.Dense(5)(layer)
        decisions = tf.keras.activations.tanh(layer)
        layer = tf.keras.layers.Dense(1)(layer)
        decisions = tf.nn.tanh(layer) # double_tanh(layer)
            
        super().__init__(inputs=[layer_input], outputs=[decisions])
        
        self.compile(
                optimizer='Adam', 
                loss ='mse',
                metrics = [self.monitor_loss])
        
        
        
class Trader():
    def __init__(self, deviser: Deviser = None, 
                 decider: Decider = None,
                 input_len: int = None,
                 rn: bool = True
                ):

        self.rn=rn
        
        if rn:
            self.deviser = Deviser(input_len)
            self.decider = Decider(self.deviser.output.shape[1])
        else:
            self.deviser = Deviser(input_len)
            self.decider = NoDecider(self.deviser.output.shape[1])
        
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


    def compute_apply_grads(self, tape, loss):

        grad_dev, grad_dec = tape.gradient(loss,
                                        [self.deviser.trainable_weights, 
                                            self.decider.trainable_weights])
        
        grads = [tf.clip_by_value(g, -1000, 1000) for g in grad_dev]
        self.deviser.optimizer.apply_gradients(zip(grads, self.deviser.trainable_weights))
        
        grads = [tf.clip_by_value(g, -1000, 1000) for g in grad_dec]
        self.decider.optimizer.apply_gradients(zip(grads, self.decider.trainable_weights))

        return grad_dev, grad_dec
    

    # def call(self, inputs, training=False):
    #     devise_pred = self.deviser(inputs, training=training)
    #     dec = self.decider(devise_pred, training=training)
    #     reward = self.financier([inputs, dec], training=training)
    #     return reward
        

    def train_iteration(self, XV, Y, step_by_step:bool=False, online_learning:bool=False):
        if self.rn:
            with tf.GradientTape() as tape:
                devise_pred = self.deviser(XV)
                pred = self.decider(self.multiply_decisions(devise_pred))
                dec_known = self.solve_decision_path(pred)
                dec = self.decider([devise_pred, dec_known])
                dec = dec[:,0]
                loss_value = tf.reduce_mean((Y-dec)**2)

        else:
            with tf.GradientTape() as tape:
                devise_pred = self.deviser(XV)
                dec = self.decider(devise_pred)
                dec = dec[:,0]
                loss_value = tf.reduce_mean((Y-dec)**2)
        reward=-loss_value
        grad_dev, grad_dec = tape.gradient(loss_value,
                                        [self.deviser.trainable_weights, 
                                            self.decider.trainable_weights])

        grads = [tf.clip_by_value(g, -1000, 1000) for g in grad_dev]
        self.deviser.optimizer.apply_gradients(zip(grads, self.deviser.trainable_weights))
        
        grads = [tf.clip_by_value(g, -1000, 1000) for g in grad_dec]
        self.decider.optimizer.apply_gradients(zip(grads, self.decider.trainable_weights))
        
        # decisions = tf.argmax(dec_known,1)-1
        return grad_dev+grad_dec, dec, reward, loss_value
    
    
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
            devise_pred = self.deviser.predict(XV, batch_size = batch_size, verbose=0)
            pred = self.decider(self.multiply_decisions(devise_pred))
            dec_known = self.solve_decision_path(pred)
            
            if just_historical_path:
                return dec_known[:-1]
            
        reward = self.financier( [BA, self.decider.predict([devise_pred, dec_known],
                                                           batch_size = batch_size)])
        loss_value = 1/reward
        decisions = tf.argmax(dec_known,1)-1
        
        return [], decisions, reward, loss_value
    
    def fit(self, XV, Y, epochs, verbose=0):
        for e in range(epochs):
            grads, decisions, rewards, loss_value = self.train_iteration(XV, Y,
                                                                            step_by_step=False,
                                                                            online_learning=False
                                                                            )
            grads = [sum([x.numpy().reshape(-1,).tolist() for x in grads],[])]
            # else:
            #     tr_it = list(range(int(batch_size*np.random.uniform(1,1.5)),len(XV), batch_size))
            #     random.shuffle(tr_it)
            #     for _i in tr_it:  
            #         xv,ba = XV[_i-batch_size:_i], BA[_i-batch_size:_i]
            #         grad, decisions, reward, loss_value = trader.train_iteration(xv, ba)
            #         rewards.append(reward)
            #         grads.append(sum([x.numpy().reshape(-1,).tolist() for x in grad],[]))
            
            return_rate = np.round(np.mean(rewards),5) # round((np.prod(rewards)-1)*100,3) if 'sharpe' == 'sharpe' else np.round(np.mean(rewards),5)
            avg_grads = np.mean(np.abs(sum(grads,[])))
            if verbose>0:
                print(return_rate)
    
    # def fit(self, XV, Y, epochs, batch_size=1024):
    #     dataset = tf.data.Dataset.from_tensor_slices((XV, Y)).shuffle(buffer_size=1024).batch(batch_size)
    #     for e in range(epochs):
    #         epoch_losses = []
    #         for batch_X, batch_Y in dataset:
    #             with tf.GradientTape() as tape:
    #                 devise_pred = self.deviser(batch_X, training=True)
    #                 dec = self.decider(devise_pred, training=True)
    #                 reward = self.financier([batch_Y, dec], training=True)
    #                 loss_value = -reward  # Assuming reward is -MSE

    #             # Collect trainable variables from sub-models
    #             trainable_vars_deviser = self.deviser.trainable_variables
    #             trainable_vars_decider = self.decider.trainable_variables

    #             grads_deviser,grads_decider = tape.gradient(loss_value, [trainable_vars_deviser,trainable_vars_decider])

    #             self.deviser.optimizer.apply_gradients(zip(grads_deviser, trainable_vars_deviser))
    #             self.decider.optimizer.apply_gradients(zip(grads_decider, trainable_vars_decider))

    #             epoch_losses.append(loss_value.numpy())

    #         avg_loss = np.mean(epoch_losses)
    #         print(f"Epoch {e+1}/{epochs} - Loss: {avg_loss}")
    
    def set_lr(self, lr: float):
        self.decider.optimizer.lr.assign(lr)
        self.deviser.optimizer.lr.assign(lr)
    
    
    def __call__(XV, BA):
        return 
    