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
    
        return general_return
    

def rsin(x, r, pi = 3.141592653589793):
    return tf.math.sin(x*pi*r*2)/r  


def min_max(t):
    # ma = tf.math.reduce_max(t,axis=1)
    mi = tf.math.reduce_min(t,axis=1)

    mi = tf.transpose( tf.reshape(tf.concat([mi]*t.shape[1],0),(t.shape[1],-1)) )
    # ma = tf.transpose( tf.reshape(tf.concat([ma]*t.shape[1],0),(t.shape[1],-1)) )

    return (t-mi)#*100 #(ma-mi)

@njit
def get_decs(lines):
    dec = 2 if sum(lines[0]) >1 else 0
    decs_ = [dec]

    # deterministic decisions only
    for l in lines:
        dec = l[dec]
        decs_.append(dec)
        
    return decs_


class Trader:
    
    
    def __init__(self, x_shape: int, length: int = None):        
        
        
        def submodel(layer,current_state_input, activ):
            layer = dense(layer, 5, activ, False)
            layer = tf.concat([layer,  current_state_input],1)
            layer = dense(layer, 5, activ, True)
            layer = dense(layer, 1, activ, False)

            return layer
        
        
        # @tf.function
        def two_to_three(x):
            x = tf.concat([x[:,:1],get_gauss(x[:,-1:]),x[:,-1:]],1)
            return  x/tf.reshape(tf.reduce_sum(x,1),(-1,1))
        
        def attention(x):
            x = tf.expand_dims(x,2)
            x = x + tf.keras.layers.Attention()([x,x])
            x = tf.keras.layers.Flatten()(x)
            return x
        
        
        def att(x,q):
            x = tf.expand_dims(x,2)
            q = tf.expand_dims(q,2)
            x = tf.keras.layers.Attention()([x,q])
            x = tf.keras.layers.Flatten()(x)
            return x
        
        def dense(layer, n_neurons, activ, rest: bool = True):

            layer = tf.keras.layers.Dense(n_neurons)(layer) # 50  tf.keras.layers.LeakyReLU(0.2) 
            layer = tf.keras.layers.Dropout(0.1)(layer)

            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = activ(layer)
            
            if rest:
                layer_rest = tf.keras.layers.Dense(n_neurons)(layer) # 50  tf.keras.layers.LeakyReLU(0.2) 
                layer_rest = tf.keras.layers.Dropout(0.1)(layer)

                layer += layer_rest
                
                layer = tf.keras.layers.BatchNormalization()(layer)
                layer = activ(layer)
            
            return layer
        
        
        def conv(x, filters, length, expand: bool = False, flatten: bool = False, 
                 convl: bool = True, drout: bool = True, strides: int = 1, att:bool =False, padding='same'):
            if expand:
                compat_input = x = tf.expand_dims(x,2)
            else:
                compat_input = x
            
            if convl:
                x = tf.keras.layers.Conv1D(filters, length, strides=strides, padding=padding)(x)
                    
                # x = tf.split(x,10,axis=-1)
                # x = tf.concat([tf.reduce_sum(cannals,-1,keepdims=True) for cannals in x],-1)
                # print(x)
                
            x = tf.keras.layers.BatchNormalization()(x)
                        
            if att:
                x = x + tf.keras.layers.Attention()([x,x])
            
            if flatten:
                compat_input = x = tf.keras.layers.Flatten()(x)
            if drout:
                x = tf.keras.layers.Dropout(0.2)(x)
            return x
        
#         def add_norm_conv_block(*args, **kwargs):
#             inputx, x = conv(*args, **kwargs)
# #             print(x, tf.shape(x)[1])
# #             print(inputx, tf.shape(inputx)[1])
#             inputx = inputx[:,tf.shape(inputx)[1]-tf.shape(x)[1]:,:]
#             x += inputx
#             x = tf.keras.layers.BatchNormalization()(x)
#             return x

        
        activ=tf.keras.layers.LeakyReLU(0.1) # lambda x: 0.1*x+double_tanh(x) # tf.keras.layers.LeakyReLU(0.05)
        
        
        # @tf.function
        def sampling(mu,std):
            eps = tf.random.normal(tf.shape(mu), 0., 1.)
            std = tf.math.exp(std/2)
            return mu + eps*std
        
        
        history_input = tf.keras.Input(shape=(x_shape))
        current_state_input = tf.keras.Input((3))
        self.bid_ask_input = tf.keras.Input(2)
        
        layer = history_input
        layer = min_max(layer)
        
        layer = tf.keras.layers.BatchNormalization()(layer)
#         layer = tf.concat([layer, current_state_input],1)
        
#         KD = 1
#         NH = 3
        
#         att = tf.keras.layers.MultiHeadAttention(num_heads=NH, key_dim=KD)(layer[..., None],
#                                                                              layer[..., None])
#         layer += tf.keras.layers.Flatten()(att)
#         layer = tf.keras.layers.BatchNormalization()(layer)
        
        
        # layer = dense(layer, 50, activ, True)
        
        # layer = tf.keras.layers.Flatten()(layer)
        
#         layer = tf.keras.layers.BatchNormalization()(layer)
        
#         print(layer)
#         import sys;sys.exit(1)
              
        ### core predictive part start ###
            
            
        layer = conv(layer,20,4,strides=2,expand=True,flatten=False,att=False,convl=True)
        layer = conv(layer,50,4,strides=1,expand=False,flatten=False,att=False,convl=True)
        layer = conv(layer,50,4,strides=2,expand=False,flatten=False,att=False,convl=True)
        layer = conv(layer,50,4,strides=1,expand=False,flatten=False,att=False,convl=True)
        layer = conv(layer,50,4,strides=2,expand=False,flatten=True,att=False,convl=True)
        
# #         sys.exit(1)
#         layer = conv(layer,20,3,strides=1,expand=False,flatten=False,att=False,convl=True)
#         layer = conv(layer,30,4,strides=1,expand=False,flatten=False,att=False,convl=True)
#         layer = conv(layer,30,4,strides=1,expand=False,flatten=False,att=False,convl=True)

#         layer = conv(layer,30,4,strides=1,expand=False,flatten=False,att=False,convl=True)
    
#         layer += conv(layer,30,3,strides=1,expand=False,flatten=False,att=False,convl=True)
#         layer = tf.keras.layers.BatchNormalization()(layer)
#         layer = conv(layer,20,6,strides=2,expand=False,flatten=False,att=False,convl=True)
#         layer = tf.keras.layers.BatchNormalization()(layer)
#         layer = activ(layer)
        
#         layer = conv(layer,40,6,strides=2,expand=False,flatten=True,att=False,convl=True)
#         layer = tf.keras.layers.BatchNormalization()(layer)
#         layer = activ(layer)
#         layer += conv(layer,30,4,strides=1,expand=False,flatten=False,att=False,convl=True)
#         layer = tf.keras.layers.BatchNormalization()(layer)
        
#         layer = conv(layer,50,5,strides=1,expand=False,flatten=False,att=False,convl=True)
        
#         layer += conv(layer,50,5,strides=1,expand=False,flatten=False,att=False,convl=True)
#         layer = tf.keras.layers.BatchNormalization()(layer)
# #         layer += conv(layer,50,5,strides=1,expand=False,flatten=False,att=False,convl=True)
# #         layer = tf.keras.layers.BatchNormalization()(layer)
        
#         layer = conv(layer,30,5,strides=1,expand=False,flatten=True,att=False,convl=True)
        
        
#         layer = add_norm_conv_block(layer,40,5,strides=1,expand=True,flatten=False,att=False,convl=True)
#         layer = add_norm_conv_block(layer,20,3,strides=1,expand=False,flatten=False,att=False,convl=True)
#         layer = add_norm_conv_block(layer,20,5,strides=2,expand=False,flatten=True,att=False,convl=True)
        
        print(layer)
        
#         sys.exit(1)
        
#         layer = dense(layer, 30, activ, True)
#         layer = tf.random.normal(tf.shape(layer), layer, 0.001*layer)
        
#         for _ in range(8):
#             layer = dense(layer, 40, activ, True)
#             layer = tf.concat([layer, current_state_input],1)
        
        # layer += tf.keras.layers.MultiHeadAttention(num_heads=NH, key_dim=KD)(layer[..., None],
        #                                                                      layer[..., None])
        # layer = tf.keras.layers.BatchNormalization()(layer)
        
        # layer += tf.keras.layers.MultiHeadAttention(num_heads=NH, key_dim=KD)(layer[..., None],
        #                                                                      layer[..., None])
        # layer = tf.keras.layers.BatchNormalization()(layer)
        
        
#         NH = 3
        
#         att = tf.keras.layers.MultiHeadAttention(num_heads=NH, key_dim=KD)(layer[..., None],
#                                                                              layer[..., None])
#         layer += tf.keras.layers.Flatten()(att)
#         layer = tf.keras.layers.BatchNormalization()(layer)
        
        layer = tf.concat([layer, current_state_input],1)
        
        rest = False
#         for i in range(10,10,60)[::-1]:
#             layer = dense(layer, i, activ, rest)
        
        layer = dense(layer, 30, activ, rest)
        layer = tf.concat([layer, current_state_input],1)
        layer = dense(layer, 20, activ, rest)
#         layer = tf.concat([layer, current_state_input],1)
#         layer = dense(layer, 15, activ, rest)
        layer = tf.concat([layer, current_state_input],1)
        layer = dense(layer, 10, activ, rest)
                
        ### core predictive part end ###
        
        layer = tf.keras.layers.Dense(1)(layer)
        decisions = double_tanh(layer)
            
    
        self.decision_maker = tf.keras.Model(inputs = [history_input, current_state_input], 
                                             outputs = [decisions])

        self.decision_maker.compile(optimizer='rmsprop', 
                    loss ='mse',
                    metrics = ['mse'])
        
        # self.encoder =  tf.keras.Model(inputs = [history_input], 
        #                                      outputs = [sampled])
        
        # self.decision_maker.compile(optimizer='rmsprop', 
        #             loss ='mse',
        #             metrics = ['mse'])
        
        decision_vector = decisions
                
        self.bid_ask_input_estimated = tf.expand_dims(tf.concat([self.bid_ask_input, decision_vector[:,-1:]],1),0)
        self.bid_ask_input_estimated = tf.reshape(self.bid_ask_input_estimated, (-1,3))
        
        estimated_reward = ReturnLayer()(self.bid_ask_input_estimated)

        self.trainer = tf.keras.Model(inputs = [history_input, self.bid_ask_input, current_state_input], 
                                      outputs = [estimated_reward])

        self.trainer.compile(optimizer='rmsprop', 
                    loss =self.difference_loss,
                    metrics = [self.monitor_loss],)
        
    @staticmethod
    @tf.function
    def difference_loss(maximum_reward_output, model_reward_output):
        return (maximum_reward_output[-1] - model_reward_output)
    
    @staticmethod
    @tf.function
    def monitor_loss(maximum_reward_output, model_reward_output):
        return model_reward_output
    
    @staticmethod
    def continuous_round(x):
        return x + 1/tf.math.pi*( -rsin(x,1)+rsin(x,2)-rsin(x,3)+rsin(x,4)-rsin(x,5)+rsin(x,6)-rsin(x,7)+rsin(x,8)-rsin(x,9)+rsin(x,10) -rsin(x,11) +rsin(x,11) )
        
    # @tf.function
    def softmax_to_decisions(self, softmax_output):
        return self.continuous_round( tf.matmul(softmax_output, self.decision_matrix) )
        
                        
        
    def train_iteration(self, XV, BA, XVs, all_dec):

        with tf.GradientTape() as tape:
            pred = self.decision_maker([
                XVs, 
                all_dec])

            lines = tf.concat( tf.split( tf.round(pred)+1, 3), 1)
            lines = lines.numpy().astype(int)
            # lines = lines.astype(np.int8)
            # lines = tf.cast(lines, tf.int32)
            
            decs_ = get_decs(lines)
            
            decs = tf.one_hot(decs_[:-1],3)

            reward = self.trainer([XV, BA, decs])

            loss_value = 1/reward 

        # self.trainer.optimizer.lr.assign(lr)
        grads = tape.gradient(loss_value, self.decision_maker.trainable_weights)
        
        grads = [tf.clip_by_value(g, -1000, 1000) for g in grads]
        
        self.decision_maker.optimizer.apply_gradients(zip(grads, self.decision_maker.trainable_weights))
        decisions = tf.argmax(decs,1)-1
        
        return grads, decisions, reward, loss_value
    
    
    
    # grads, decisions, reward, loss_value = train_iteration(XV, BA, XVs, all_dec)