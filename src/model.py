"""Core model file defining Encoder, Decoder, UtilityFunction, and Agent classes."""

import sys, os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from config import Config as cfg
import tensorflow as tf
import numpy as np
import random
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

seed_value = int(os.environ['train_log_name'].split('_')[-1].split('.')[0])
print('SEED: ', seed_value)
time.sleep(10)
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

def double_tanh(x, a=5, b=2.5):
    return (tf.tanh(x*a+b) + tf.tanh(x*a-b))/2.

class SharpeRatio(tf.keras.layers.Layer):
    """
    Computes Sharpe Ratio or related measures given the final position and price data.
    """
    @staticmethod
    def immediate_returns(inp):
        B, A, F = inp[:, :1], inp[:, 1:2], inp[:, -1:]
        dP_perc = (B[1:] - B[:-1]) / B[:-1]
        dF = F[1:] - F[:-1]
        c = tf.reduce_mean((A - B) / B)
        return dP_perc * F[:-1] - c * tf.abs(dF)

    def __init__(self, lewar: int = cfg.lewar, with_swap: bool = False):
        super(SharpeRatio, self).__init__()
        self.lewar = lewar
        self.with_swap = with_swap
        self.type = cfg.reward_type

    @tf.function
    def call(self, inp):
        if self.type == 'custom':
            spreads_perc = tf.reduce_mean((inp[:,1] - inp[:,0]) / inp[:,0], 0)
            trans_cost = tf.math.pow(
                1 - spreads_perc*self.lewar/2,
                tf.reduce_sum(tf.abs(inp[1:, -1] - inp[:-1, -1]))
            )
            dA = inp[1:,1] - inp[:-1,1]
            add_ret = tf.reduce_sum(dA*inp[:-1,2]/inp[:-1,1]) * self.lewar
            general_ret = (1 + add_ret)*trans_cost
            if self.with_swap:
                point_perc = 0.00001/1.1
                minutes_per_day = 60*24
                long_swap_per_day = 7*point_perc
                n_pos = tf.reduce_sum(tf.abs(inp[:,2]))*self.lewar
                swap_factor = ((1-long_swap_per_day)**(1/minutes_per_day))**(n_pos)
                general_ret = general_ret * swap_factor
            return general_ret
        elif self.type == 'cumulative':
            return tf.reduce_sum(self.immediate_returns(inp), axis=1)
        elif 'sharpe' in self.type:
            imm = self.immediate_returns(inp)
            return tf.reduce_mean(imm)/tf.math.reduce_std(imm)
        else:
            raise NotImplementedError()

class Encoder(tf.keras.Model):
    """
    M_E: compresses the input into a latent representation.
    """
    @staticmethod
    def min_max(t):
        ma = tf.math.reduce_max(t, axis=1)
        mi = tf.math.reduce_min(t, axis=1)
        mi = tf.transpose(tf.reshape(tf.concat([mi]*t.shape[1], 0), (t.shape[1], -1)))
        ma = tf.transpose(tf.reshape(tf.concat([ma]*t.shape[1], 0), (t.shape[1], -1)))
        return t - mi

    @staticmethod
    def conv(x, filters, length, expand=False, flatten=False, convl=True, drout=True, strides=1, att=False, padding='same'):
        if expand:
            x = tf.expand_dims(x, 2)
        if convl:
            x = tf.keras.layers.Conv1D(filters, length, strides=strides, padding=padding)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if att:
            x = x + tf.keras.layers.Attention()([x, x])
        if flatten:
            x = tf.keras.layers.Flatten()(x)
        if drout:
            x = tf.keras.layers.Dropout(0.2)(x)
        return x

    def __init__(self, x_shape):
        super().__init__()
        in_data = tf.keras.Input(shape=(x_shape))
        layer = in_data
        if cfg.mode == 'technical':
            layer = tf.keras.layers.Dense(20)(layer)
            layer = tf.keras.layers.Dropout(0.1)(layer)
            layer = tf.keras.activations.tanh(layer)
            layer = tf.keras.layers.BatchNormalization()(layer)
            layer = tf.keras.layers.Dense(15)(layer)
            layer = tf.keras.layers.Dropout(0.1)(layer)
            layer = tf.keras.activations.tanh(layer)
            layer = tf.keras.layers.BatchNormalization()(layer)

        if cfg.mode == 'diff':
            layer = tf.expand_dims(layer, 2)
            layer = tf.keras.layers.Flatten()(layer)
        else:
            if cfg.mode not in ['technical','diff']:
                layer = self.min_max(layer)
                layer = tf.keras.layers.BatchNormalization()(layer)
                layer = self.conv(layer,16,5,expand=True,flatten=False)
                layer = tf.keras.layers.MaxPooling1D(2,2)(layer)
                layer = tf.keras.layers.BatchNormalization()(layer)
                layer = tf.keras.layers.Flatten()(layer)
                layer = tf.keras.layers.Dense(20)(layer)
                layer = tf.keras.layers.Dropout(0.1)(layer)
                layer = tf.keras.layers.BatchNormalization()(layer)
                layer = tf.keras.layers.LeakyReLU(0.1)(layer)
                z1 = tf.keras.layers.Dense(15)(layer)
                z2 = tf.keras.layers.Dense(15)(layer)
                layer = z1 - z2
                layer = tf.keras.layers.BatchNormalization()(layer)

        self.model = tf.keras.Model(inputs=[in_data], outputs=[layer])
        super().__init__(inputs=[in_data], outputs=[layer])
        self.compile(optimizer='Adam', loss='mse', metrics=['mse'])

class Decoder(tf.keras.Model):
    """
    M_D: merges latent input + previous action to produce the next action.
    """
    @staticmethod
    @tf.function
    def monitor_loss(_, model_reward_output):
        return model_reward_output

    def __init__(self, layer_shape, activation=tf.keras.layers.LeakyReLU(0.1)):
        super().__init__()
        z_in = tf.keras.Input(shape=(layer_shape))
        prev_action_in = tf.keras.Input((3))
        merged = tf.concat([z_in, prev_action_in], 1)
        out = tf.keras.layers.Dense(1)(merged)
        dec = tf.keras.activations.tanh(out)
        self.model = tf.keras.Model(inputs=[z_in, prev_action_in], outputs=[dec])
        super().__init__(inputs=[z_in, prev_action_in], outputs=[dec])
        self.compile(optimizer='Adam', loss='mse', metrics=[self.monitor_loss])

class UtilityFunction(tf.keras.Model):
    """
    Global reward connection: merges (bid,ask) + action vector, returns final reward.
    """
    def __init__(self):
        super().__init__()
        in_bid_ask = tf.keras.Input((2))
        in_action = tf.keras.Input((1))
        cat = tf.expand_dims(tf.concat([in_bid_ask, in_action[:,-1:]], 1), 0)
        cat = tf.reshape(cat, (-1,3))
        rew = SharpeRatio()(cat)
        self.model = tf.keras.Model(inputs=[in_bid_ask, in_action], outputs=[rew])
        super().__init__(inputs=[in_bid_ask, in_action], outputs=[rew])

class Agent:
    """
    High-level API combining Encoder, Decoder, and UtilityFunction.
    Trains using offline RRL, online RRL, or RML.
    """
    def __init__(self, encoder=None, decoder=None, utility_function=None, input_len=None):
        if None in [encoder, decoder, utility_function]:
            self.encoder = Encoder(input_len)
            self.decoder = Decoder(self.encoder.output.shape[1])
            self.utility_function = UtilityFunction()
        else:
            self.encoder = encoder
            self.decoder = decoder
            self.utility_function = utility_function
        self._phi_X = []
        self._phi_actions = []

    def multiply_decisions(self, ZV):
        if len(ZV) != int(len(self._phi_X)/3):
            zeros = tf.zeros((len(ZV),1))
            ones = tf.ones((len(ZV),1))
            self._phi_actions = tf.concat([
                tf.concat([ones, zeros, zeros],1),
                tf.concat([zeros, ones, zeros],1),
                tf.concat([zeros, zeros, ones],1)
            ], 0)
            self._phi_X = tf.concat([ZV, ZV, ZV], 0)
        return self._phi_X, self._phi_actions

    def phi_processing(self, stacked_preds):
        lines = tf.concat(tf.split(tf.round(stacked_preds)+1, 3), 1)
        lines_np = lines.numpy().astype(int)
        decs_ = [1]
        for row in lines_np:
            decs_.append(row[decs_[-1]])
        return tf.one_hot(decs_[:-1],3)

    def compute_apply_grads(self, tape, loss):
        grad_enc, grad_dec = tape.gradient(loss, [self.encoder.trainable_weights, self.decoder.trainable_weights])
        grad_enc = [tf.clip_by_value(g, -1000, 1000) for g in grad_enc]
        grad_dec = [tf.clip_by_value(g, -1000, 1000) for g in grad_dec]
        self.encoder.optimizer.apply_gradients(zip(grad_enc, self.encoder.trainable_weights))
        self.decoder.optimizer.apply_gradients(zip(grad_dec, self.decoder.trainable_weights))
        return grad_enc, grad_dec

    def train_iteration(self, XV, BA, offline=False, online_learning=False):
        if online_learning:
            dec_seq = tf.constant([0.])
            A, B, eta = 0., 0.01, 0.05
            for i, xrow in enumerate(XV[:-1]):
                with tf.GradientTape(persistent=True) as tape:
                    prev_a = tf.one_hot(tf.cast(tf.round(dec_seq[-1:,None]), tf.int32),3)[0,...]
                    new_a = self.decoder([self.encoder(xrow[None,:]), prev_a])
                    dec_seq = tf.concat([dec_seq, new_a[0]], axis=0)
                    rt = (BA[i+1,0]-BA[i,0]) / BA[i,0] * new_a
                    dA = rt - A
                    dB = rt**2 - B
                    if B - A**2 > 0:
                        Dt = (B*dA - 0.5*A*dB)/((B - A**2)**1.5)
                    else:
                        Dt = 0
                    loss_value = -Dt
                    A = A + eta*dA
                    B = B + eta*dB
                self.compute_apply_grads(tape, loss_value)
            dec_seq = dec_seq[:-1,None]
            reward = self.utility_function([BA[:len(dec_seq)], dec_seq])
            loss_value = -reward
        elif offline:
            with tf.GradientTape() as tape:
                dec_seq = tf.constant([0.])
                for xrow in XV:
                    prev_a = tf.one_hot(tf.cast(tf.round(dec_seq[-1:,None])+1.0, tf.int32),3)[0,...]
                    pred_a = self.decoder([self.encoder(xrow[None,:]), prev_a])
                    dec_seq = tf.concat([dec_seq, pred_a[0]], axis=0)
                dec_seq = dec_seq[:-1,None]
                reward = self.utility_function([BA, dec_seq])
                loss_value = -reward
            ge, gd = tape.gradient(loss_value, [self.encoder.trainable_weights, self.decoder.trainable_weights])
            ge = [tf.clip_by_value(g, -1000, 1000) for g in ge]
            gd = [tf.clip_by_value(g, -1000, 1000) for g in gd]
            self.encoder.optimizer.apply_gradients(zip(ge, self.encoder.trainable_weights))
            self.decoder.optimizer.apply_gradients(zip(gd, self.decoder.trainable_weights))
            dec_seq = np.round(dec_seq.numpy().reshape(-1,)).astype(int)
        else:
            with tf.GradientTape() as tape:
                z_out = self.encoder(XV)
                stacked_z, stacked_a = self.multiply_decisions(z_out)
                stacked_preds = self.decoder([stacked_z, stacked_a])
                phi_seq = self.phi_processing(stacked_preds)
                final_dec = self.decoder([z_out, phi_seq])
                reward = self.utility_function([BA, final_dec])
                loss_value = -reward
            ge, gd = tape.gradient(loss_value, [self.encoder.trainable_weights, self.decoder.trainable_weights])
            ge = [tf.clip_by_value(g, -1000, 1000) for g in ge]
            gd = [tf.clip_by_value(g, -1000, 1000) for g in gd]
            self.encoder.optimizer.apply_gradients(zip(ge, self.encoder.trainable_weights))
            self.decoder.optimizer.apply_gradients(zip(gd, self.decoder.trainable_weights))
            dec_seq = (tf.argmax(phi_seq,1)-1).numpy()
        return ge+gd, dec_seq, reward, loss_value

    def test_iteration(self, XV, BA, batch_size=10000, offline=False, just_historical_path=False):
        if offline:
            dec_seq = tf.constant([1.])
            for xrow in XV:
                prev_a = tf.one_hot(tf.cast(tf.round(dec_seq[-1:,None]), tf.int32),3)[0,...]
                pred_a = self.decoder([self.encoder(xrow[None,:]), prev_a])
                dec_seq = tf.concat([dec_seq, tf.cast(tf.argmax(pred_a,1), tf.float32)], axis=0)
            if just_historical_path:
                return dec_seq
        else:
            z_out = self.encoder.predict(XV, batch_size=batch_size, verbose=0)
            if len(z_out.shape) == 1:
                z_out = np.expand_dims(z_out, 1)
            z_out_tf = tf.constant(z_out, tf.float32)
            stacked_z, stacked_a = self.multiply_decisions(z_out_tf)
            stacked_preds = self.decoder([stacked_z, stacked_a])
            phi_seq = self.phi_processing(stacked_preds)
            if just_historical_path:
                return phi_seq[:-1]
            dec_seq = tf.argmax(phi_seq,1)-1
        reward = self.utility_function([BA, self.decoder.predict([z_out, phi_seq], batch_size=batch_size)])
        loss_value = 1/reward
        return [], dec_seq, reward, loss_value

    def set_lr(self, lr: float):
        self.decoder.optimizer.lr.assign(lr)
        self.encoder.optimizer.lr.assign(lr)
