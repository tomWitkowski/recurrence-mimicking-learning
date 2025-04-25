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

for thr_env_name in ['MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[thr_env_name] = '1'

if 'train_log_name' not in os.environ:
    os.environ['train_log_name'] = 'train_log_0.log'
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
            eps = 0.0001
            return tf.reduce_mean(imm)/(tf.math.reduce_std(imm)+eps)
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
    def conv(x, filters, kernel_size, strides=1, padding='same', dropout=0.2, attention=False):
        x = tf.keras.layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if attention:
            x = x + tf.keras.layers.Attention()([x, x])
        x = tf.keras.layers.Dropout(dropout)(x)
        return x

    @staticmethod
    def lstm(x, units, return_sequences=False, dropout=0.2):
        x = tf.keras.layers.LSTM(units, return_sequences=return_sequences, dropout=dropout)(x)
        return x

    @staticmethod
    def self_attention(x):
        return x + tf.keras.layers.Attention()([x, x])

    @staticmethod
    def dense(x, units, activation='relu', dropout=0.2):
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        return x

    def __init__(self, x_shape, model_config):
        """
        x_shape: Input shape of the model.
        model_config: Dictionary defining model architecture.
        """
        # super().__init__()
        in_data = tf.keras.Input(shape=(x_shape))
        layer = in_data

        for layer_config in model_config:
            layer_type = layer_config['type']
            if layer_type == 'conv':
                layer = tf.expand_dims(layer, 2)
                layer = self.conv(
                    layer,
                    filters=layer_config.get('filters', 32),
                    kernel_size=layer_config.get('kernel_size', 3),
                    strides=layer_config.get('strides', 1),
                    padding=layer_config.get('padding', 'same'),
                    dropout=layer_config.get('dropout', 0.2),
                    attention=layer_config.get('attention', False),
                )
                layer = tf.keras.layers.Flatten()(layer)
            elif layer_type == 'lstm':
                layer = tf.expand_dims(layer, 2)
                layer = self.lstm(
                    layer,
                    units=layer_config.get('units', 64),
                    return_sequences=layer_config.get('return_sequences', False),
                    dropout=layer_config.get('dropout', 0.2),
                )
                layer = tf.keras.layers.Flatten()(layer)

            elif layer_type == 'self_attention':
                layer = tf.expand_dims(layer, 2)
                layer = self.self_attention(layer)
                layer = tf.keras.layers.Flatten()(layer)
            elif layer_type == 'dense':
                layer = self.dense(
                    layer,
                    units=layer_config.get('units', 128),
                    activation=layer_config.get('activation', 'relu'),
                    dropout=layer_config.get('dropout', 0.2),
                )
        # self.model = tf.keras.Model(inputs=[in_data], outputs=[layer])
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

    @staticmethod
    def conv(x, filters, kernel_size, strides=1, padding='same', dropout=0.2, attention=False):
        x = tf.keras.layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if attention:
            x = x + tf.keras.layers.Attention()([x, x])
        x = tf.keras.layers.Dropout(dropout)(x)
        return x

    @staticmethod
    def lstm(x, units, return_sequences=False, dropout=0.2):
        x = tf.keras.layers.LSTM(units, return_sequences=return_sequences, dropout=dropout)(x)
        return x

    @staticmethod
    def self_attention(x):
        return x + tf.keras.layers.Attention()([x, x])

    @staticmethod
    def dense(x, units, activation='relu', dropout=0.2):
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        return x

    def __init__(self, layer_shape, model_config: dict):
        # super().__init__()
        z_in = tf.keras.Input(shape=(layer_shape))
        prev_action_in = tf.keras.Input((3))
        layer = merged = tf.concat([z_in, prev_action_in], 1)

        for layer_config in model_config:
            layer_type = layer_config['type']
            if layer_type == 'conv':
                layer = self.conv(
                    layer,
                    filters=layer_config.get('filters', 16),
                    kernel_size=layer_config.get('kernel_size', 3),
                    strides=layer_config.get('strides', 1),
                    padding=layer_config.get('padding', 'same'),
                    dropout=layer_config.get('dropout', 0.2),
                    attention=layer_config.get('attention', False),
                )
            elif layer_type == 'lstm':
                layer = self.lstm(
                    layer,
                    units=layer_config.get('units', 16),
                    return_sequences=layer_config.get('return_sequences', False),
                    dropout=layer_config.get('dropout', 0.2),
                )
            elif layer_type == 'self_attention':
                layer = self.self_attention(layer)
            elif layer_type == 'dense':
                layer = self.dense(
                    layer,
                    units=layer_config.get('units', 16),
                    activation=layer_config.get('activation', 'relu'),
                    dropout=layer_config.get('dropout', 0.2),
                )

        out = tf.keras.layers.Dense(1)(layer)
        dec = tf.keras.activations.tanh(out)
        # dec = double_tanh(out)
        # self.model = tf.keras.Model(inputs=[z_in, prev_action_in], outputs=[dec])
        super().__init__(inputs=[z_in, prev_action_in], outputs=[dec])
        self.compile(optimizer='Adam', loss='mse', metrics=[self.monitor_loss])

class UtilityFunction(tf.keras.Model):
    """
    Global reward connection: merges (bid,ask) + action vector, returns final reward.
    """
    def __init__(self):
        # super().__init__()
        in_bid_ask = tf.keras.Input((2))
        in_action = tf.keras.Input((1))
        cat = tf.expand_dims(tf.concat([in_bid_ask, in_action[:,-1:]], 1), 0)
        cat = tf.reshape(cat, (-1,3))
        rew = SharpeRatio()(cat)
        # self.model = tf.keras.Model(inputs=[in_bid_ask, in_action], outputs=[rew])
        super().__init__(inputs=[in_bid_ask, in_action], outputs=[rew])

class Agent:
    """
    High-level API combining Encoder, Decoder, and UtilityFunction.
    Trains using offline RRL, online RRL, or RML.
    """
    def __init__(self, encoder=None, decoder=None, utility_function=None, input_len=None):
        if encoder is None:
            self.encoder = Encoder(input_len)
        else:
            self.encoder = encoder
        
        if decoder is None:
            self.decoder = Decoder(self.encoder.output.shape[1])
        else:
            self.decoder = decoder
        
        if utility_function is None:
            self.utility_function = UtilityFunction()
        else:
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

    def train_iteration(self, XV, BA):
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
        dec_seq = tf.argmax(phi_seq,1)-1
        return ge+gd, dec_seq, reward, loss_value

    def test_iteration(self, XV, BA, batch_size=10000,  just_historical_path=False):
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
        decoder_pred = self.decoder.predict([z_out, phi_seq], batch_size=batch_size, verbose=0)
        reward = self.utility_function([BA, tf.round(decoder_pred)])
        loss_value = 1/reward
        return [], dec_seq, reward, loss_value, decoder_pred

    def set_lr(self, lr: float):
        self.decoder.optimizer.lr.assign(lr)
        self.encoder.optimizer.lr.assign(lr)


    def fit(self, XV, BA, epochs, verbose=0):
        xv = XV.values
        ba = BA.values
        for e in range(epochs):
            grads, decisions, rewards, loss_value = self.train_iteration(xv, ba
                                                                            )
            grads = [sum([x.numpy().reshape(-1,).tolist() for x in grads],[])]

            return_rate = np.round(np.mean(rewards),5) # round((np.prod(rewards)-1)*100,3) if 'sharpe' == 'sharpe' else np.round(np.mean(rewards),5)
            avg_grads = np.mean(np.abs(sum(grads,[])))
            if verbose>0:
                print(return_rate)
        
        return self





class MADecoder(tf.keras.Model):
    """
    M_D: merges latent input + previous action to produce the next action.
    """
    @staticmethod
    @tf.function
    def monitor_loss(_, model_reward_output):
        return model_reward_output

    @staticmethod
    def conv(x, filters, kernel_size, strides=1, padding='same', dropout=0.2, attention=False):
        x = tf.keras.layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if attention:
            x = x + tf.keras.layers.Attention()([x, x])
        x = tf.keras.layers.Dropout(dropout)(x)
        return x

    @staticmethod
    def lstm(x, units, return_sequences=False, dropout=0.2):
        x = tf.keras.layers.LSTM(units, return_sequences=return_sequences, dropout=dropout)(x)
        return x

    @staticmethod
    def self_attention(x):
        return x + tf.keras.layers.Attention()([x, x])

    @staticmethod
    def dense(x, units, activation='relu', dropout=0.2):
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        return x

    def __init__(self, layer_shape, model_config: dict):
        # super().__init__()
        z_in = tf.keras.Input(shape=(layer_shape))
        prev_action_in = tf.keras.Input((3))
        ma_difference = tf.keras.Input((1))
        layer = merged = tf.concat([z_in, prev_action_in, ma_difference], 1)

        for layer_config in model_config:
            layer_type = layer_config['type']
            if layer_type == 'conv':
                layer = self.conv(
                    layer,
                    filters=layer_config.get('filters', 16),
                    kernel_size=layer_config.get('kernel_size', 3),
                    strides=layer_config.get('strides', 1),
                    padding=layer_config.get('padding', 'same'),
                    dropout=layer_config.get('dropout', 0.2),
                    attention=layer_config.get('attention', False),
                )
            elif layer_type == 'lstm':
                layer = self.lstm(
                    layer,
                    units=layer_config.get('units', 16),
                    return_sequences=layer_config.get('return_sequences', False),
                    dropout=layer_config.get('dropout', 0.2),
                )
            elif layer_type == 'self_attention':
                layer = self.self_attention(layer)
            elif layer_type == 'dense':
                layer = self.dense(
                    layer,
                    units=layer_config.get('units', 16),
                    activation=layer_config.get('activation', 'relu'),
                    dropout=layer_config.get('dropout', 0.2),
                )

        out = tf.keras.layers.Dense(1)(layer)
        dec = tf.keras.activations.tanh(out)
        # dec = double_tanh(out)
        # self.model = tf.keras.Model(inputs=[z_in, prev_action_in], outputs=[dec])
        super().__init__(inputs=[z_in, prev_action_in, ma_difference], outputs=[dec])
        self.compile(optimizer='Adam', loss='mse', metrics=[self.monitor_loss])



class MAgent:
    """
    High-level API combining Encoder, Decoder, and UtilityFunction.
    Trains using offline RRL, online RRL, or RML.
    """
    def __init__(self, encoder=None, decoder=None, utility_function=None, input_len=None):
        if encoder is None:
            self.encoder = Encoder(input_len)
        else:
            self.encoder = encoder
        
        if decoder is None:
            self.decoder = MADecoder(self.encoder.output.shape[1])
        else:
            self.decoder = decoder
        
        if utility_function is None:
            self.utility_function = UtilityFunction()
        else:
            self.utility_function = utility_function
        self._phi_X = []
        self._phi_actions = []

    def multiply_decisions(self, ZV, diff):
        if len(ZV) != int(len(self._phi_X)/3):
            zeros = tf.zeros((len(ZV),1))
            ones = tf.ones((len(ZV),1))
            self._phi_actions = tf.concat([
                tf.concat([ones, zeros, zeros],1),
                tf.concat([zeros, ones, zeros],1),
                tf.concat([zeros, zeros, ones],1)
            ], 0)
            self._phi_X = tf.concat([ZV, ZV, ZV], 0)
            self._phi_diff = tf.concat([diff, diff, diff], 0)
        return self._phi_X, self._phi_actions, self._phi_diff

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

    def train_iteration(self, XV, BA, diff):
        with tf.GradientTape() as tape:
            z_out = self.encoder(XV)
            stacked_z, stacked_a, stacked_diff = self.multiply_decisions(z_out, diff)
            stacked_preds = self.decoder([stacked_z, stacked_a, stacked_diff])
            phi_seq = self.phi_processing(stacked_preds)
            final_dec = self.decoder([z_out, phi_seq, diff])
            reward = self.utility_function([BA, final_dec])
            loss_value = -reward
        ge, gd = tape.gradient(loss_value, [self.encoder.trainable_weights, self.decoder.trainable_weights])
        ge = [tf.clip_by_value(g, -1000, 1000) for g in ge]
        gd = [tf.clip_by_value(g, -1000, 1000) for g in gd]
        self.encoder.optimizer.apply_gradients(zip(ge, self.encoder.trainable_weights))
        self.decoder.optimizer.apply_gradients(zip(gd, self.decoder.trainable_weights))
        dec_seq = tf.argmax(phi_seq,1)-1
        return ge+gd, dec_seq, reward, loss_value

    def test_iteration(self, XV, BA, diff, batch_size=10000,  just_historical_path=False):
        z_out = self.encoder.predict(XV, batch_size=batch_size, verbose=0)
        if len(z_out.shape) == 1:
            z_out = np.expand_dims(z_out, 1)
        z_out_tf = tf.constant(z_out, tf.float32)
        stacked_z, stacked_a, stacked_diff = self.multiply_decisions(z_out_tf, diff)
        stacked_preds = self.decoder([stacked_z, stacked_a, stacked_diff])
        phi_seq = self.phi_processing(stacked_preds)
        if just_historical_path:
            return phi_seq[:-1]
        dec_seq = tf.argmax(phi_seq,1)-1
        decoder_pred = self.decoder.predict([z_out, phi_seq,diff], batch_size=batch_size, verbose=0)
        reward = self.utility_function([BA, tf.round(decoder_pred)])
        loss_value = 1/reward
        return [], dec_seq, reward, loss_value, decoder_pred

    def set_lr(self, lr: float):
        self.decoder.optimizer.lr.assign(lr)
        self.encoder.optimizer.lr.assign(lr)


    def fit(self, XV, BA, diff, epochs, verbose=0):
        xv = XV.values
        ba = BA.values
        for e in range(epochs):
            grads, decisions, rewards, loss_value = self.train_iteration(xv, ba, diff
                                                                            )
            grads = [sum([x.numpy().reshape(-1,).tolist() for x in grads],[])]

            return_rate = np.round(np.mean(rewards),5) # round((np.prod(rewards)-1)*100,3) if 'sharpe' == 'sharpe' else np.round(np.mean(rewards),5)
            avg_grads = np.mean(np.abs(sum(grads,[])))
            if verbose>0:
                print(return_rate)
        
        return self