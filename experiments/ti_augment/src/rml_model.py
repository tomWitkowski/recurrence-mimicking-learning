import os
import random
import time
from typing import Iterable, List, Optional

import numpy as np
import tensorflow as tf


tf_float = tf.float32
tf_int = tf.int32


def _normalize_config(model_config) -> List[dict]:
    if not model_config:
        return []
    if isinstance(model_config, dict):
        return model_config.get("layers", [])
    if isinstance(model_config, Iterable):
        return list(model_config)
    raise TypeError(f"Unsupported model_config type: {type(model_config)}")


def _glu_activation(x):
    units = x.shape[-1]
    if units is None:
        raise ValueError("GLU activation requires known last dimension.")
    if units % 2 != 0:
        raise ValueError("GLU activation requires an even number of units.")
    a, b = tf.split(x, num_or_size_splits=2, axis=-1)
    return a * tf.sigmoid(b)


def _get_activation(name):
    if name is None:
        return None
    if name == "glu":
        return _glu_activation
    return tf.keras.activations.get(name)


def _adjust_units(units, activation):
    if activation == "glu":
        return units * 2
    return units


def _maybe_set_seed():
    seed_value = os.environ.get("train_log_name")
    if not seed_value:
        return
    try:
        seed_value = int(seed_value.split("_")[-1].split(".")[0])
    except ValueError:
        return
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


_maybe_set_seed()


class SharpeRatio(tf.keras.layers.Layer):
    @staticmethod
    def immediate_returns(inp):
        bid, ask, action = inp[:, :1], inp[:, 1:2], inp[:, -1:]
        dP_perc = (bid[1:] - bid[:-1]) / bid[:-1]
        dF = action[1:] - action[:-1]
        c = tf.reduce_mean((ask - bid) / bid)
        return dP_perc * action[:-1] - c * tf.abs(dF)

    def __init__(self):
        super().__init__()

    @tf.function
    def call(self, inp):
        imm = self.immediate_returns(inp)
        return tf.reduce_mean(imm) / (tf.math.reduce_std(imm)+0.00001)


class Encoder(tf.keras.Model):
    def __init__(self, x_shape, model_config=None):
        cfg = _normalize_config(model_config)
        inputs = tf.keras.Input(shape=(x_shape,))
        x = inputs
        for layer_cfg in cfg:
            if layer_cfg.get("type") != "dense":
                raise ValueError(f"Unsupported layer type: {layer_cfg.get('type')}")
            activation = layer_cfg.get("activation")
            units = _adjust_units(layer_cfg["units"], activation)
            x = tf.keras.layers.Dense(
                units,
                activation=_get_activation(activation),
            )(x)
            dropout = layer_cfg.get("dropout")
            if dropout:
                x = tf.keras.layers.Dropout(dropout)(x)
        outputs = x
        super().__init__(inputs=inputs, outputs=outputs, name="Encoder")
        self.compile(optimizer="Adam", loss="mse", metrics=["mse"])


class Decoder(tf.keras.Model):
    @staticmethod
    @tf.function
    def monitor_loss(_, model_reward_output):
        return model_reward_output

    def __init__(self, layer_shape, model_config=None, K: int = 3):
        cfg = _normalize_config(model_config)
        z_in = tf.keras.Input(shape=(layer_shape,))
        prev_action_in = tf.keras.Input((K,))
        x = tf.keras.layers.Concatenate(axis=1)([z_in, prev_action_in])
        for layer_cfg in cfg:
            if layer_cfg.get("type") != "dense":
                raise ValueError(f"Unsupported layer type: {layer_cfg.get('type')}")
            activation = layer_cfg.get("activation")
            units = _adjust_units(layer_cfg["units"], activation)
            x = tf.keras.layers.Dense(
                units,
                activation=_get_activation(activation),
            )(x)
            dropout = layer_cfg.get("dropout")
            if dropout:
                x = tf.keras.layers.Dropout(dropout)(x)
        if not cfg or not cfg[-1].get("output"):
            x = tf.keras.layers.Dense(1)(x)
            x = tf.keras.activations.tanh(x)
        super().__init__(inputs=[z_in, prev_action_in], outputs=x, name="Decoder")
        self.compile(optimizer="Adam", loss="mse", metrics=[self.monitor_loss])


class UtilityFunction(tf.keras.Model):
    def __init__(self):
        in_bid_ask = tf.keras.Input((2,))
        in_action = tf.keras.Input((1,))
        cat = tf.expand_dims(tf.concat([in_bid_ask, in_action[:, -1:]], 1), 0)
        cat = tf.reshape(cat, (-1, 3))
        rew = SharpeRatio()(cat)
        super().__init__(inputs=[in_bid_ask, in_action], outputs=rew, name="UtilityFunction")


class DummyUtilityFunction(tf.keras.Model):
    def __init__(self, K: int = 3):
        in_bid_ask = tf.keras.Input((2,))
        in_action = tf.keras.Input((K,))
        in_action_me = tf.reduce_mean(in_action, axis=1, keepdims=True)
        cat = tf.expand_dims(tf.concat([in_bid_ask, in_action_me[:, -1:]], 1), 0)
        cat = tf.reshape(cat, (-1, 3))
        rew = SharpeRatio()(cat)
        super().__init__(inputs=[in_bid_ask, in_action], outputs=rew, name="DummyUtilityFunction")


class Agent:
    def __init__(self, encoder=None, decoder=None, utility_function=None, input_len=None, K: int = 3):
        if encoder is None and decoder is None and utility_function is None:
            self.encoder = Encoder(input_len)
            self.decoder = Decoder(self.encoder.output.shape[1], K=K)
            self.utility_function = UtilityFunction() if K == 3 else DummyUtilityFunction(K=K)
        else:
            self.encoder = encoder if encoder is not None else Encoder(input_len)
            self.decoder = decoder if decoder is not None else Decoder(self.encoder.output.shape[1], K=K)
            if utility_function is None:
                utility_function = UtilityFunction() if K == 3 else DummyUtilityFunction(K=K)
            self.utility_function = utility_function
        self._phi_X = []
        self._phi_actions = []
        self.K = K

    def multiply_decisions(self, ZV):
        K = int(self.K)
        ZV = tf.convert_to_tensor(ZV)
        N = tf.shape(ZV)[0]
        if len(ZV) != int(len(self._phi_X) / K):
            eyeK = tf.eye(K, dtype=tf.float32)
            self._phi_actions = tf.repeat(eyeK, repeats=N, axis=0)
        multiples = [K] + [1] * (ZV.shape.rank - 1)
        self._phi_X = tf.tile(ZV, multiples)
        return self._phi_X, self._phi_actions

    @property
    def pred_to_range(self):
        return lambda pred: tf.round(pred) + 1

    def phi_processing(self, stacked_preds, initial_action=1):
        lines = tf.concat(tf.split(self.pred_to_range(stacked_preds), self.K), 1)
        lines_np = lines.numpy().astype(int)
        decs_ = [initial_action]
        for row in lines_np:
            decs_.append(row[decs_[-1]])
        return tf.one_hot(decs_[:-1], self.K)

    def recurrence_mimicking_forward_pass(self, XV):
        z_out = self.encoder(XV)
        stacked_z, stacked_a = self.multiply_decisions(z_out)
        stacked_preds = self.decoder([stacked_z, stacked_a])
        phi_seq = self.phi_processing(stacked_preds)
        final_dec = self.decoder([z_out, phi_seq])
        return final_dec

    def compute_apply_grads(self, tape, loss):
        grad_enc, grad_dec = tape.gradient(
            loss, [self.encoder.trainable_weights, self.decoder.trainable_weights]
        )
        grad_enc = [tf.clip_by_value(g, -1000, 1000) for g in grad_enc]
        grad_dec = [tf.clip_by_value(g, -1000, 1000) for g in grad_dec]
        self.encoder.optimizer.apply_gradients(zip(grad_enc, self.encoder.trainable_weights))
        self.decoder.optimizer.apply_gradients(zip(grad_dec, self.decoder.trainable_weights))
        return grad_enc, grad_dec

    def train_iteration(self, XV, BA):
        with tf.GradientTape() as tape:
            dec_seq = self.recurrence_mimicking_forward_pass(XV)
            reward = self.utility_function([BA, dec_seq])
            loss_value = -reward
        ge, gd = self.compute_apply_grads(tape, loss_value)
        dec_seq = np.round(dec_seq.numpy().reshape(-1,)).astype(int)
        return ge + gd, dec_seq, reward, loss_value

    def test_iteration(self, XV, BA, batch_size=10000, just_historical_path=False):
        z_out = self.encoder.predict(XV, batch_size=batch_size, verbose=0)
        if len(z_out.shape) == 1:
            z_out = np.expand_dims(z_out, 1)
        z_out_tf = tf.constant(z_out, tf_float)
        stacked_z, stacked_a = self.multiply_decisions(z_out_tf)
        stacked_preds = self.decoder([stacked_z, stacked_a])
        phi_seq = self.phi_processing(stacked_preds)
        if just_historical_path:
            return phi_seq[:-1]
        dec_seq = tf.argmax(phi_seq, 1) - 1
        reward = self.utility_function(
            [BA, self.decoder.predict([z_out, phi_seq], batch_size=batch_size)]
        )
        loss_value = 1 / reward
        return [], dec_seq, reward, loss_value, None

    def set_lr(self, lr: float):
        self.decoder.optimizer.lr.assign(lr)
        self.encoder.optimizer.lr.assign(lr)

    def fit(self, XV, BA, epochs, verbose=0):
        xv = XV.values
        ba = BA.values
        for _ in range(epochs):
            grads, decisions, rewards, loss_value = self.train_iteration(xv, ba)
            grads = [sum([x.numpy().reshape(-1,).tolist() for x in grads], [])]
            return_rate = np.round(np.mean(rewards), 5)
            if verbose > 0:
                print(return_rate)
        self.decisions = decisions
        return self
