"""Unified, lightweight Recurrence Mimicking Learning (RML) utilities.

This module is intentionally model-agnostic: you can plug in any encoder/decoder
as long as the decoder accepts [latent, prev_action_one_hot] and returns either:
  - scalar action values (shape [T, 1]) OR
  - action logits/probabilities (shape [T, K]).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf


def seed_everything(seed: int) -> None:
    """Best-effort seeding for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


@dataclass(frozen=True)
class ActionSpace:
    """Discrete action space with optional numeric values.

    For quasi-continuous actions, pass a dense grid of numeric values.
    """
    values: Sequence[float]
    names: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        vals = np.asarray(self.values, dtype=np.float32).reshape(-1)
        if vals.size < 2:
            raise ValueError("ActionSpace requires at least 2 actions.")
        object.__setattr__(self, "values", vals)
        object.__setattr__(self, "K", int(vals.size))
        object.__setattr__(self, "_values_tf", tf.constant(vals, dtype=tf.float32))

    def index_to_value(self, idx: tf.Tensor) -> tf.Tensor:
        return tf.gather(self._values_tf, idx)

    def value_to_index(self, value: tf.Tensor) -> tf.Tensor:
        """Nearest-neighbor mapping from value(s) to discrete action indices."""
        value = tf.cast(value, tf.float32)
        value = tf.squeeze(value, axis=-1) if value.shape.rank and value.shape.rank > 1 else value
        diffs = tf.abs(tf.expand_dims(value, axis=-1) - self._values_tf)
        return tf.argmin(diffs, axis=-1, output_type=tf.int32)

    def one_hot(self, idx: tf.Tensor, dtype: tf.DType = tf.float32) -> tf.Tensor:
        return tf.one_hot(idx, depth=self.K, dtype=dtype)


class DecisionRule:
    """Maps decoder outputs to action indices."""

    def __init__(
        self,
        action_space: ActionSpace,
        mode: str = "nearest",
        custom: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
        round_offset: Optional[int] = None,
    ) -> None:
        self.action_space = action_space
        self.mode = mode
        self.custom = custom
        self.round_offset = 0 if round_offset is None else int(round_offset)

    def pred_to_index(self, pred: tf.Tensor) -> tf.Tensor:
        if self.custom is not None:
            return self.custom(pred)
        if self.mode == "argmax":
            return tf.argmax(pred, axis=-1, output_type=tf.int32)
        if self.mode == "nearest":
            return self.action_space.value_to_index(pred)
        if self.mode == "round":
            pred = tf.squeeze(pred, axis=-1) if pred.shape.rank and pred.shape.rank > 1 else pred
            idx = tf.cast(tf.round(pred) + self.round_offset, tf.int32)
            return tf.clip_by_value(idx, 0, self.action_space.K - 1)
        raise ValueError(f"Unsupported decision rule mode: {self.mode}")


class RML:
    """Core RML wrapper: encoder + decoder + action space."""

    def __init__(
        self,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        action_space: ActionSpace,
        decision_rule: Optional[DecisionRule] = None,
        encoder_optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        decoder_optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        grad_clip: Optional[float] = 1000.0,
        dtype: tf.DType = tf.float32,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.action_space = action_space
        self.decision_rule = decision_rule or DecisionRule(action_space, mode="nearest")
        self.encoder_optimizer = encoder_optimizer or tf.keras.optimizers.Adam()
        self.decoder_optimizer = decoder_optimizer or tf.keras.optimizers.Adam()
        self.grad_clip = grad_clip
        self.dtype = dtype

    def _stack_decisions(self, z: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Tile latent vectors across all possible previous actions."""
        z = tf.convert_to_tensor(z, dtype=self.dtype)
        n = tf.shape(z)[0]
        eye = tf.eye(self.action_space.K, dtype=self.dtype)
        phi_actions = tf.repeat(eye, repeats=n, axis=0)
        rank = tf.rank(z)
        multiples = tf.concat([[self.action_space.K], tf.ones(rank - 1, dtype=tf.int32)], axis=0)
        stacked_z = tf.tile(z, multiples)
        return stacked_z, phi_actions

    def _resolve_initial_action(self, initial_action: Optional[Any]) -> int:
        if initial_action is None:
            return int(self.action_space.K // 2)
        if isinstance(initial_action, (int, np.integer)):
            return int(np.clip(initial_action, 0, self.action_space.K - 1))
        # Assume value
        val = tf.constant(initial_action, dtype=self.dtype)
        idx = self.action_space.value_to_index(val)
        return int(idx.numpy())  # single scalar

    def phi_processing(self, stacked_preds: tf.Tensor, initial_action: Optional[Any] = None) -> tf.Tensor:
        """Reconstruct decision path that mimics recurrence."""
        idx = self.decision_rule.pred_to_index(stacked_preds)
        idx = tf.reshape(idx, (self.action_space.K, -1))
        idx = tf.transpose(idx)  # [T, K]
        idx_np = idx.numpy().astype(int)
        start_idx = self._resolve_initial_action(initial_action)
        decs = [start_idx]
        for row in idx_np:
            decs.append(int(row[decs[-1]]))
        decs = np.asarray(decs[:-1], dtype=np.int32)
        return tf.one_hot(decs, depth=self.action_space.K, dtype=self.dtype)

    def recurrence_mimicking_forward(
        self,
        x: tf.Tensor,
        initial_action: Optional[Any] = None,
        return_details: bool = True,
    ) -> Dict[str, tf.Tensor]:
        """Two-pass RML forward: stacked pass -> phi -> final pass."""
        z = self.encoder(x)
        stacked_z, stacked_a = self._stack_decisions(z)
        stacked_preds = self.decoder([stacked_z, stacked_a])
        phi_seq = self.phi_processing(stacked_preds, initial_action=initial_action)
        final_raw = self.decoder([z, phi_seq])
        final_index = self.decision_rule.pred_to_index(final_raw)
        final_one_hot = tf.one_hot(final_index, depth=self.action_space.K, dtype=self.dtype)
        final_value = self.action_space.index_to_value(final_index)
        return {
            "z": z,
            "stacked_preds": stacked_preds,
            "phi_seq": phi_seq,
            "final_raw": final_raw,
            "final_index": final_index,
            "final_one_hot": final_one_hot,
            "final_value": final_value,
        }

    def train_step(
        self,
        x: tf.Tensor,
        context: Any,
        reward_fn: Callable[[Any, Dict[str, tf.Tensor]], tf.Tensor],
        initial_action: Optional[Any] = None,
    ) -> Dict[str, tf.Tensor]:
        """Single RML training step. reward_fn should return a scalar (or batch of scalars)."""
        with tf.GradientTape() as tape:
            out = self.recurrence_mimicking_forward(x, initial_action=initial_action, return_details=True)
            reward = reward_fn(context, out)
            reward = tf.reduce_mean(reward)
            loss = -reward

        enc_vars = self.encoder.trainable_weights
        dec_vars = self.decoder.trainable_weights
        grads = tape.gradient(loss, enc_vars + dec_vars)
        if self.grad_clip is not None:
            grads = [None if g is None else tf.clip_by_value(g, -self.grad_clip, self.grad_clip) for g in grads]
        enc_grads = grads[: len(enc_vars)]
        dec_grads = grads[len(enc_vars) :]
        self.encoder_optimizer.apply_gradients(zip(enc_grads, enc_vars))
        self.decoder_optimizer.apply_gradients(zip(dec_grads, dec_vars))
        return {"loss": loss, "reward": reward, "output": out}

    def predict_actions(
        self,
        x: tf.Tensor,
        initial_action: Optional[Any] = None,
        as_values: bool = True,
    ) -> tf.Tensor:
        out = self.recurrence_mimicking_forward(x, initial_action=initial_action, return_details=True)
        return out["final_value"] if as_values else out["final_index"]

    def set_lr(self, lr: float) -> None:
        """Update learning rate for both encoder and decoder optimizers."""
        lr = float(lr)
        self.encoder_optimizer.learning_rate.assign(lr)
        self.decoder_optimizer.learning_rate.assign(lr)


def build_mlp_encoder(
    input_dim: int,
    hidden: Sequence[int],
    activation: str = "relu",
    dropout: float = 0.0,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for units in hidden:
        x = tf.keras.layers.Dense(units, activation=activation)(x)
        if dropout and dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x)
    return tf.keras.Model(inputs=inputs, outputs=x, name="RML_Encoder")


def build_mlp_decoder(
    latent_dim: int,
    action_dim: int,
    hidden: Sequence[int],
    output_dim: int,
    activation: str = "relu",
    output_activation: Optional[str] = None,
) -> tf.keras.Model:
    z_in = tf.keras.Input(shape=(latent_dim,))
    prev_action_in = tf.keras.Input(shape=(action_dim,))
    x = tf.keras.layers.Concatenate(axis=1)([z_in, prev_action_in])
    for units in hidden:
        x = tf.keras.layers.Dense(units, activation=activation)(x)
    out = tf.keras.layers.Dense(output_dim, activation=output_activation)(x)
    return tf.keras.Model(inputs=[z_in, prev_action_in], outputs=out, name="RML_Decoder")
