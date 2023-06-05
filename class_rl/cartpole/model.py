import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal
import time
from class_rl.cartpole.utils import Buffer_rl


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


def get_model(observation_dimensions, steps_per_epoch, hidden_sizes, num_actions,
              policy_learning_rate, value_function_learning_rate):
    buffer = Buffer_rl(observation_dimensions, steps_per_epoch)
    # Initialize the actor and the critic as keras models
    observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
    logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
    actor = keras.Model(inputs=observation_input, outputs=logits)
    value = tf.squeeze(
        mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
    critic = keras.Model(inputs=observation_input, outputs=value)

    # Initialize the policy and the value function optimizers
    policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
    value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)
    render = False
    return actor, critic, render, buffer, policy_optimizer, value_optimizer
