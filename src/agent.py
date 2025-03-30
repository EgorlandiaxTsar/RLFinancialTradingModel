import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow import keras
from keras.optimizers import Adam
from env import MarketEnv
from model import TransformerModel


class Agent:
    def __init__(self, input_shape: tuple, n_actions: int, gamma: float, lr: float, gae_lambda: float, policy_clip: float, entropy_coef: float, batch_size: int, n_epochs: int, memory: AgentMemory):
        self.input_shape, self.n_actions = input_shape, n_actions
        self.lr, self.gamma, self.gae_lambda, self.policy_clip, self.entropy_coef, self.batch_size, self.n_epochs = lr, gamma, gae_lambda, policy_clip, entropy_coef, batch_size, n_epochs
        self.memory = memory
        self.actor, self.critic = self._build_models()

    def train(self, log=True):
        actor_losses, critic_losses = [], []
        for index_epoch in range(self.n_epochs):
            if log:
                print(f'epoch {index_epoch + 1} out of {self.n_epochs}')
            states_buf, actions_buf, old_probs_buf, values_buf, rewards_buf, dones_buf, batches = self.memory.generate_batches()
            advantage = np.zeros(len(rewards_buf), dtype=np.float32)
            for i in range(len(rewards_buf)):
                discount, i_advantage = 1, 0
                for k in range(i, len(rewards_buf) - 1):
                    i_advantage += discount * (rewards_buf[k] + self.gamma * values_buf[k + 1] * (1 - int(dones_buf[k])) - values_buf[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[i] = i_advantage
            index_batch = 0
            for batch in batches:
                with tf.GradientTape(persistent=True) as tape:
                    states, old_probs, actions = tf.convert_to_tensor(states_buf[batch]), tf.convert_to_tensor(old_probs_buf[batch]), tf.convert_to_tensor(actions_buf[batch])
                    probs = self.actor(np.array(states))
                    dist = tfp.distributions.Categorical(probs)
                    log_probs = dist.log_prob(actions)
                    entropy = dist.entropy()
                    critic_value = tf.squeeze(self.critic(np.array(states)), 1)
                    probs_ratio = tf.math.exp(log_probs - old_probs)
                    weighted_probs = advantage[batch] * probs_ratio
                    weighted_clipped_probs = tf.clip_by_value(probs_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]
                    actor_loss = -tf.math.minimum(weighted_probs, weighted_clipped_probs) - tf.reduce_mean(entropy) * self.entropy_coef
                    actor_loss = tf.math.reduce_mean(actor_loss)
                    critic_loss = keras.losses.MSE(critic_value, advantage[batch] + values_buf[batch])
                    actor_losses.append(actor_loss)
                    critic_losses.append(np.mean(critic_loss))
                actor_params = self.actor.trainable_variables
                actor_gradients = tape.gradient(actor_loss, actor_params)
                critic_params = self.critic.trainable_variables
                critic_gradients = tape.gradient(critic_loss, critic_params)
                self.actor.optimizer.apply_gradients(zip(actor_gradients, actor_params))
                self.critic.optimizer.apply_gradients(zip(critic_gradients, critic_params))
                if log:
                    bar_len, step = 25, index_batch + 1
                    percent = ('{0:.2f}').format(100 * (step / float(len(batches))))
                    filled_len = int(bar_len * step // len(batches))
                    bar = '=' * filled_len + '_' * (bar_len - filled_len)
                    sys.stdout.write(f'\r{step} [{bar}] {len(batches)}\t{percent}%\tactor_loss={float(actor_loss):.5f}\tcritic_loss={float(np.mean(critic_loss)):.5f}')
                    sys.stdout.flush()
                index_batch += 1
            print('')
        return actor_losses, critic_losses

    def go(self, env: MarketEnv, log=True):
        self.memory.clear_memory_if_max_capacity_reached()
        done, step, price, observation, fiat_cap, asset_cap, total_cap, prev_total_cap, profit = env.get_state()
        price_history, rewards, actions, probabilities = [], [], [], []
        interaction = 0
        while not done:
            action, value, log_prob, probs = self._choose_action(observation)
            done, step, price, observation, fiat_cap, asset_cap, total_cap, prev_total_cap, profit, reward = env.forward(action)
            self.memory.store_memory(observation, action, log_prob, value, reward, done)
            if log:
                log_reward = sum(rewards) if len(rewards) > 0 else 0
                bar_len, step = 25, interaction + 1
                percent = ('{0:.2f}').format(100 * (step / float(len(env.get_data()[1]))))
                filled_len = int(bar_len * step // (len(env.get_data()[1]) - 1))
                bar = '=' * filled_len + '_' * (bar_len - filled_len)
                sys.stdout.write(f'\r{step} [{bar}] {len(env.get_data()[1])}\t{percent}%\treward={float(log_reward):.4f}\ttotal_cap={float(total_cap):.4f}\taction={action}\tvalue={float(value.numpy()[0]):.6f}\tlog_prob={float(log_prob):.6f}\tprobs={np.round(probs[0], 6)}')
                sys.stdout.flush()
            price_history.append(price)
            rewards.append(reward)
            actions.append(action)
            probabilities.append(np.round(probs[0], 6).tolist())
            interaction += 1
        print('')
        return price_history, env.get_total_capital_history(), rewards, actions, probabilities

    def get_models(self):
        return self.actor, self.critic

    def inject_weights(self, actor_weights_loc, critic_weights_loc):
        self.actor.save_weights(actor_weights_loc)
        self.critic.save_weights(critic_weights_loc)

    def info(self):
        print('ACTOR MODEL')
        print(self.actor.summary())
        print('CRITIC MODEL')
        print(self.critic.summary())

    def _choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)
        action, value, log_prob = action.numpy()[0], value[0], log_prob.numpy()[0]
        return action, value, log_prob, probs

    def _build_models(self):
        actor, critic = TransformerModel(
            input_shape=self.input_shape,
            output_node_activation='softmax',
            n_output_nodes=3,
            n_head_nodes=128,
            n_heads=4,
            n_filters=8,
            n_transformer_blocks=4,
            n_mlp_units=[64],
            mlp_dropout_rate=0.5,
            base_dropout_rate=0.35
        ), self._build_default_model(1, 'linear')
        actor.compile(optimizer=Adam(learning_rate=self.lr))
        critic.compile(optimizer=Adam(learning_rate=self.lr))
        actor(tf.convert_to_tensor(np.zeros((1, self.input_shape[0], self.input_shape[1]))))
        critic(tf.convert_to_tensor(np.zeros((1, self.input_shape[0], self.input_shape[1]))))
        return actor, critic

    def _build_default_model(self, n_output_nodes, output_node_activation):
        return TransformerModel(
            input_shape=self.input_shape,
            output_node_activation=output_node_activation,
            n_output_nodes=n_output_nodes,
            n_head_nodes=256,
            n_heads=8,
            n_filters=16,
            n_transformer_blocks=8,
            n_mlp_units=[128],
            mlp_dropout_rate=0.5,
            base_dropout_rate=0.35
        )


class AgentMemory:
    def __init__(self, batch_size, max_capacity=-1):
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        self.max_capacity = max_capacity
        self.current_capacity = 0

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return np.array(self.states, dtype=np.float32), np.array(self.actions), np.array(self.probs, dtype=np.float32), np.array(self.values, dtype=np.float32), np.array(self.rewards, dtype=np.float32), np.array(self.dones), batches

    def store_memory(self, state, action, probs, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.current_capacity += 1

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear_memory_if_max_capacity_reached(self):
        if self.max_capacity > 0:
            if self.current_capacity >= self.max_capacity:
                self.clear_memory()
                self.current_capacity = 0
        else:
            self.clear_memory()
