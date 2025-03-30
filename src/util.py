import json
import os
import shutil

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


class DataManager:
    def __init__(self, loc):
        self.loc = loc
        self.buffer = self._load()

    def get_data(self):
        return self.buffer

    def get_config(self):
        return self.buffer['config']

    def get_updated_config(self, checkpoint_timeout=1):
        n_episodes = self.buffer['config']['episodes'] - len(self.buffer['runtimes']) * checkpoint_timeout
        config = self.buffer['config'].copy()
        config['episodes'] = 0 if n_episodes < 0 else n_episodes
        return config

    def get_weights_loc(self):
        loc = ''
        max_timestamp = 0
        for e in self.buffer['runtimes']:
            if int(e['timestamp']) > max_timestamp:
                max_timestamp = int(e['timestamp'])
                loc = e['file_loc']
        if max_timestamp == 0:
            return -1, -1
        return f'{loc}/actor.weights.h5', f'{loc}/critic.weights.h5'

    def load_data(self, data):
        self.buffer = data
        self._update()

    def load_config(self, config):
        for e in config:
            self.buffer['config'][e] = config[e]
        self._update()

    def load_runtime(self, timestamp: int, actor_loss: float, critic_loss: float, actor, critic, tot_rewards=[], capital_gains=[]):
        timestamp = int(timestamp)
        file_loc = self._create_checkpoint(actor, critic, timestamp)
        self.buffer['runtimes'].append({
            'timestamp': str(timestamp),
            'actor_loss': str(actor_loss),
            'critic_loss': str(critic_loss),
            'file_loc': str(file_loc),
            'tot_rewards': str(tot_rewards),
            'capital_gains': str(capital_gains)
        })
        self._update()

    def delete_runtimes(self):
        self.buffer['runtimes'] = []
        self._delete_checkpoints()
        self._update()

    def _create_checkpoint(self, actor, critic, timestamp):
        loc = f'cache/checkpoints/{timestamp}'
        os.mkdir(loc)
        actor.save_weights(f'{loc}/actor.weights.h5')
        critic.save_weights(f'{loc}/critic.weights.h5')
        return loc

    def _delete_checkpoints(self):
        loc = f'cache/checkpoints'
        for filename in os.listdir(loc):
            file_path = os.path.join(loc, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def _load(self):
        with open(self.loc, 'r') as file:
            data = json.load(file)
        return data

    def _update(self):
        if (os.path.exists(self.loc)):
            os.remove(self.loc)
        file = open(self.loc, 'x')
        data = json.dumps(self.buffer)
        file.write(data)
        file.close()


def rsi(source, length=14):
    return 100 - (100 / (1 + source['c'].diff(1).mask(source['c'].diff(1) < 0, 0).ewm(alpha=1 / length, adjust=False).mean() / source['c'].diff(1).mask(source['c'].diff(1) > 0, -0.0).abs().ewm(alpha=1 / length, adjust=False).mean()))


def ma(source, length=20):
    return source['c'].rolling(length).mean()


def atr(df, length=14):
    data = df.copy()
    high = data['h']
    low = data['l']
    close = data['c']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    return atr


def adx(df, length=14):
    data = df.copy()
    data['tr'] = np.maximum(data['h'] - data['l'], np.maximum(abs(data['h'] - data['c'].shift(1)), abs(data['l'] - data['c'].shift(1))))
    data['+dm'] = np.where((data['h'] - data['h'].shift(1)) > (data['l'].shift(1) - data['l']), np.maximum(data['h'] - data['h'].shift(1), 0), 0)
    data['-dm'] = np.where((data['l'].shift(1) - data['l']) > (data['h'] - data['h'].shift(1)), np.maximum(data['l'].shift(1) - data['l'], 0), 0)
    data['atr'] = data['tr'].rolling(window=length, min_periods=1).mean()
    data['+dm_smoothed'] = data['+dm'].rolling(window=length, min_periods=1).mean()
    data['-dm_smoothed'] = data['-dm'].rolling(window=length, min_periods=1).mean()
    data['+di'] = 100 * (data['+dm_smoothed'] / data['atr'])
    data['-di'] = 100 * (data['-dm_smoothed'] / data['atr'])
    data['dx'] = 100 * abs(data['+di'] - data['-di']) / (data['+di'] + data['-di'])
    data['adx'] = data['dx'].rolling(window=length, min_periods=1).mean()
    adx_data = data['adx']
    return adx_data


def macd(df, fast_ma_length=12, slow_ma_length=26, signal_length=9):
    data = pd.DataFrame()
    data['ma_fast'] = ma(df, fast_ma_length)
    data['ma_slow'] = ma(df, slow_ma_length)
    data['macd'] = data['ma_fast'] - data['ma_slow']
    data['signal'] = data['macd'].ewm(span=signal_length, min_periods=signal_length).mean()
    return data['macd']


def plot(candles, indicators=[], separated_indicators=[], plot_size=(20, 6), main_color='b'):
    labels = ['Close Price']
    plt.figure(figsize=plot_size)
    plt.plot(candles['c'].values, color=main_color)
    for indicator in indicators:
        plt.plot(indicator[2].values, color=indicator[0])
        labels.append(indicator[1])
    plt.legend(labels, loc='upper left')
    plt.grid(True)
    plt.show()
    for indicator in separated_indicators:
        label = [indicator[1]]
        val_min = indicator[2]
        val_max = indicator[3]
        plt.figure(figsize=plot_size)
        if val_min >= 0:
            plt.plot(val_min, color='w')
            label.insert(0, 'Min')
        if val_max > val_min:
            plt.plot(val_max, color='w')
            label.insert(1, 'Max')
        plt.plot(indicator[4].values, color=indicator[0])
        plt.legend(label, loc='upper left')
        plt.grid(True)
        plt.show()


def plot_trades(history, price_change, repeat: bool, plot_size=(20, 6), color='lightcoral'):
    plt.figure(figsize=plot_size)
    plt.plot([e[3] for e in price_change], color=color)
    plt.legend(['Price Change'], loc='upper left')
    prev_act = 'n'
    for i in range(len(history)):
        trade = history[i]
        color = 'g' if trade['side'] == 'b' else 'r'
        marker = '^' if color == 'g' else 'v'
        if prev_act != trade['side']:
            plt.scatter(trade['step'], trade['price'], color=color, marker=marker, s=80, alpha=1)
        elif not repeat:
            plt.scatter(trade['step'], trade['price'], color=color, marker=marker, s=80, alpha=1)
        prev_act = trade['side']
    plt.grid(True)
    plt.show()


def plot_rewards_distribution(rewards, actions, colors, zero_value_percent=0.001, zero_value_line_color='b'):
    actions_distribution = []
    step = 0
    limit = max([abs(min(rewards)), max(rewards)])
    limit += limit * 0.05
    zero_value = limit * zero_value_percent
    for reward, action in zip(rewards, actions):
        actions_distribution.append({
            'action': action,
            'step': step,
            'color': colors[action],
            'reward': reward if reward != 0 else zero_value
        })
        step += 1
    plt.figure(figsize=(20, 6))
    plt.axes().set_ylim([-limit, limit])
    plt.plot([zero_value for _ in range(len(rewards))], color=zero_value_line_color)
    for i in range(len(actions_distribution)):
        plt.bar(i, actions_distribution[i]['reward'], color=actions_distribution[i]['color'])
    plt.grid(True)
    plt.show()


def plot_probabilities_distribution(probs, colors):
    plt.figure(figsize=(20, 6))
    plt.axes().set_ylim([0, 1])
    index = 0
    for i in range(len(probs)):
        e = probs[i]
        for j in range(len(e)):
            plt.bar(index + j, e[j], color=colors[j])
        index += len(e)
    plt.grid(True)
    plt.show()
