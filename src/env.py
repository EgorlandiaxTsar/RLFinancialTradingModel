import random

import numpy as np
import pandas as pd


class MarketEnvProvider:
    def __init__(self, data: pd.DataFrame, initial_capital: int, position_size: float, commission: float, timestamps: int, env_size: int):
        self.data = data
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission = commission
        self.timestamps = timestamps
        self.env_size = env_size
        self.envs = []

    def get_envs_history(self):
        return self.envs

    def get_env(self):
        start = random.randint(0, len(self.data) - self.env_size - self.timestamps)
        df = self.data.iloc[start:(start + self.timestamps + self.env_size)].reset_index()
        env = MarketEnv(df, self.initial_capital, self.position_size, self.commission, self.timestamps)
        self.envs.append(env)
        return env


class MarketEnv:
    def __init__(self, data: pd.DataFrame, initial_capital: int, position_size: float, commission: float, timestamps: int):
        self.data = data.iloc[timestamps:].reset_index().drop(columns=['level_0', 'index']).to_numpy()
        self.data_starter = data.iloc[:timestamps].reset_index().drop(columns=['level_0', 'index']).to_numpy()
        self.position_size = position_size
        self.commission = commission
        self.timestamps = timestamps
        self.initial_capital, self.fiat_capital, self.asset_capital, self.total_capital, self.prev_total_capital = initial_capital, initial_capital, 0, initial_capital, initial_capital
        self.act_b, self.act_s, self.act_h = 2, 0, 1
        self.actions_history, self.trades_history, self.total_capital_history = [], [], []
        self.reward_info = {
            'entry_cap': -1,
            'entry_price': -1,
            'entry_prices': [],
            'stop_price': -1,
            'take_price': -1,
            'stop_percent': 0.004,
            'take_percent': 0.007
        }
        self.step = 0
        self.additional_info_len = 3

    def reset(self):
        self.fiat_capital, self.asset_capital, self.total_capital, self.prev_total_capital = self.initial_capital, 0, self.initial_capital, self.initial_capital
        self.act_b, self.act_s, self.act_h = 2, 0, 1
        self.actions_history, self.total_capital_history, self.trades_history = [], [], []
        self.step = 0
        self.reward_info['entry_cap'] = -1
        self.reward_info['entry_price'] = -1
        self.reward_info['entry_prices'] = []
        self.reward_info['stop_price'] = -1
        self.reward_info['take_price'] = -1
        return self.get_state()

    def get_history(self):
        return self.trades_history.copy()

    def get_data(self):
        return self.data_starter, self.data

    def get_additional_info_len(self):
        return self.additional_info_len

    def get_state(self):
        return self.is_active(), self.step, self.data[self.step], self.get_market_state(), self.fiat_capital, self.asset_capital, self.total_capital, self.prev_total_capital, (self.total_capital - self.prev_total_capital) / self.prev_total_capital

    def get_market_state(self):
        entry_price = self.reward_info['entry_price'] if self.reward_info['entry_price'] != -1 else 0
        stop_price = self.reward_info['stop_price'] if self.reward_info['stop_price'] != -1 else 0
        take_price = self.reward_info['take_price'] if self.reward_info['take_price'] != -1 else 0
        market_state = self.data_starter[-(self.timestamps - self.step):].tolist() + self.data[:self.step].tolist() if self.step < self.timestamps else self.data[(self.step - self.timestamps):self.step].tolist()
        additional_info = [entry_price, stop_price, take_price]
        for e in market_state:
            e += additional_info
        return np.array(market_state)

    def get_price(self):
        return self.data[self.step][3]

    def get_total_capital_history(self):
        return self.total_capital_history.copy()

    def is_active(self) -> bool:
        return self.step == len(self.data) - 1

    def forward(self, action: int):
        if action not in [self.act_b, self.act_s, self.act_h]:
            raise ValueError(f'Invalid action: {action}. Action must be one of {[self.act_b, self.act_s, self.act_h]}')
        if action == self.act_b:
            action, info = self._buy()
        elif action == self.act_s:
            action, info = self._sell()
        else:
            action, info = self._hold()
        done, step, price, observation, fiat_cap, asset_cap, total_cap, prev_total_cap, profit = self.get_state()
        reward = self._get_reward(action)
        self._update_env(action, info)
        return done, step, price, observation, fiat_cap, asset_cap, total_cap, prev_total_cap, profit, reward

    def _get_reward(self, action) -> float:
        # reward = 0
        # price = self.data[self.step][3]
        # entry_cap, entry, take, stop = self.reward_info['entry_cap'], self.reward_info['entry_price'], self.reward_info['take_price'], self.reward_info['stop_price']
        # if entry == -1:
        #     reward = 0 if (action == self.act_h or action == self.act_s) else 2.5
        # else:
        #     cap_gain_bonus = ((self.total_capital - entry_cap) / entry_cap) * 10
        #     if price >= take or price <= stop:
        #         if action == self.act_b:
        #             reward = -1
        #         else:
        #             reward = -0.2 + cap_gain_bonus if action == self.act_h else 2.5 + cap_gain_bonus
        #     else:
        #         reward = 0.2 + cap_gain_bonus if action == self.act_h else -0.2 + cap_gain_bonus
        # return reward
        # reward = 0
        
        
        # -----------
        
        
        # price = self.data[self.step][3]
        # entry_cap, entry, take, stop = self.reward_info['entry_cap'], self.reward_info['entry_price'], self.reward_info['take_price'], self.reward_info['stop_price']
        # if entry == -1:
        #     reward = 0 if (action == self.act_h or action == self.act_s) else 0.02
        # else:
        #     cap_gain_bonus = ((self.total_capital - entry_cap) / entry_cap) * 10
        #     take_bonus = ((price - take) / take) * 10
        #     stop_bonus = ((price - stop) / stop) * 10
        #     if price >= take or price <= stop:
        #         if self.total_capital > entry_cap:
        #             reward = take_bonus + cap_gain_bonus if action == self.act_s else -take_bonus * 1.5
        #         else:
        #             reward = abs(stop_bonus) * 1.5 if action == self.act_s else stop_bonus + cap_gain_bonus
        #     else:
        #         if self.total_capital > entry_cap:
        #             reward = abs(take_bonus) if action == self.act_h else take_bonus
        #             reward = reward if action != self.act_b else 0
        #         else:
        #            reward = stop_bonus if action == self.act_h else -stop_bonus
        #            reward = reward if action != self.act_b else 0
        
        
        # -----------
        
        
        # ((self.total_capital - entry_cap) / entry_cap) * 10 if entry_cap != -1 else 0
        
        
        # -----------
        entry_cap = self.reward_info['entry_cap']
        if action == self.act_b:
            reward = -0.05 if entry_cap != -1 else 0.05
        elif action == self.act_h:
            reward = 0
        else:
            reward = 0 if entry_cap == -1 else ((self.total_capital - entry_cap) / entry_cap) * 10
        return reward

    def _buy(self):
        qty = self.fiat_capital * self.position_size
        self.asset_capital += (qty / self.get_price())
        self.fiat_capital -= (qty + (qty * self.commission))
        return 2, {'side': 'b', 'qty_fiat': qty, 'qty_asset': qty / self.get_price(), 'price': self.get_price(), 'step': self.step}

    def _sell(self):
        qty = self.asset_capital * self.get_price()
        self.fiat_capital += (qty - qty * self.commission)
        self.asset_capital = 0
        return 0, {'side': 's', 'qty_fiat': qty, 'qty_asset': self.asset_capital, 'price': self.get_price(), 'step': self.step}

    def _hold(self):
        return 1, None

    def _update_env(self, act_p: int, act_description=None):
        price = self.get_price()
        teorical_asset_capital = self.asset_capital * price
        self.step += 1
        self.total_capital = self.fiat_capital + (teorical_asset_capital - teorical_asset_capital * self.commission)
        self.actions_history.append(act_p)
        self.total_capital_history.append(self.total_capital)
        if act_description != None and act_p == self.act_b:
            self.reward_info['entry_prices'].append({
                'qty': act_description['qty_asset'],
                'price': act_description['price']
            })
        if act_p == self.act_b:
            if self.reward_info['entry_cap'] == -1:
                self.reward_info['entry_cap'] = self.total_capital
                self.reward_info['entry_price'] = price
                self.reward_info['stop_price'] = price - price * self.reward_info['stop_percent']
                self.reward_info['take_price'] = price + price * self.reward_info['take_percent']
            else:
                medium_price = 0
                for e in self.reward_info['entry_prices']:
                    medium_price += e['price'] * e['qty']
                medium_price /= self.asset_capital
                self.reward_info['entry_price'] = medium_price
                self.reward_info['stop_price'] = medium_price - medium_price * self.reward_info['stop_percent']
                self.reward_info['take_price'] = medium_price + medium_price * self.reward_info['take_percent']
        elif act_p == self.act_s:
            self.reward_info['entry_cap'] = -1
            self.reward_info['entry_price'] = -1
            self.reward_info['entry_prices'] = []
            self.reward_info['stop_price'] = -1
            self.reward_info['take_price'] = -1
        if act_description != None:
            self.trades_history.append(act_description)
