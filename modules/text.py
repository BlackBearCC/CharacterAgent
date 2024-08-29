import os
import time
import numpy as np
import pandas as pd
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
# import tensorflow as tf
from datetime import datetime, timedelta

from scipy.signal import find_peaks
import logging

# 设置日志记录，并添加控制台输出
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler('trading_log.log'),  # 写入文件
                        logging.StreamHandler()  # 输出到控制台
                    ])

# Binance API 配置
api_key = "7XbBmjA1UxBzNBe0AriKyYlwt2HvOlNEzftJ9bN2g5kbUFACDKppATNlqGBtvlNE"
api_secret = "2BLZojVtSzDfyVgE1TW6U6MCSxDoDh5pnNZnz0BohEOGc7duHsT7mob2jf42ksOA"
client = Client(api_key, api_secret, testnet=True)



# 初始化参数
timestamp = 5  # 时间步长
initial_money = 10000
buy_amount = 1000
max_sell = 10
stop_loss = 0.03
take_profit = 0.07
trend_window = 14  # 预测窗口长度
minmax = MinMaxScaler()

cash = initial_money
current_inventory = 0
states_buy = []
states_sell = []
portfolio_value = [cash]
data_history = []
trades = []
predicted_prices = []
trend_list = []
market_states = []
future_prices_all = []
peaks_all = []
valleys_all = []
future_datetimes_all = []
processed_trades = set()

symbol = "BTCUSDT"  # 交易对

force_trade = True  # 设置为True以强制触发交易，方便测试
account_info = client.get_account()
print(f"账户信息：{account_info}")