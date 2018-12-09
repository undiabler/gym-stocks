import gym
from gym import error, spaces, utils
from gym.utils import seeding

import glob
import os
import random

import pandas as pd
import numpy as np

class StocksEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, datadir):
        self.startBalance = 10000

        self.comission = 0.25 / 100.
        # TODO: add random initial position

        self.balanceBase = self.startBalance

        self.posBalance = 0
        self.posPrice = 0

        self.prevAction = 0

        self.generators = []
        self.stockMarket = None

        for path in glob.glob(datadir + '/*.csv'):
            if not os.path.isfile(path):
                continue

            self.generators.append(WindowGenerator(path))

        # contains space of input data
        self.observation_space = spaces.Tuple((
            # position price, normalized
            spaces.Box(low=0, high=1., shape=(1,)),
            # total balance used for SHORT/LONG position
            spaces.Box(low=-1., high=1., shape=(1,)),
            # normalized price (ohlc) + volume + changed percent
            spaces.Box(low=0, high=1., shape=(5,1)),
        ))

        # action is float with self risk management
        #  -1.0 means SHORT with 100% of balance
        #  +1.0 means LONG with 100% of balance
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,))

        if len(self.generators) == 0:
            raise NameError('Invalid empty directory {}'.format(dirname))

    def step(self, action):
        # assert self.action_space.contains(action)

        raw, done = self.stockMarket.next()
        state = raw[['Open', 'High', 'Low', 'Close','Volume_Currency',]].values

        return state, 0, done, None

    def reset(self):
        self.stockMarket = random.choice(self.generators).GetNewTraderWindow()

        self.money = 1000000
        self.equity = 0

        state, _, done,_ = self.step(0.0)
        return state

    def _render(self, mode='human', close=False):
        pass


import pandas as pd
import datetime
from datetime import timedelta
from random import randint

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_finance import candlestick2_ohlc,volume_overlay2

class WindowGenerator(object):
    """docstring for WindowGenerator"""
    def __init__(self, file):
        # super(WindowGenerator, self).__init__()
        df = pd.read_csv(file, sep=',', header=1, index_col='unix', names=['unix', 'Open', 'High', 'Low', 'Close', 'Volume_BTC', 'Volume_Currency','Weighted_Price'])
        df = df.dropna()
        # df = df.fillna(method='bfill')
        df.index = pd.to_datetime(df.index)
        df = df.loc[datetime.date(year=2015,month=1,day=1):]
        # print(df.head(10))
        # TODO: add price pcnt to understand profit
        self.df_days = grouper(df,'D')
        self.df_hours = grouper(df,'H')
        self.df_mins = grouper(df,'15min')

        self.aviable_days = len(self.df_days)
    
    def getrandomsample(self, dbars=90, hbars=48, mbars=48, train=96):

        randomSeek = randint(0, len(self.df_days) - int(hbars/24) - 5)

        randomHoursSeek = datetime.timedelta(hours = randint(0,24))
        # 15 min bars in day
        randomMinutesSeek = datetime.timedelta(minutes = 15*randint(0,96))

        # select first days range
        days = self.df_days[randomSeek:randomSeek+dbars]
        lastday = days.tail(1).index.to_pydatetime()[0]+datetime.timedelta(hours=24)

        hours = self.df_hours[lastday:].head(hbars)
        lasthour = hours.tail(1).index.to_pydatetime()[0]+datetime.timedelta(hours=1)

        minutes = self.df_mins[lasthour:].head(mbars)
        lastmins = minutes.tail(1).index.to_pydatetime()[0]+datetime.timedelta(minutes=15)

        trainData = self.df_mins[lastmins:].head(train)

        return days,hours,minutes,trainData

    def GetNewTraderWindow(self, dbars=90, hbars=48, mbars=48, train=96):

        d,h,m,e = self.getrandomsample(dbars,hbars,mbars,train = 2*train)

        # add some random shift to prevent situation when traning ranges start every time from 00:00:00
        randShift = randint(0, train)

        mt = MixTraderWindow(d,h,m,e, dbars=90, hbars=48, mbars=48)
        mt.next(randShift)

        return mt


class MixTraderWindow(object):
    """docstring for MixWindowChart"""
    def __init__(self, df_days, df_hours, df_mins, tradeBars, dbars=90, hbars=48, mbars=48):
        # super(MixWindowChart, self).__init__()
        self.df_days = df_days
        self.df_hours = df_hours
        self.df_mins = df_mins
        self.tradeBars = tradeBars

        self.limitDays = dbars
        self.limitHours = hbars
        self.limitMins = mbars

    def next(self, shift=1):
        if len(self.tradeBars)==0:
            return [],True

        bar = self.tradeBars.head(shift)
        self.tradeBars = self.tradeBars.iloc[shift:]

        self.df_mins = self.df_mins.append(bar)
        return self.getRaw(),False


    def getRaw(self):
        hours = grouper(self.df_mins, 'H')
        # print(self.df_hours.tail(3))
        # print(hours)
        self.df_hours = self.df_hours.combine_first(hours)
        self.df_hours.update(hours)
        # print(self.df_hours.head(3))
        
        days = grouper(self.df_hours, 'D')
        self.df_days = self.df_days.combine_first(days)
        self.df_days.update(days)
        
        tmp = pd.concat([self.df_days.tail(self.limitDays),self.df_hours.tail(self.limitHours),self.df_mins.tail(self.limitMins)])
        return normalize(tmp)

    def plot(self):

        fig = plt.figure()
        self._ax = fig.add_subplot(2, 1, 1)
        self._av = fig.add_subplot(2, 1, 2)

        df = self.getRaw()

        xdate = df.index.tolist()
        self._ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
        self._ax.grid(True)
        for label in self._ax.xaxis.get_ticklabels():
            label.set_rotation(45)

        def mydate(x,pos):
            try:
                return xdate[int(x)]
            except IndexError:
                return ''

        self._ax.xaxis.set_major_formatter(ticker.FuncFormatter(mydate))

        self._a, self._b = candlestick2_ohlc(self._ax,df['Open'],df['High'],df['Low'],df['Close'],width=0.6)
        volume_overlay2(self._av,df['Close'],df['Volume_Currency'],width=0.6)
        
        def on_key(event):
            if event.key == 'right':
                # plt.clf()
                # ax.remove()
                self._ax.clear()
                self._av.clear()
                self._ax.grid(True)
                self._a.remove()
                self._b.remove()
                self.getnext()
                df = self.getRaw()
                self._a, self._b = candlestick2_ohlc(self._ax,df['Open'],df['High'],df['Low'],df['Close'],width=0.6)
                volume_overlay2(self._av,df['Close'],df['Volume_Currency'],width=0.6)
                plt.gcf().canvas.draw_idle()
                plt.draw()

            # print('you pressed', event.key, event.xdata, event.ydata)

        cid = fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()
        plt.draw()

        



def grouper(df,freq):
    resampled = df.groupby([pd.Grouper(freq=freq)])
    df_high = resampled['High'].max()
    df_low = resampled['Low'].min()
    df_open = resampled['Open'].first()
    df_close = resampled['Close'].last()
    df_volume = resampled['Volume_Currency'].sum()
    # print("range {} are loaded".format(freq))
    return pd.concat([df_open, df_high, df_low, df_close, df_volume], axis=1)

def normalize(df):

    normed = df
    
    maxValue = (df[['High','Low','Open','Close']].max(axis=1)).max()
    normed['High'] = normed['High'] / maxValue
    normed['Low'] = normed['Low'] / maxValue
    normed['Open'] = normed['Open'] / maxValue
    normed['Close'] = normed['Close'] / maxValue

    maxVolume = df['Volume_Currency'].max()
    normed['Volume_Currency'] = normed['Volume_Currency'] / maxVolume
    return normed