#!/usr/bin/env python3

import itertools
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import pathlib
import numpy as np
import csv

def main():
    # filenames = get_filenames()
    # bidder_sets = BidderSets()
    # for filename in filenames:
    #     bidder_sets.append(read_file(filename))
    df = pd.read_csv('../processed_data/com_value_survey.csv')
    df = append_group_signals(df)
    df = append_model_bids(df)

def append_group_signals(df):
    print(df.groupby(['Session', 'GroupId', 'Period'])['SignalsPurchased'])

def append_model_bids(df):
    pass

def read_file(filename):
    bidders = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            signals = int(row[0])
            midpoints = np.array(next(reader),dtype=np.float64)
            if signals == 1:
                uncertainty = next(reader)
                bids = np.array(next(reader), dtype=np.float64)
                bidders.append(SingleSignalBidder(signals, midpoints, uncertainty, bids))
            else:
                uncertainties = np.array(next(reader), dtype=np.float64)
                bids = []
                for _ in uncertainties:
                    bids.append(np.array(next(reader), dtype=np.float64))
                bids = np.array(bids)
                bidders.append(Bidder(signals, midpoints, uncertainties, bids))
    return BidderSet(bidders)

class SingleSignalBidder:
    def __init__(self, signals, midpoints, uncertainty, bids):
        self.signals = signals
        self.midpoints = midpoints
        self.uncertainty = uncertainty
        self.bids = bids
        self.mid_max = midpoints[-1]
        self.mid_min = midpoints[0]
        self.mid_delta = (self.mid_max - self.mid_min)/(len(self.midpoints) - 1)

    def __getitem__(self, mid):
        return self.interpolate(mid)

    def interpolate(self, mid):
        mid = max(self.mid_min+0.001, min(mid, self.mid_max-0.001))
        ind_m,alpha_m = get_ind_alpha(mid, self.mid_min, self.mid_delta)
        return  (self.bids[ind_m]*(1-alpha_m)
                 + self.bids[ind_m+1]*(alpha_m))

    def valid_midpoints(self):
        return (self.midpoints+500 > 500) & (self.midpoints-500 < 9500)

class Bidder:
    def __init__(self, signals, midpoints, uncertainties, bids):
        self.signals = signals
        self.midpoints = midpoints
        self.uncertainties = uncertainties
        self.bids = bids
        self.mid_max = midpoints[-1]
        self.mid_min = midpoints[0]
        self.mid_delta = (self.mid_max - self.mid_min)/(len(self.midpoints) - 1)
        self.unc_max = uncertainties[-1]
        self.unc_min = uncertainties[0]
        self.unc_delta = (self.unc_max - self.unc_min)/(len(self.uncertainties) - 1)

    def __getitem__(self, pos):
        if type(pos) == tuple:
            if type(pos[0]) == slice:
                return self.midpoint_slice(pos[1])
            elif type(pos[1]) == slice:
                return self.uncertainty_slice(pos[0])
            else:
                return self.interpolate_point(pos[0], pos[1])
        else:
            return self.uncertainty_slice(pos)

    def valid_midpoints(self, unc):
        return (self.midpoints+unc/2 > 500) & (self.midpoints-unc/2 < 9500)

    def valid_uncertainties(self, mid):
        return (mid + self.uncertainties/2 > 500) & (mid-self.uncertainties/2 < 9500)

    def uncertainty_slice(self, unc):
        unc = max(self.unc_min+0.001, min(unc, self.unc_max-0.001))
        ind_u,alpha_u = get_ind_alpha(unc, self.unc_min, self.unc_delta)
        return self.bids[ind_u, :] * (1-alpha_u) + self.bids[ind_u+1,:]*alpha_u

    def midpoint_slice(self, mid):
        mid = max(self.mid_min+0.001, min(mid, self.mid_max-0.001))
        ind_m,alpha_m = get_ind_alpha(mid, self.mid_min, self.mid_delta)
        return self.bids[:, ind_m] * (1-alpha_m) + self.bids[:, ind_m+1]*alpha_m

    def interpolate_point(self, unc, mid):
        mid = max(self.mid_min+0.001, min(mid, self.mid_max-0.001))
        unc = max(self.unc_min+0.001, min(unc, self.unc_max-0.001))
        ind_u,alpha_u = get_ind_alpha(unc, self.unc_min, self.unc_delta)
        ind_m,alpha_m = get_ind_alpha(mid, self.mid_min, self.mid_delta)
        return  (self.bids[ind_u, ind_m]*(1-alpha_u)*(1-alpha_m)
                 + self.bids[ind_u, ind_m+1]*(1-alpha_u)*(alpha_m)
                 + self.bids[ind_u+1, ind_m]*(alpha_u)*(1-alpha_m)
                 + self.bids[ind_u+1, ind_m+1]*(alpha_u)*(alpha_m))

def get_ind_alpha(x, x_min, x_delta):
    ind = (x - x_min)/x_delta
    return int(ind), ind - int(ind)

class BidderSet:
    def __init__(self, bidders):
        self.bidders = bidders

    def __iter__(self):
        yield from self.bidders

    def signals(self):
        return [b.signals for b in self.bidders]

    def signal_string(self):
        return ''.join([str(x) for x in self.signals()])

    def __getitem__(self, i):
        return self.bidders[i]

class BidderSets:
    def __init__(self, bid_sets=None):
        if not bid_sets:
            bid_sets = []
        self.bid_sets = bid_sets

    def append(self, bid_set):
        self.bid_sets.append(bid_set)

    def __iter__(self):
        yield from self.bid_sets


    def select(self, signals):
        ss = sorted(signals)
        for bs in self.bid_sets:
            if sorted(bs.signals) == ss:
                return bs

def get_filenames():
    return pathlib.Path.cwd().glob("bid_funcs_*.csv")

if __name__ == "__main__":
    main()
