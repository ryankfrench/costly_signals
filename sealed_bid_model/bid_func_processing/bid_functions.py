
import csv
import numpy as np
import os
from numpy.core.numeric import zeros_like


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
                # for _ in midpoints:
                #     bids.append(np.array(next(reader), dtype=np.float64))
                # bids = np.array(bids)
                # bids = np.transpose(bids)
                bidders.append(Bidder(signals, midpoints, uncertainties, bids))
    return BidderSet(bidders)


def write_file(bid_set, filename):
    newline = '' if os.name == 'nt' else None
    with open(filename, 'w', newline=newline) as csvfile:
        writer = csv.writer(csvfile)
        for bidder in bid_set:
            writer.writerow([bidder.signals])
            writer.writerow(bidder.midpoints)
            if bidder.signals == 1:
                writer.writerow(bidder.uncertainty)
                writer.writerow(bidder.bids)
            else:
                writer.writerow(bidder.uncertainties)
                writer.writerows(bidder.bids)


def clean_bid_func(bidder_set, direction, thresh=200):
    dy,dx = direction
    rel_bids = bidder_set.bids - bidder_set.midpoints
    n_rows, n_cols = rel_bids.shape
    x_end = n_cols - 1
    y_end = n_rows - 1
    slopes = rel_bids - np.roll(rel_bids, direction, (0,1))
    # print(np.min(slopes), np.max(slopes))
    slopes = np.minimum(50,np.maximum(-50,slopes))
    predicted = np.roll(rel_bids + slopes, direction, (0,1))
    if dx == 1:
        predicted[:,:2] = rel_bids[:,:2]
    if dx == -1:
        predicted[:,-2:] = rel_bids[:,-2:]
    if dy == 1:
        predicted[:2,:] = rel_bids[:2,:]
    if dy == -1:
        predicted[-2:,:] = rel_bids[-2:,:]
    diff = (rel_bids - predicted)
    to_fix = (diff < -thresh)
    # print(np.histogram(diff))
    # print(to_fix.sum())
    bidder_set.bids = (to_fix * predicted + (1-to_fix)*rel_bids) + bidder_set.midpoints


class SingleSignalBidder:
    def __init__(self, signals, midpoints, uncertainty, bids):
        self.signals = signals
        self.midpoints = midpoints
        self.uncertainty = uncertainty
        self.bids = bids
        self.mid_max = midpoints[-1]
        self.mid_min = midpoints[0]
        self.mid_delta = (self.mid_max - self.mid_min)/(len(self.midpoints) - 1)

    def __getitem__(self, mids):
        return self.interpolate(mids)

    def interpolate(self, mids):
        mids = np.maximum(self.mid_min+0.001, np.minimum(mids, self.mid_max-0.001))
        ind_ms,alpha_ms = get_ind_alpha(mids, self.mid_min, self.mid_delta)
        return  (self.bids[ind_ms]*(1-alpha_ms)
                 + self.bids[ind_ms+1]*(alpha_ms))

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
        unc = np.maximum(self.unc_min+0.001, np.minimum(unc, self.unc_max-0.001))
        ind_u,alpha_u = get_ind_alpha(unc, self.unc_min, self.unc_delta)
        return self.bids[ind_u, :] * (1-alpha_u) + self.bids[ind_u+1,:]*alpha_u

    def midpoint_slice(self, mid):
        mid = max(self.mid_min+0.001, min(mid, self.mid_max-0.001))
        ind_m,alpha_m = get_ind_alpha(mid, self.mid_min, self.mid_delta)
        return self.bids[:, ind_m] * (1-alpha_m) + self.bids[:, ind_m+1]*alpha_m

    def interpolate_point(self, unc, mid):
        mid = np.maximum(self.mid_min+0.001, np.minimum(mid, self.mid_max-0.001))
        unc = np.maximum(self.unc_min+0.001, np.minimum(unc, self.unc_max-0.001))
        ind_u,alpha_u = get_ind_alpha(unc, self.unc_min, self.unc_delta)
        ind_m,alpha_m = get_ind_alpha(mid, self.mid_min, self.mid_delta)
        return  (self.bids[ind_u, ind_m]*(1-alpha_u)*(1-alpha_m)
                 + self.bids[ind_u, ind_m+1]*(1-alpha_u)*(alpha_m)
                 + self.bids[ind_u+1, ind_m]*(alpha_u)*(1-alpha_m)
                 + self.bids[ind_u+1, ind_m+1]*(alpha_u)*(alpha_m))

    def clone(self):
        return Bidder(self.signals, self.midpoints.copy(), self.uncertainties.copy(), self.bids.copy())

def get_ind_alpha(x, x_min, x_delta):
    ind = (x - x_min)/x_delta
    return ind.astype(int), ind - ind.astype(int)


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

    def __len__(self):
        return len(self.bidders)

    def get_by_signal(self, signal):
        return [bidder for bidder in self.bidders if bidder.signals == signal]


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
            if sorted(bs.signals()) == ss:
                return bs

