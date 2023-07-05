#!/usr/bin/env python3

import itertools
import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import numpy as np
import csv
import scipy.stats as stats
from pathos.multiprocessing import ProcessingPool as Pool
from bid_functions import *
import numba as nb

def sample_true_value(midpoint, uncertainty, size=1):
    lower = max(midpoint - uncertainty/2, 500)
    upper = min(midpoint + uncertainty/2, 9500)
    return stats.uniform.rvs(lower, upper - lower, size=size)


def sample_signals(true_values):
    return stats.uniform.rvs(true_values-500, 1000, (4, 5, len(true_values)))


def get_sufficient_stats(signal_samples, bidder_set):
    sufficient_stats = []
    for i in range(4):
        signals = signal_samples[i,:bidder_set[i],:]
        mins = np.min(signals, axis=0)
        maxs = np.max(signals, axis=0)
        # max_poss = np.minimum(mins+500, 9500)
        # min_poss = np.maximum(maxs-500, 500)
        # uncs = (max_poss - min_poss)
        # mids = (max_poss + min_poss)/2
        uncs = 1000 - (maxs - mins)
        mids = (maxs + mins)/2
        sufficient_stats.append(np.stack((mids, uncs), axis=-1))
    return np.array(sufficient_stats)


def get_sample_data(midpoint, unc, bidder_set, size):
    true_values = sample_true_value(midpoint, unc, size)
    signals = sample_signals(true_values)
    return true_values, get_sufficient_stats(signals, bidder_set)


def get_true_signals(midpoint, unc, size):
    true_values = sample_true_value(midpoint, unc, size)
    signals = sample_signals(true_values)
    return true_values, signals

# argsort == mathematica's ordering
# np.argot(np.argsort(list))


def convert_to_bids(bid_func, mids, uncs):
    if isinstance(bid_func,SingleSignalBidder):
        return bid_func[mids]
    return bid_func[uncs, mids]


def convert_grid_bids(bid_funcs, mids, uncs):
    return np.array([convert_to_bids(bid_funcs[i], mids[i], uncs[i]) for i in range(len(bid_funcs))])


'''
Added max_uncertainty on 2/8/23
 - Ryan
'''
def sample_rel_sufficient_stats(n_signals, n_samples=10000, max_uncertainty=1000):
    uncs = []
    for signal in n_signals:
        if signal == 1:
            uncs.append(np.full(n_samples, max_uncertainty))
        else:
            uncs.append(max_uncertainty - max_uncertainty*stats.beta.rvs(signal-1, 2, size=n_samples))
    uncs = np.array(uncs)
    rel_mids = stats.uniform.rvs(-uncs/2, uncs)
    return rel_mids, uncs


def plot_one_profit_comp(bidder_set, n_signals, midpoint, unc, b_id):
    true, signals = get_true_signals(midpoint, unc, 1000)

    signal_samples = get_sufficient_stats(signals, n_signals)
    bids = []
    for i in range(4):
        bids.append(convert_to_bids(bidder_set[i], signal_samples[i]))
    bids = np.array(bids)

    selections = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
    other_highest = np.max(bids[selections], axis=1)
    potential_bids = np.arange(midpoint-600, midpoint+600)
    avgs = [np.mean((other_highest[b_id] < bid)*(true-bid)) for bid in potential_bids]

    if isinstance(bidder_set[b_id], SingleSignalBidder):
        ga_bid = bidder_set[b_id][midpoint]
    else:
        ga_bid = bidder_set[b_id][unc, midpoint]
    plt.plot(potential_bids, avgs)
    plt.plot([ga_bid, ga_bid], [np.min(avgs), max(50,np.max(avgs))])
    plt.ylim([-10, max(50, np.max(avgs))])
    plt.show()


def calc_opts_m(m, mids, uncs, sample_rel_mids, sample_uncs, bidder_set, n_samples, max_uncertainty = 1000, value_bounds = [500, 9500]):
    print(mids[m])
    bids = np.zeros((4, len(uncs)))
    bid_profs = np.zeros_like(bids)
    opt_bids = np.zeros_like(bids)
    opt_profs = np.zeros_like(bids)
    n_signals = bidder_set.signals()

    for u in range(len(uncs)):
        mid = mids[m]
        unc = uncs[u]

        if mid + unc/2 < value_bounds[0] or mid - unc/2 > value_bounds[1]:
            bids[:,u] = 0
            bid_profs[:,u] = 0
            opt_bids[:,u] = 0
            opt_profs[:,u] = 0
            continue

        lower = max(mid - unc/2, value_bounds[0])
        upper = min(mid + unc/2, value_bounds[1])
        # Since true values are uniformly distributed, just make an evenly spread out array of them.
        sample_trues = lower + (upper-lower)*np.linspace(0,1,n_samples)
        sample_mids = sample_rel_mids + sample_trues
        sample_bids = convert_grid_bids(bidder_set, sample_mids, sample_uncs)

        selections = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
        other_highest = np.max(sample_bids[selections], axis=1)
        potential_bids = np.arange(max(value_bounds[0],mid-800), min(mid+600, value_bounds[1]))
        for b_id in range(4):
            if unc < max_uncertainty and n_signals[b_id] == 1:
                continue
            if isinstance(bidder_set[b_id], SingleSignalBidder):
                ga_bid = bidder_set[b_id][mid]
            else:
                ga_bid = bidder_set[b_id][unc, mid]

            avgs = [np.mean((other_highest[b_id] < bid)*(sample_trues-bid)) for bid in potential_bids]

            ga_prof = np.mean((other_highest[b_id] < ga_bid)*(sample_trues-ga_bid))
            opt_ind = np.argmax(avgs)
            opt_bid = potential_bids[opt_ind]
            opt_prof = avgs[opt_ind]

            bids[b_id, u] = ga_bid
            bid_profs[b_id, u] = ga_prof
            opt_bids[b_id, u] = opt_bid
            opt_profs[b_id, u] = opt_prof
    return bids, bid_profs, opt_bids, opt_profs

def calc_opts_m_numba(m, mids, uncs, sample_rel_mids, sample_uncs, bidder_set, n_samples, max_uncertainty = 1000, value_bounds = [500, 9500]):
    print(mids[m])
    bids = np.zeros((4, len(uncs)))
    bid_profs = np.zeros_like(bids)
    opt_bids = np.zeros_like(bids)
    opt_profs = np.zeros_like(bids)
    n_signals = bidder_set.signals()

    for u in range(len(uncs)):
        mid = mids[m]
        unc = uncs[u]

        if mid + unc/2 < value_bounds[0] or mid - unc/2 > value_bounds[1]:
            bids[:,u] = 0
            bid_profs[:,u] = 0
            opt_bids[:,u] = 0
            opt_profs[:,u] = 0
            continue

        lower = max(mid - unc/2, value_bounds[0])
        upper = min(mid + unc/2, value_bounds[1])
        # Since true values are uniformly distributed, just make an evenly spread out array of them.
        sample_trues = lower + (upper-lower)*np.linspace(0,1,n_samples)
        sample_mids = sample_rel_mids + sample_trues
        sample_bids = convert_grid_bids(bidder_set, sample_mids, sample_uncs)

        selections = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
        other_highest = np.max(sample_bids[selections], axis=1)
        potential_bids = np.arange(max(value_bounds[0],mid-800), min(mid+600, value_bounds[1]))
        for b_id in range(4):
            if unc < max_uncertainty and n_signals[b_id] == 1:
                continue
            if isinstance(bidder_set[b_id], SingleSignalBidder):
                ga_bid = bidder_set[b_id][mid]
            else:
                ga_bid = bidder_set[b_id][unc, mid]

            avgs = find_opt_bid(other_highest[b_id], sample_trues, potential_bids)

            ga_prof = np.mean((other_highest[b_id] < ga_bid)*(sample_trues-ga_bid))
            opt_ind = np.argmax(avgs)
            opt_bid = potential_bids[opt_ind]
            opt_prof = avgs[opt_ind]

            bids[b_id, u] = ga_bid
            bid_profs[b_id, u] = ga_prof
            opt_bids[b_id, u] = opt_bid
            opt_profs[b_id, u] = opt_prof
    return bids, bid_profs, opt_bids, opt_profs

'''
NEW PLAN: ALLOW LOWEST PROFIT BIDDER AT EACH MIDPOINT TO UPDATE FIRST WHEN RUNNING CALC_OPTS_M
UPDATE IF NEW BID IS LARGER THAN 2% CHANGE
THEN RUN CALC_BEST_RESPONSE_B_M FOR EACH BIDDER TO ALLOW THEM TO RESPOND INDEPENDENTLY.
'''
def calc_opts_iter_m_numba(m, mids, uncs, sample_rel_mids, sample_uncs, bidder_set, n_samples, max_uncertainty = 1000, value_bounds = [500, 9500], threshold = 0.02):
    print(mids[m])
    bids = np.zeros((4, len(uncs)))
    bid_profs = np.zeros_like(bids)
    opt_bids = np.zeros_like(bids)
    opt_profs = np.zeros_like(bids)
    n_signals = bidder_set.signals()

    for u in range(len(uncs)):
        mid = mids[m]
        unc = uncs[u]

        if mid + unc/2 < value_bounds[0] or mid - unc/2 > value_bounds[1]:
            bids[:,u] = 0
            bid_profs[:,u] = 0
            opt_bids[:,u] = 0
            opt_profs[:,u] = 0
            continue

        lower = max(mid - unc/2, value_bounds[0])
        upper = min(mid + unc/2, value_bounds[1])
        # Since true values are uniformly distributed, just make an evenly spread out array of them.
        sample_trues = lower + (upper-lower)*np.linspace(0,1,n_samples)
        sample_mids = sample_rel_mids + sample_trues
        sample_bids = convert_grid_bids(bidder_set, sample_mids, sample_uncs)

        selections = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
        other_highest = np.max(sample_bids[selections], axis=1)
        
        #order from highest to lowest so highest bid with same profit will be used.
        potential_bids = np.arange(min(mid+600, value_bounds[1]), max(value_bounds[0],mid-800), -1)
        for b_id in range(4):
            if unc < max_uncertainty and n_signals[b_id] == 1:
                continue
            if isinstance(bidder_set[b_id], SingleSignalBidder):
                ga_bid = bidder_set[b_id][mid]
            else:
                ga_bid = bidder_set[b_id][unc, mid]
            bids[b_id, u] = ga_bid
            bid_profs[b_id, u] = np.mean((other_highest[b_id] < ga_bid)*(sample_trues-ga_bid))

        for b_id in np.argsort(bid_profs[:,u]):

            avgs = find_opt_bid(other_highest[b_id], sample_trues, potential_bids)
            opt_ind = np.argmax(avgs)
            opt_bid = potential_bids[opt_ind]
            opt_prof = avgs[opt_ind]
            ga_bid = bids[b_id, u]

            if abs(opt_bid - ga_bid)/max_uncertainty > threshold:
                opt_bids[b_id, u] = opt_bid
                opt_profs[b_id, u] = opt_prof
            else:
                opt_bids[b_id, u] = ga_bid
                opt_profs[b_id, u] = bid_profs[b_id,u]

    return bids, bid_profs, opt_bids, opt_profs

@nb.njit(fastmath=True, parallel=True)
def calc_opts(mids, uncs, sample_rel_mids, sample_uncs, bidder_set, n_samples, max_uncertainty = 1000, value_bounds = [500, 9500]):
    
    bids = np.zeros((len(mids), 4, len(uncs)))
    bid_profs = np.zeros_like(bids)
    opt_bids = np.zeros_like(bids)
    opt_profs = np.zeros_like(bids)
    n_signals = bidder_set.signals()

    for m in nb.prange(len(mids)):
        for u in nb.prange(len(uncs)):
            mid = mids[m]
            unc = uncs[u]

            if mid + unc/2 < value_bounds[0] or mid - unc/2 > value_bounds[1]:
                bids[m,:,u] = 0
                bid_profs[m,:,u] = 0
                opt_bids[m,:,u] = 0
                opt_profs[m,:,u] = 0
                continue

            lower = max(mid - unc/2, value_bounds[0])
            upper = min(mid + unc/2, value_bounds[1])
            # Since true values are uniformly distributed, just make an evenly spread out array of them.
            sample_trues = lower + (upper-lower)*np.linspace(0,1,n_samples)
            sample_mids = sample_rel_mids + sample_trues
            sample_bids = convert_grid_bids(bidder_set, sample_mids, sample_uncs)

            selections = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
            other_highest = np.max(sample_bids[selections], axis=1)
            potential_bids = np.arange(max(value_bounds[0],mid-800), min(mid+600, value_bounds[1]))
            for b_id in nb.prange(4):
                if unc < max_uncertainty and n_signals[b_id] == 1:
                    continue
                if isinstance(bidder_set[b_id], SingleSignalBidder):
                    ga_bid = bidder_set[b_id][mid]
                else:
                    ga_bid = bidder_set[b_id][unc, mid]

                avgs = [np.mean((other_highest[b_id] < bid)*(sample_trues-bid)) for bid in potential_bids]

                ga_prof = np.mean((other_highest[b_id] < ga_bid)*(sample_trues-ga_bid))
                opt_ind = np.argmax(avgs)
                opt_bid = potential_bids[opt_ind]
                opt_prof = avgs[opt_ind]

                bids[m,b_id, u] = ga_bid
                bid_profs[m,b_id, u] = ga_prof
                opt_bids[m,b_id, u] = opt_bid
                opt_profs[m,b_id, u] = opt_prof
    return bids, bid_profs, opt_bids, opt_profs

'''
***ONLY UPDATES A BID FUNCTION IF THE MID AND UNC ARE DEFINED FOR THAT BIDDER.  WILL NOT UPDATE INTERPOLATED POINTS****
Attempts to find the best response bid at midpoint m for each bidder.  This process is done iteratively
starting with the bidder who's bid function deviates the most from the others (greatest MSE when compared to average of others).  Once this bidders best 
response bid function is created the next most deviating bidder can best respond. Finally, all of the original
and best response bid functions expected profits are calculated.
'''
def calc_opts_iterative_m(m, mids, uncs, sample_rel_mids, sample_uncs, bidder_set, n_samples, max_uncertainty = 1000, value_bounds = [500, 9500]):
    print(mids[m])
    bids = np.zeros((4, len(uncs)))
    bid_profs = np.zeros_like(bids)
    opt_bids = np.zeros_like(bids)
    opt_profs = np.zeros_like(bids)
    n_signals = bidder_set.signals()

    n_bidders = len(n_signals)
    selections = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]]) 
    init_bids = convert_grid_bids(bidder_set, [mids]*n_bidders, [np.repeat(np.transpose([uncs]),len(mids),axis=1)]*n_bidders)
    avg_opponent_bids = np.mean(init_bids[selections], axis=1)
    mse = ((init_bids-avg_opponent_bids)**2/3).reshape((n_bidders, len(mids)*len(uncs))).sum(axis=1)

    cur_bidder_set = BidderSet([b.clone() for b in bidder_set])
    for u in range(len(uncs)):
        mid = mids[m]
        unc = uncs[u]

        if mid + unc/2 < value_bounds[0] or mid - unc/2 > value_bounds[1]:
            bids[:,u] = 0
            bid_profs[:,u] = 0
            opt_bids[:,u] = 0
            opt_profs[:,u] = 0
            continue

        lower = max(mid - unc/2, value_bounds[0])
        upper = min(mid + unc/2, value_bounds[1])
        # Since true values are uniformly distributed, just make an evenly spread out array of them.
        sample_trues = lower + (upper-lower)*np.linspace(0,1,n_samples)
        sample_mids = sample_rel_mids + sample_trues
        sample_bids = convert_grid_bids(cur_bidder_set, sample_mids, sample_uncs)

        potential_bids = np.arange(max(value_bounds[0],mid-800), min(mid+600, value_bounds[1]))
        for b_id in mse.argsort():
            other_highest = np.max(sample_bids[selections], axis=1)
            if unc < max_uncertainty and n_signals[b_id] == 1:
                continue
            if isinstance(bidder_set[b_id], SingleSignalBidder):
                ga_bid = bidder_set[b_id][mid]
            else:
                ga_bid = bidder_set[b_id][unc, mid]

            avgs = [np.mean((other_highest[b_id] < bid)*(sample_trues-bid)) for bid in potential_bids]

            ga_prof = np.mean((other_highest[b_id] < ga_bid)*(sample_trues-ga_bid))
            opt_ind = np.argmax(avgs)
            opt_bid = potential_bids[opt_ind]
            opt_prof = avgs[opt_ind]

            bids[b_id, u] = ga_bid
            bid_profs[b_id, u] = ga_prof
            opt_bids[b_id, u] = opt_bid
            opt_profs[b_id, u] = opt_prof

            bidder_u_idx = np.where(cur_bidder_set[b_id].uncertainties == unc)[0]
            bidder_m_idx = np.where(cur_bidder_set[b_id].midpoints == mid)[0]
            if len(bidder_u_idx) > 0 and len(bidder_m_idx) > 0:
                cur_bidder_set[b_id].bids[bidder_u_idx[0]][bidder_m_idx[0]] = opt_bid
                sample_bids[b_id] = convert_to_bids(cur_bidder_set[b_id], sample_mids[b_id], sample_uncs[b_id])

    return bids, bid_profs, opt_bids, opt_profs

'''
Will calculate the expected profit of each player at midpoint mids[m] using montecarlo simulations.
'''
def calc_exp_prof_m(m, mids, uncs, sample_rel_mids, sample_uncs, bidder_set, n_samples, max_uncertainty = 1000, value_bounds = [500, 9500]):
    print(mids[m])
    bids = np.zeros((4, len(uncs)))
    bid_profs = np.zeros_like(bids)
    n_signals = bidder_set.signals()
    n_bidders = len(n_signals)
    selections = np.array([[i for i in range(n_bidders) if i != j] for j in range(n_bidders)])

    for u in range(len(uncs)):
        mid = mids[m]
        unc = uncs[u]

        if mid + unc/2 < value_bounds[0] or mid - unc/2 > value_bounds[1]:
            bids[:,u] = 0
            bid_profs[:,u] = 0
            continue

        lower = max(mid - unc/2, value_bounds[0])
        upper = min(mid + unc/2, value_bounds[1])
        # Since true values are uniformly distributed, just make an evenly spread out array of them.
        sample_trues = lower + (upper-lower)*np.linspace(0,1,n_samples)
        sample_mids = sample_rel_mids + sample_trues
        sample_bids = convert_grid_bids(bidder_set, sample_mids, sample_uncs)

        for b_id in range(len(n_signals)):
            other_highest = np.max(sample_bids[selections], axis=1)
            if unc < max_uncertainty and n_signals[b_id] == 1:
                continue
            if isinstance(bidder_set[b_id], SingleSignalBidder):
                ga_bid = bidder_set[b_id][mid]
            else:
                ga_bid = bidder_set[b_id][unc, mid]

            ga_prof = np.mean((other_highest[b_id] < ga_bid)*(sample_trues-ga_bid))

            bids[b_id, u] = ga_bid
            bid_profs[b_id, u] = ga_prof

    return bids, bid_profs

def calc_best_resp_b_m(b_id, m, mids, uncs, sample_rel_mids, sample_uncs, bidder_set, n_samples, max_uncertainty = 1000, value_bounds = [500, 9500]):
    print(f'Best Response {(b_id, mids[m])}')

    opt_bids = np.zeros(len(uncs))
    n_signals = bidder_set.signals()
    n_bidders = len(n_signals)
    selection = np.array([i for i in range(n_bidders) if i != b_id])

    for u in range(len(uncs)):
        unc = uncs[u]
        mid = mids[m]
            

        if mid + unc/2 < value_bounds[0] or mid - unc/2 > value_bounds[1]:
            opt_bids[u] = 0
            continue

        lower = max(mid - unc/2, value_bounds[0])
        upper = min(mid + unc/2, value_bounds[1])
        # Since true values are uniformly distributed, just make an evenly spread out array of them.
        sample_trues = lower + (upper-lower)*np.linspace(0,1,n_samples)
        sample_mids = sample_rel_mids + sample_trues
        sample_bids = convert_grid_bids(bidder_set, sample_mids, sample_uncs)

        potential_bids = np.arange(max(value_bounds[0],mid-800), min(mid+600, value_bounds[1]))

        # find optimal bid
        other_highest = np.max(sample_bids[selection], axis=0)
        if unc < max_uncertainty and n_signals[b_id] == 1:
            continue

        avgs = [np.mean((other_highest < bid)*(sample_trues-bid)) for bid in potential_bids]
        opt_ind = np.argmax(avgs)
        opt_bid = potential_bids[opt_ind]

        opt_bids[u] = opt_bid

    return opt_bids


@nb.njit(fastmath=True, parallel=True)
def find_opt_bid(other_highest, sample_trues, potential_bids):
    other_highest = np.ascontiguousarray(other_highest)
    sample_trues = np.ascontiguousarray(sample_trues)

    n = potential_bids.shape[0]
    s = other_highest.shape[0]
    avg = np.empty(n, dtype=other_highest.dtype)
    for i in nb.prange(n):
        t = 0.0
        for j in nb.prange(s):
            t += (other_highest[j] < potential_bids[i]) * (sample_trues[j] - potential_bids[i])
        avg[i] = t / s;
    return avg

'''
Best response for a single bidder at a single mid point using parallel numba compiled function.
'''
def calc_best_resp_b_m_2(b_id, m, mids, uncs, sample_rel_mids, sample_uncs, bidder_set, n_samples, max_uncertainty = 1000, value_bounds = [500, 9500], threshold = 0.02, best_resp_func = find_opt_bid):
    print(f'Best Response {(b_id, mids[m])}')

    opt_bids = np.zeros(len(uncs))
    opt_profs = np.zeros(len(uncs))
    n_signals = bidder_set.signals()
    n_bidders = len(n_signals)
    selection = np.array([i for i in range(n_bidders) if i != b_id])

    for u in range(len(uncs)):
        unc = uncs[u]
        mid = mids[m]

        if mid + unc/2 < value_bounds[0] or mid - unc/2 > value_bounds[1]:
            opt_bids[u] = 0
            opt_profs[u] = 0
            continue

        if unc < max_uncertainty and n_signals[b_id] == 1:
            continue

        lower = max(mid - unc/2, value_bounds[0])
        upper = min(mid + unc/2, value_bounds[1])
        # Since true values are uniformly distributed, just make an evenly spread out array of them.
        sample_trues = lower + (upper-lower)*np.linspace(0,1,n_samples)
        sample_mids = sample_rel_mids + sample_trues
        sample_bids = convert_grid_bids(bidder_set, sample_mids, sample_uncs)

        potential_bids = np.arange(min(mid+600, value_bounds[1]), max(value_bounds[0],mid-800), -1)

        # find optimal bid
        other_highest = np.max(sample_bids[selection], axis=0)
        
        avgs = best_resp_func(other_highest, sample_trues, potential_bids)
        opt_ind = np.argmax(avgs)
        opt_bid = potential_bids[opt_ind]
        opt_prof = avgs[opt_ind]

        if isinstance(bidder_set[b_id], SingleSignalBidder):
            ga_bid = bidder_set[b_id][mid]
        else:
            ga_bid = bidder_set[b_id][unc, mid]

        if abs(ga_bid - opt_bid) / max_uncertainty > threshold:
            opt_bids[u] = opt_bid
            opt_profs[u] = opt_prof
        else:
            opt_bids[u] = ga_bid
            opt_profs[u] = np.mean((other_highest < ga_bid)*(sample_trues-ga_bid))

    return opt_bids, opt_profs

'''
Best response for a single bidder at a single mid point using parallel numba compiled function.
The best response is only accepted if it improves the profit above the supplied threshold
'''
def calc_best_resp_b_m_threshold(b_id, m, mids, uncs, sample_rel_mids, sample_uncs, bidder_set, n_samples, profs, max_uncertainty = 1000, value_bounds = [500, 9500], threshold = 0.02):
    print(f'Best Response {(b_id, mids[m])}')

    opt_bids = np.zeros(len(uncs))
    opt_profs = np.zeros(len(uncs))
    n_signals = bidder_set.signals()
    n_bidders = len(n_signals)
    selection = np.array([i for i in range(n_bidders) if i != b_id])

    for u in range(len(uncs)):
        unc = uncs[u]
        mid = mids[m]

        if mid + unc/2 < value_bounds[0] or mid - unc/2 > value_bounds[1]:
            opt_bids[u] = 0
            opt_prof[u] = 0
            continue

        if unc < max_uncertainty and n_signals[b_id] == 1:
            continue

        lower = max(mid - unc/2, value_bounds[0])
        upper = min(mid + unc/2, value_bounds[1])
        # Since true values are uniformly distributed, just make an evenly spread out array of them.
        sample_trues = lower + (upper-lower)*np.linspace(0,1,n_samples)
        sample_mids = sample_rel_mids + sample_trues
        sample_bids = convert_grid_bids(bidder_set, sample_mids, sample_uncs)

        potential_bids = np.arange(max(value_bounds[0],mid-800), min(mid+600, value_bounds[1]))

        # find optimal bid
        other_highest = np.max(sample_bids[selection], axis=0)
        
        avgs = find_opt_bid(other_highest, sample_trues, potential_bids)
        opt_ind = np.argmax(avgs)
        opt_bid = potential_bids[opt_ind]
        opt_prof = avgs[opt_ind]
        base_prof = profs[m, b_id, u]

        opt_bids[u] = opt_bid
        opt_prof[u] = opt_prof

    return opt_bids, opt_prof

@nb.njit(fastmath=True)
def find_opt_bid_single(other_highest, sample_trues, potential_bids):
    other_highest = np.ascontiguousarray(other_highest)
    sample_trues = np.ascontiguousarray(sample_trues)

    n = potential_bids.shape[0]
    s = other_highest.shape[0]
    avg = np.empty(n, dtype=other_highest.dtype)
    for i in range(n):
        t = 0.0
        for j in range(s):
            t += (other_highest[j] < potential_bids[i]) * (sample_trues[j] - potential_bids[i])
        avg[i] = t / s;
    return avg

def best_response_bidders(bidder_set):
    n_mids = 121
    n_uncs = 61
    mids = np.linspace(0, 10000, n_mids)
    uncs = np.linspace(0, 1000, n_uncs)
    bids = np.zeros((4, n_mids, n_uncs))
    n_signals = bidder_set.signals()

    n_samples = 100000
    sample_rel_mids, sample_uncs = sample_rel_sufficient_stats(n_signals, n_samples)
    pool = Pool(4)
    bids, bid_profs, opt_bids, opt_profs = zip(*pool.map(lambda m: calc_opts_m(m, mids, uncs, sample_rel_mids, sample_uncs, bidder_set, n_samples), range(n_mids)))

    bids = np.array(bids)
    print(bids.shape)
    bid_profs = np.array(bid_profs)
    opt_bids = np.array(opt_bids)
    opt_profs = np.array(opt_profs)


    br_bidders = []
    for b_id in range(4):
        if n_signals[b_id] == 1:
            br_bidders.append(SingleSignalBidder(n_signals[b_id], mids, [1000], opt_bids[:,b_id,-1]))
        else:
            br_bidders.append(Bidder(n_signals[b_id], mids, uncs, opt_bids[:,b_id]))

    # plt.pcolormesh(uncs, mids, prof_diffs)
    # plt.colorbar()
    # plt.show()

    # plt.pcolormesh(uncs, mids, bid_diffs)
    # plt.colorbar()
    # plt.show()
    for b_id in range(4):
        print(opt_profs[:,b_id] - bid_profs[:,b_id])
        print(opt_bids[:,b_id] - bids[:,b_id])
        prof_diffs = np.maximum(0, opt_profs[:,b_id] - bid_profs[:,b_id])
        bid_diffs = opt_bids[:,b_id] - bids[:,b_id]
        plt.plot(mids, bids[:,b_id,-1], '-')
        plt.show()

        plt.plot(mids, opt_profs[:,b_id,-1], '-')
        plt.plot(mids, bid_profs[:,b_id,-1], '-')
        plt.show()

        plt.plot(mids, opt_bids[:,b_id,-1] - mids, '-')
        plt.plot(mids, bids[:,b_id,-1] - mids, '-')
        plt.show()

        plt.plot(mids, prof_diffs[:,-1], '-')
        plt.show()

        plt.plot(mids, bid_diffs[:,-1], '-')
        plt.show()
    return BidderSet(br_bidders)

def get_filenames(prefix=""):
    return pathlib.Path.cwd().glob(f"{prefix}*.csv")

def main():
    bidder_sets = BidderSets()
    filenames = get_filenames("..\\bid_funcs_")
    for filename in filenames:
        print(filename)
        bidder_sets.append(read_file(filename))

    print(len(bidder_sets.bid_sets))
    for bidder_set in [bidder_sets.select([1,1,1,1])]:
        print(bidder_set.signal_string())
        br_bids = best_response_bidders(bidder_set)
        # write_file(br_bids, 'br_bids_' + br_bids.signal_string() + '.csv')
    bidder_id = 3
    midpoint = 2000
    unc = 800

    plot_one_profit_comp(bidder_set, (1,1,1,1), midpoint, unc, bidder_id)


    # true, signals = get_true_signals(midpoint, unc, 100000)
    # signal_samples = get_sufficient_stats(signals, n_signals)
    # bids = []
    # for i in range(4):
    #     bids.append(convert_to_bids(bidder_set[i], signal_samples[i]))
    # bids = np.array(bids)

if __name__ == "__main__":
    main()
