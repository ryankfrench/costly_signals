#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pathlib
import numpy as np
import scipy.stats as stats
from bid_functions import *

def sample_true_value(midpoint, uncertainty, size=1):
    lower = max(midpoint - uncertainty/2, 500)
    upper = min(midpoint + uncertainty/2, 9500)
    return stats.uniform.rvs(lower, upper - lower, size=size)


def sample_signals(true_values):
    return stats.uniform.rvs(true_values-500, 1000, (4, 5, len(true_values)))


def get_sufficient_stats(signal_samples, signal_set):
    sufficient_stats = []
    for i in range(4):
        signals = signal_samples[i,:signal_set[i],:]
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


def get_sample_data(midpoint, unc, signal_set, size):
    true_values = sample_true_value(midpoint, unc, size)
    signals = sample_signals(true_values)
    return true_values, get_sufficient_stats(signals, signal_set)


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


def sample_rel_sufficient_stats(n_signals, n_samples=10000):
    uncs = []
    for signal in n_signals:
        if signal == 1:
            uncs.append(np.full(n_samples, 1000))
        else:
            uncs.append(1000 - 1000*stats.beta.rvs(signal-1, 2, size=n_samples))
    uncs = np.array(uncs)
    rel_mids = stats.uniform.rvs(-uncs/2, uncs)
    return rel_mids, uncs


def plot_one_profit_comp(signal_set, n_signals, midpoint, unc, b_id):
    true, signals = get_true_signals(midpoint, unc, 10)

    signal_samples = get_sufficient_stats(signals, n_signals)
    bids = []
    for i in range(4):
        bids.append(convert_to_bids(signal_set[i], signal_samples[i]))
    bids = np.array(bids)

    selections = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
    other_highest = np.max(bids[selections], axis=1)
    potential_bids = np.arange(midpoint-600, midpoint+600)
    avgs = [np.mean((other_highest[b_id] < bid)*(true-bid)) for bid in potential_bids]

    if isinstance(signal_set[b_id], SingleSignalBidder):
        ga_bid = signal_set[b_id][midpoint]
    else:
        ga_bid = signal_set[b_id][unc, midpoint]
    plt.plot(potential_bids, avgs)
    plt.plot([ga_bid, ga_bid], [np.min(avgs), max(50,np.max(avgs))])
    plt.ylim([-10, max(50, np.max(avgs))])
    plt.show()


def calc_opts_m(m, mids, uncs, sample_rel_mids, sample_uncs, signal_set, n_samples):
    print(mids[m])
    bids = np.zeros((4, len(uncs)))
    bid_profs = np.zeros_like(bids)
    opt_bids = np.zeros_like(bids)
    opt_profs = np.zeros_like(bids)
    n_signals = signal_set.signals()

    for u in range(len(uncs)):
        mid = mids[m]
        unc = uncs[u]

        if mid + unc/2 < 500 or mid - unc/2 > 9500:
            bids[:,u] = 0
            bid_profs[:,u] = 0
            opt_bids[:,u] = 0
            opt_profs[:,u] = 0
            continue

        lower = max(mid - unc/2, 500)
        upper = min(mid + unc/2, 9500)
        sample_trues =  + lower + (upper-lower)*np.linspace(0,1,n_samples)
        sample_mids = sample_rel_mids + sample_trues
        sample_bids=convert_grid_bids(signal_set, sample_mids, sample_uncs)

        selections = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
        other_highest = np.max(sample_bids[selections], axis=1)
        potential_bids = np.arange(mid-600, mid+600)
        for b_id in range(4):
            if unc < 1000 and n_signals[b_id] == 1:
                continue
            if isinstance(signal_set[b_id], SingleSignalBidder):
                ga_bid = signal_set[b_id][mid]
            else:
                ga_bid = signal_set[b_id][unc, mid]

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


def best_response_bidders(signal_set):
    n_mids = 121
    n_uncs = 61
    mids = np.linspace(0, 10000, n_mids)
    uncs = np.linspace(0, 1000, n_uncs)
    bids = np.zeros((4, n_mids, n_uncs))
    n_signals = signal_set.signals()

    n_samples = 100000
    sample_rel_mids, sample_uncs = sample_rel_sufficient_stats(n_signals, n_samples)
    bids, bid_profs, opt_bids, opt_profs = zip(*map(lambda m: calc_opts_m(m, mids, uncs, sample_rel_mids, sample_uncs, signal_set, n_samples), range(n_mids)))

    bids = np.array(bids)
    print(bids.shape)
    bid_profs = np.array(bid_profs)
    opt_bids = np.array(opt_bids)
    opt_profs = np.array(opt_profs)

    print(opt_profs[:,3] - bid_profs[:,3])
    print(opt_bids[:,3] - bids[:,3])
    prof_diffs = np.log(np.maximum(0,opt_profs[:,3] - bid_profs[:,3]) + 1)
    bid_diffs = np.log(np.abs(opt_bids[:,3] - bids[:,3]) + 1)

    br_bidders = []
    for b_id in range(4):
        if n_signals[b_id] == 1:
            br_bidders.append(SingleSignalBidder(n_signals[b_id], mids, [1000], opt_bids[:,b_id,-1]))
        else:
            br_bidders.append(Bidder(n_signals[b_id], mids, uncs, opt_bids[:,b_id]))
    return BidderSet(br_bidders)
    # plt.pcolormesh(uncs, mids, prof_diffs)
    # plt.colorbar()
    # plt.show()

    # plt.pcolormesh(uncs, mids, bid_diffs)
    # plt.colorbar()
    # plt.show()

def get_filenames():
    return pathlib.Path.cwd().glob("bid_funcs_*.csv")

def main():
    bidder_sets = BidderSets()
    filenames = get_filenames()
    for filename in filenames:
        bidder_sets.append(read_file(filename))

    for signal_set in bidder_sets:
        br_bids = best_response_bidders(signal_set)
        write_file(br_bids, 'br_bids_' + br_bids.signal_string() + '.csv')
    # bidder_id = 3
    # midpoint = 2000
    # unc = 800

    # plot_one_profit_comp(signal_set, n_signals, midpoint, unc, bidder_id)


    # true, signals = get_true_signals(midpoint, unc, 100000)
    # signal_samples = get_sufficient_stats(signals, n_signals)
    # bids = []
    # for i in range(4):
    #     bids.append(convert_to_bids(signal_set[i], signal_samples[i]))
    # bids = np.array(bids)

if __name__ == "__main__":
    main()
