''' 
Profit Anaylsis

# Use Monte Carlo to estimate the profits in the lower region.

1. Look at nash bidders first to get a baseline.
2. Allow 1 bidder to deviate use the final bid functions from the GA?
3. Allow all but 1 to deviate.
4. Use all GA strats
'''

import os
import csv
import numpy as np
import math
import importlib.util
import sys
from pathos.multiprocessing import ProcessingPool as Pool
import scipy.stats as stats
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
import sim_combiner
import bid_functions
import verify_bid_funcs
from celluloid import Camera
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation

class Counterfactual:
    def __init__(self, name, bidders, legend):
        self.name = name
        self.bidders = bidders
        self.legend = legend

def one_signal_nash(signal, bounds, max_uncertainty, n_bidders):
    epsilon = max_uncertainty / 2
    if ((bounds[0] - epsilon) <= signal) and (signal <= (bounds[0] + epsilon)):
        return bounds[0] + 1 / (n_bidders + 1) * (signal - bounds[0] + epsilon)
    elif ((bounds[1] - epsilon) <= signal) and (signal <= (bounds[1] + epsilon)):
        return (math.exp(-32 - 2*math.atan(9 - signal/1000) - 
                2*math.atan((-9000 + signal)/1000))*(201392000000000000000 * math.exp(32) + 
                1000000000000000 * math.exp(2 * math.atan(9 - signal/1000)) + 
                3000000000000000 * math.exp(32 + 2 * math.atan(9 - signal/1000)) - 
                89472000000000000 *math.exp(32) * signal + 14237000000000 * math.exp(32) * signal**2 - 
                875000000 *math.exp(32) * signal**3 + 4500 *  math.exp(32) * signal**4 + math.exp(32) * signal**5))/(5 * (-8000 + 
                signal)**2 * (82000000 - 18000 * signal + signal**2))
    else:
        return signal - epsilon + max_uncertainty / (n_bidders + 1) * math.exp(-n_bidders / max_uncertainty * (signal - (bounds[0] + epsilon)))

def read_ga_results(signal_counts, max_uncertainty):
    n_lines = [2 + (s>1) for s in signal_counts]
    n_tot_lines = sum(n_lines) + 1  # extra line for profits
    signal_str = '_'.join([str(s) for s in signal_counts])
    with open(f'F:/Dropbox (Chapman)/Costly Signals/sealed_bid_model/raw_data/common_value_{signal_str}_large.csv', 'r') as csvfile:
        data = csvfile.readlines()
        data = [line.split(',') for line in data]

    bid_dict = {i: [] for i in range(len(signal_counts))}
    for per_idx in range(0, len(data), n_tot_lines):
        cur_idx = per_idx
        for i in range(len(signal_counts)):
            next_idx = cur_idx + n_lines[i]
            bid_dict[i].append(sim_combiner.read_bid_func(data[cur_idx:next_idx], signal_counts[i], max_uncertainty))
            cur_idx = next_idx
        
        for i in range(len(signal_counts)):
            bid_dict[i][-1]["profit"] = float(data[cur_idx][i])
    
    return bid_dict

def calc_profits(m, mids, uncs, sample_rel_mids, sample_uncs, bidder_set, n_samples):
    #print(mids[m])
    bids = np.zeros((4, len(uncs)))
    bid_profs = np.zeros_like(bids)
    n_signals = bidder_set.signals()

    for u in range(len(uncs)):
        mid = mids[m]
        unc = uncs[u]

        if mid + unc/2 < 500 or mid - unc/2 > 9500:
            bids[:,u] = 0
            bid_profs[:,u] = 0
            continue

        lower = max(mid - unc/2, 500)
        upper = min(mid + unc/2, 9500)
        # Since true values are uniformly distributed, just make an evenly spread out array of them.
        sample_trues = lower + (upper-lower)*np.linspace(0,1,n_samples)
        sample_mids = sample_rel_mids + sample_trues
        sample_bids = verify_bid_functions.convert_grid_bids(bidder_set, sample_mids, sample_uncs)

        selections = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
        other_highest = np.max(sample_bids[selections], axis=1)
        for b_id in range(4):
            if unc < 1000 and n_signals[b_id] == 1:
                continue
            if isinstance(bidder_set[b_id], bid_functions.SingleSignalBidder):
                ga_bid = bidder_set[b_id][mid]
            else:
                ga_bid = bidder_set[b_id][unc, mid]

            ga_prof = np.mean((other_highest[b_id] < ga_bid)*(sample_trues-ga_bid))

            bids[b_id, u] = ga_bid
            bid_profs[b_id, u] = ga_prof
    return bids, bid_profs

def create_nash_bidders(signal_counts, max_uncertainty, mids):
    n = len(signal_counts)
    bids = np.array([one_signal_nash(s, [500,9500], max_uncertainty, n) for s in mids], dtype=np.float64)

    bidders = []
    for _ in range(n):
        bidders.append(bid_functions.SingleSignalBidder(1, mids, max_uncertainty, bids))

    return bid_functions.BidderSet(bidders)

def create_counterfactual_bidders(name, nash_bidders, idx_to_alt_strats, max_uncertainty):
    n = len(nash_bidders.signals())

    if len(idx_to_alt_strats) > n:
        raise Exception("More ga strategies provided than bidders")

    bidders = list(nash_bidders)
    legend = ["RNNE" for i in range(len(nash_bidders))]
    for idx in idx_to_alt_strats:
        strat = idx_to_alt_strats[idx]
        midpoints = strat['midpoints']
        bids = strat['bids'][0]
        bidders[idx] = bid_functions.SingleSignalBidder(1, midpoints, max_uncertainty, bids)
        legend[idx] = f'Bidder {idx + 1}'

    return Counterfactual(name, bid_functions.BidderSet(bidders), legend)

def profit_sim(mids, bidders):
    uncs = np.array([1000])

    n_samples = 100000
    sample_rel_mids, sample_uncs = verify_bid_funcs.sample_rel_sufficient_stats(bidders.signals(), n_samples)

    pool = Pool(4)
    bids, bid_profs = zip(*pool.map(lambda m: calc_profits(m, mids, uncs, sample_rel_mids, sample_uncs, bidders, n_samples), range(len(mids))))
    return bids, bid_profs

def write_profits(mids, bid_profs, signal_counts):
    with open('profits_' + '_'.join([str(s) for s in signal_counts]) + '.csv', 'w') as csvfile:
        csvfile.write(f'midpoint,min_prof,max_prof,avg_prof\n')
        for i in range(len(bid_profs)):
            csvfile.write(f'{mids[i]},')
            for p in range(len(bid_profs[i])):
                csvfile.write(f'{bid_profs[i][p]},')
            csvfile.write(f'{bid_profs[i].min()}, {bid_profs[i].max()}, {bid_profs[i].mean()}\n')

def single_signal_case():
    signal_counts = (1,1,1,1)
    max_uncertainty = 1000
    mids = np.linspace(0, 10000, 121)
    nash_bidders = create_nash_bidders(signal_counts, max_uncertainty, mids)

    bidder_sets = bid_functions.BidderSets()
    bidder_sets.append(Counterfactual("All RNNE", nash_bidders, ["RNNE" for i in range(4)]))

    bid_dict = read_ga_results(signal_counts, max_uncertainty)
    
    # single GA Bidder for all
    for i in range(4):
        bidder_sets.append(create_counterfactual_bidders(f'GA Bidder {i + 1}', nash_bidders, {i: bid_dict[i][-1]}, max_uncertainty))
    # two nash bidder (rest GA)
    bidder_sets.append(create_counterfactual_bidders(f'GA Bidders 1 - 2', nash_bidders, {i: bid_dict[i][-1] for i in range(2)}, max_uncertainty))
    # single nash bidder (rest GA)
    bidder_sets.append(create_counterfactual_bidders('GA Bidders 1 - 3', nash_bidders, {i: bid_dict[i][-1] for i in range(3)}, max_uncertainty))
    # all ga bidders
    bidder_sets.append(create_counterfactual_bidders('GA Bidders 1 - 4',nash_bidders, {i: bid_dict[i][-1] for i in range(4)}, max_uncertainty))
    # all GA pairs
    for b1 in range(3):
        for b2 in range(b1 + 1, 4):
            bidder_sets.append(create_counterfactual_bidders(f'GA Bidders {b1 + 1},{b2 + 1}', nash_bidders, {b1: bid_dict[b1][-1], b2: bid_dict[b2][-1]}, max_uncertainty))

    results = []
    for counterfactual in bidder_sets:
        results.append((counterfactual, [np.array(res) for res in profit_sim(mids, counterfactual.bidders)]))
    
    fig, axs = plt.subplots(4, 4)

    # first plot bids
    ax = axs[0][0]
    for p in range(len(signal_counts)):
        ax.plot(mids, results[7][1][0][:,p,0])
    ax.plot(mids, results[0][1][0][:,0,0], color='black')
    ax.title.set_text(f'GA Bids')
    ax.set_xlim([0,1500])
    ax.set_ylim([0,1500])
    ax.legend(['1','2','3','4','RNNE'])

    for i in range(1, len(results) + 1):
        ax = axs[i // 4][i % 4]
        for p in range(len(signal_counts)):
            ax.plot(mids, results[i - 1][1][1][:,p,0])
        ax.title.set_text(results[i - 1][0].name)
        ax.set_xlim([0,1500])
        ax.set_ylim([-20,100])
        ax.legend(results[i - 1][0].legend)
        
    plt.show()

def sim_profits(bidders, samples):
    true_values = np.random.randint(500,9501,samples)
    signals = [np.random.randint(x-500,x + 501, 4) for x in true_values]
    bids = np.array([[bidders[i][signal[i]] for i in range(len(bidders))] for signal in signals])
    max_bids = np.max(bids, axis=1)
    return np.mean(np.array([prof * (max_bid == bid) for (prof, max_bid, bid) in zip((true_values - max_bids), max_bids, bids)]), axis=0)

def simulate_nash_profits(samples):
    signal_counts = (1,1,1,1)
    max_uncertainty = 1000
    mids = np.linspace(0, 10000, 121)
    nash_bidders = create_nash_bidders(signal_counts, max_uncertainty, mids)

    return sim_profits(nash_bidders, samples)

def simulate_ga_profits(samples):
    signal_counts = (1,1,1,1)
    max_uncertainty = 1000
    mids = np.linspace(0, 10000, 121)
    bid_dict = read_ga_results(signal_counts, max_uncertainty)
    nash_bidders = create_nash_bidders(signal_counts, max_uncertainty, mids)
    ga_bidders = create_counterfactual_bidders('GA Bidders 1 - 4',nash_bidders, {i: bid_dict[i][-1] for i in range(4)}, max_uncertainty)

    return sim_profits(ga_bidders.bidders, samples)

def prob_mid_given_value(mid, value, uncertainty):
    if (mid <= value + uncertainty / 2) and (mid >= value - uncertainty / 2):
        return 1 / uncertainty
    else:
        return 0
    
def prob_value(value, val_range):
    if (value >= val_range[0]) and (value <= val_range[1]):
        return 1 / (val_range[1] - val_range[0])
    else:
        return 0
    
def prob_mid(mid, uncertainty, val_range):
    epsilon = uncertainty / 2
    return 1/uncertainty * 1 / (val_range[1] - val_range[0]) * (min(mid - epsilon, val_range[1]) - max(mid + epsilon, val_range[0]))

def expected_profit_m(m, mids, uncs, sample_rel_mids, sample_uncs, bidder_set, n_samples, max_uncertainty = 1000, value_bounds = [500, 9500]):
    print(mids[m])
    bids = np.zeros((4, len(uncs)))
    bid_profs = np.zeros_like(bids)
    n_signals = bidder_set.signals()

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
        sample_bids = verify_bid_funcs.convert_grid_bids(bidder_set, sample_mids, sample_uncs)

        # remove bidder so we can use the defined mid instead of the randomly drawn mid
        selections = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
        other_highest = np.max(sample_bids[selections], axis=1)

        for b_id in range(4):
            if unc < max_uncertainty and n_signals[b_id] == 1:
                continue
            if isinstance(bidder_set[b_id], bid_functions.SingleSignalBidder):
                ga_bid = bidder_set[b_id][mid]
            else:
                ga_bid = bidder_set[b_id][unc, mid]

            ga_prof = np.mean((other_highest[b_id] < ga_bid)*(sample_trues-ga_bid))

            bids[b_id, u] = ga_bid
            bid_profs[b_id, u] = ga_prof
    return bids, bid_profs

def nash_expected_profits():
    max_uncertainty = 1000
    value_bounds = (500, 9500)
    n_mids = 1001
    n_uncs = 101
    mids = np.linspace(value_bounds[0] - max_uncertainty/2, value_bounds[1] + max_uncertainty/2, n_mids)
    uncs = np.linspace(0, max_uncertainty, n_uncs)
    bids = np.zeros((4, n_mids, n_uncs))
    nash_bidders = create_nash_bidders((1,1,1,1), max_uncertainty, mids)

    bids, bid_profs = expected_profits(mids, uncs, nash_bidders, max_uncertainty, value_bounds, 100000)
    plot_exp_profit(mids, uncs, bids, bid_profs)


def ga_expected_profits(signal_count, n_samples, uncs = None, mids = None):
    max_uncertainty = 1000
    value_bounds = (500, 9500)

    if uncs is None:
        n_uncs = 61
        uncs = np.linspace(0, max_uncertainty, n_uncs)

    if mids is None:
        n_mids = 1001
        mids = np.linspace(value_bounds[0] - max_uncertainty/2, value_bounds[1] + max_uncertainty/2, n_mids)

    file_name = f'F:/Dropbox (Chapman)/Costly Signals/sealed_bid_model/bid_funcs_{"".join([str(i) for i in signal_count])}.csv'
    bidders_set = bid_functions.read_file(file_name)

    bids, bid_profs = expected_profits(mids, uncs, bidders_set, max_uncertainty, value_bounds, n_samples)
    #plot_results_3d_save(mids, uncs, bids, bid_profs, signal_count)
    #plot_exp_profit_3d(mids, uncs, bids, bid_profs, signal_count)
    plot_exp_profit(mids, uncs, bids, bid_profs, signal_count, True)

def ga_best_responses(signal_count, n_samples, uncs = None, mids = None):
    max_uncertainty = 1000
    value_bounds = (500, 9500)

    if uncs is None:
        n_uncs = 61
        uncs = np.linspace(0, max_uncertainty, n_uncs)

    if mids is None:
        n_mids = 61
        mids = np.linspace(value_bounds[0] - max_uncertainty/2, value_bounds[1] + max_uncertainty/2, n_mids)

    file_name = f'F:/Dropbox (Chapman)/Costly Signals/sealed_bid_model/bid_funcs_{"".join([str(i) for i in signal_count])}.csv'
    bidders_set = bid_functions.read_file(file_name)


    bids, bid_profs, opt_bids, opt_profs = best_responses_iterative(mids, uncs, bidders_set, max_uncertainty, value_bounds, n_samples)
    #plot_results_3d_save(mids, uncs, bids, bid_profs, signal_count)
    #plot_exp_profit_3d(mids, uncs, bids, bid_profs, signal_count)
    plot_exp_profit(mids, uncs, bids, bid_profs, signal_count, True)
    plot_exp_profit(mids, uncs, opt_bids, opt_profs, signal_count, True, 'br')

def ga_best_responses_2(signal_count, n_samples, uncs = None, mids = None):
    max_uncertainty = 1000
    value_bounds = (500, 9500)
    n_bidders = len(signal_count)

    if uncs is None:
        n_uncs = 61
        uncs = np.linspace(0, max_uncertainty, n_uncs)

    if mids is None:
        n_mids = 61
        mids = np.linspace(value_bounds[0] - max_uncertainty/2, value_bounds[1] + max_uncertainty/2, n_mids)

    file_name = f'F:/Dropbox (Chapman)/Costly Signals/sealed_bid_model/bid_funcs_{"".join([str(i) for i in signal_count])}.csv'
    bidders_set = bid_functions.read_file(file_name)

    sample_rel_mids, sample_uncs = verify_bid_funcs.sample_rel_sufficient_stats(signal_count, n_samples, max_uncertainty)
    pool = Pool(6)
    bids, bid_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_exp_prof_m(m, mids, uncs, sample_rel_mids, sample_uncs, bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))

    opt_bidders_set = bid_functions.BidderSet([b.clone() for b in bidders_set.bidders])
    best_response_reps = 1
    selections = np.array([[i for i in range(n_bidders) if i != j] for j in range(n_bidders)])
    for _ in range(best_response_reps):
        func_mids = opt_bidders_set[0].midpoints
        func_uncs = opt_bidders_set[0].uncertainties
        init_bids = verify_bid_funcs.convert_grid_bids(opt_bidders_set, [func_mids]*n_bidders, [np.repeat(np.transpose([func_uncs]),len(func_mids),axis=1)]*n_bidders)
        avg_opponent_bids = np.mean(init_bids[selections], axis=1)
        mse = ((init_bids-avg_opponent_bids)**2/3).reshape((n_bidders, len(func_mids)*len(func_uncs))).sum(axis=1)

        for b_id in (-mse).argsort():
            opt_bids, opt_profs = zip(*map(lambda m: verify_bid_funcs.calc_best_resp_b_m_2(b_id, m, func_mids, func_uncs, sample_rel_mids, sample_uncs, opt_bidders_set, n_samples, max_uncertainty, value_bounds), range(len(func_mids))))
            opt_bidders_set.bidders[b_id] = bid_functions.Bidder(signal_count[b_id], func_mids, func_uncs, np.array(opt_bids).T)
        
    pool = Pool(6)
    opt_bids, opt_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_exp_prof_m(m, mids, uncs, sample_rel_mids, sample_uncs, opt_bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))
    plot_exp_profit(mids, uncs, bids, bid_profs, signal_count, True)
    plot_exp_profit(mids, uncs, opt_bids, opt_profs, signal_count, True, f'br_{best_response_reps}')

'''
This will allow bidders to best respond from lowest to highest expected profit.
'''
def ga_best_responses_3(signal_count, n_samples, uncs = None, mids = None):
    max_uncertainty = 1000
    value_bounds = (500, 9500)
    n_bidders = len(signal_count)

    if uncs is None:
        n_uncs = 61
        uncs = np.linspace(0, max_uncertainty, n_uncs)

    if mids is None:
        n_mids = 61
        mids = np.linspace(value_bounds[0] - max_uncertainty/2, value_bounds[1] + max_uncertainty/2, n_mids)

    file_name = f'F:/Dropbox (Chapman)/Costly Signals/sealed_bid_model/bid_funcs_{"".join([str(i) for i in signal_count])}.csv'
    bidders_set = bid_functions.read_file(file_name)

    sample_rel_mids, sample_uncs = verify_bid_funcs.sample_rel_sufficient_stats(signal_count, n_samples, max_uncertainty)
    pool = Pool(6)
    bids, bid_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_exp_prof_m(m, mids, uncs, sample_rel_mids, sample_uncs, bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))

    opt_bidders_set = bid_functions.BidderSet([b.clone() for b in bidders_set.bidders])
    best_response_reps = 4
    selections = np.array([[i for i in range(n_bidders) if i != j] for j in range(n_bidders)])
    for _ in range(best_response_reps):
        func_mids = opt_bidders_set[0].midpoints
        func_uncs = opt_bidders_set[0].uncertainties
        profs = bid_profs.sum(axis=0).sum(axis=1)

        for b_id in profs.argsort():
            opt_bids, opt_profs = zip(*map(lambda m: verify_bid_funcs.calc_best_resp_b_m_2(b_id, m, func_mids, func_uncs, sample_rel_mids, sample_uncs, opt_bidders_set, n_samples, max_uncertainty, value_bounds), range(len(func_mids))))
            opt_bidders_set.bidders[b_id] = bid_functions.Bidder(signal_count[b_id], func_mids, func_uncs, np.array(opt_bids).T)
        
    pool = Pool(6)
    opt_bids, opt_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_exp_prof_m(m, mids, uncs, sample_rel_mids, sample_uncs, opt_bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))
    plot_exp_profit(mids, uncs, bids, bid_profs, signal_count, True)
    plot_exp_profit(mids, uncs, opt_bids, opt_profs, signal_count, True, f'br_{best_response_reps}')

'''
NEW PLAN: ALLOW LARGEST DEVIATOR AT EACH MIDPOINT TO UPDATE FIRST WHEN RUNNING CALC_OPTS_M
UPDATE IF NEW BID IS LARGER THAN 3% CHANGE
THEN RUN CALC_BEST_RESPONSE_B_M FOR EACH BIDDER TO ALLOW THEM TO RESPOND INDEPENDENTLY.
'''
def ga_best_responses_4(signal_count, n_samples, plot_uncs = None, mids = None):
    max_uncertainty = 1000
    value_bounds = (500, 9500)
    n_bidders = len(signal_count)

    if mids is None:
        n_mids = 61
        mids = np.linspace(value_bounds[0] - max_uncertainty/2, value_bounds[1] + max_uncertainty/2, n_mids)

    file_name = f'F:/Dropbox (Chapman)/Costly Signals/sealed_bid_model/bid_funcs_{"".join([str(i) for i in signal_count])}.csv'
    bidders_set = bid_functions.read_file(file_name)

    #mids = opt_bidders_set[0].midpoints
    uncs = bidders_set[0].uncertainties

    sample_rel_mids, sample_uncs = verify_bid_funcs.sample_rel_sufficient_stats(signal_count, n_samples, max_uncertainty)
    pool = Pool(6)
    bids, bid_profs, opt_bids, opt_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_opts_iter_m_numba(m, mids, uncs, sample_rel_mids, sample_uncs, bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))

    opt_bidders_set = bid_functions.BidderSet([b.clone() for b in bidders_set.bidders])
    for b_id in range(n_bidders):
        opt_bidders_set[b_id].bids = np.array(opt_bids)[:,b_id,:].T
    
    bids, bid_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_exp_prof_m(m, mids, uncs, sample_rel_mids, sample_uncs, bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))
    profs = np.array(bid_profs).sum(axis=0).sum(axis=1)

    best_response_reps = 4
    selections = np.array([[i for i in range(n_bidders) if i != j] for j in range(n_bidders)])
    for _ in range(best_response_reps):
        func_mids = opt_bidders_set[0].midpoints
        func_uncs = opt_bidders_set[0].uncertainties

        for b_id in profs.argsort():
            opt_bids, opt_profs = zip(*map(lambda m: verify_bid_funcs.calc_best_resp_b_m_2(b_id, m, func_mids, func_uncs, sample_rel_mids, sample_uncs, opt_bidders_set, n_samples, max_uncertainty, value_bounds), range(len(func_mids))))
            opt_bidders_set.bidders[b_id] = bid_functions.Bidder(signal_count[b_id], func_mids, func_uncs, np.array(opt_bids).T)
            #print(f'{np.array(opt_profs).shape}')
            profs[b_id] = np.array(opt_profs).sum()
        
    pool = Pool(6)
    opt_bids, opt_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_exp_prof_m(m, mids, plot_uncs, sample_rel_mids, sample_uncs, opt_bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))
    plot_exp_profit(mids, plot_uncs, bids, bid_profs, signal_count, True)
    plot_exp_profit(mids, plot_uncs, opt_bids, opt_profs, signal_count, True, f'br_{best_response_reps}')

'''
Instead of using calc_opts_iter_m_numba I am going to use calc_exp_profit because I don't want the other bid functions optimizing against the worst
bidder. I will instead let the worst bidder move first.
'''
def ga_best_responses_5(signal_count, n_samples, uncs = None, mids = None, plot_uncs = None, threshold = 0.02, iterations = 1):
    max_uncertainty = 1000
    value_bounds = (500, 9500)
    n_bidders = len(signal_count)

    if mids is None:
        n_mids = 61
        mids = np.linspace(value_bounds[0] - max_uncertainty/2, value_bounds[1] + max_uncertainty/2, n_mids)
    
    if uncs is None:
        n_uncs = 61
        uncs = np.linspace(0, max_uncertainty, n_uncs)

    if plot_uncs is None:
        plot_uncs = [uncs[0], uncs[-1]]

    file_name = f'F:/Dropbox (Chapman)/Costly Signals/sealed_bid_model/bid_funcs_{"".join([str(i) for i in signal_count])}.csv'
    bidders_set = bid_functions.read_file(file_name)

    sample_rel_mids, sample_uncs = verify_bid_funcs.sample_rel_sufficient_stats(signal_count, n_samples, max_uncertainty)
    pool = Pool(6)
    bids, bid_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_exp_prof_m(m, mids, uncs, sample_rel_mids, sample_uncs, bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))
    bids = np.array(bids)

    profs = np.array(bid_profs).sum(axis=0).sum(axis=1)
    opt_bidders_set = []
    for b in range(n_bidders):
        if signal_count[b] == 1:
            opt_bidders_set.append(bid_functions.SingleSignalBidder(signal_count[b], mids, max_uncertainty, bids[:,b,-1]))
        else:
            opt_bidders_set.append(bid_functions.Bidder(signal_count[b], mids, uncs, bids[:,b,:].T))
    opt_bidders_set = bid_functions.BidderSet(opt_bidders_set)

    for _ in range(iterations):

        for b_id in profs.argsort():
            opt_bids, opt_profs = zip(*map(lambda m: verify_bid_funcs.calc_best_resp_b_m_2(b_id, m, mids, uncs, sample_rel_mids, sample_uncs, opt_bidders_set, n_samples, max_uncertainty, value_bounds, threshold), range(len(mids))))
            if signal_count[b_id] == 1:
                opt_bidders_set.bidders[b_id] = bid_functions.SingleSignalBidder(signal_count[b_id], mids, max_uncertainty, np.array(opt_bids)[:,-1])
            else:
                opt_bidders_set.bidders[b_id] = bid_functions.Bidder(signal_count[b_id], mids, uncs, np.array(opt_bids).T)
            #print(f'{np.array(opt_profs).shape}')
            profs[b_id] = np.array(opt_profs).sum()

        opt_bids, opt_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_exp_prof_m(m, mids, plot_uncs, sample_rel_mids, sample_uncs, opt_bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))
        write_bidder(opt_bidders_set, f'br_{iterations}_thresh_{threshold * 100}_reps_{n_samples}_{"_".join([str(i) for i in signal_count])}.csv')
        
    bids, bid_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_exp_prof_m(m, mids, plot_uncs, sample_rel_mids, sample_uncs, bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))

    plot_exp_profit(mids, plot_uncs, bids, bid_profs, signal_count, True)
    plot_exp_profit(mids, plot_uncs, opt_bids, opt_profs, signal_count, True, f'br_{iterations}')

'''
Instead of using calc_opts_iter_m_numba I am going to use calc_exp_profit because I don't want the other bid functions optimizing against the worst
bidder. I will instead let the worst bidder move first.
'''
def ga_best_responses_lambda(signal_count, n_samples, uncs = None, mids = None, plot_uncs = None, threshold = 0.02, iterations = 1):
    max_uncertainty = 1000
    value_bounds = (500, 9500)
    n_bidders = len(signal_count)

    if mids is None:
        n_mids = 61
        mids = np.linspace(value_bounds[0] - max_uncertainty/2, value_bounds[1] + max_uncertainty/2, n_mids)
    
    if uncs is None:
        n_uncs = 61
        uncs = np.linspace(0, max_uncertainty, n_uncs)

    if plot_uncs is None:
        plot_uncs = [uncs[0], uncs[-1]]

    file_name = f'F:/Dropbox (Chapman)/Costly Signals/sealed_bid_model/bid_funcs_{"".join([str(i) for i in signal_count])}.csv'
    bidders_set = bid_functions.read_file(file_name)

    sample_rel_mids, sample_uncs = verify_bid_funcs.sample_rel_sufficient_stats(signal_count, n_samples, max_uncertainty)
    pool = Pool(6)
    bids, bid_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_exp_prof_m(m, mids, uncs, sample_rel_mids, sample_uncs, bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))
    bids = np.array(bids)

    profs = np.array(bid_profs).sum(axis=0).sum(axis=1)
    opt_bidders_set = bid_functions.BidderSet([bid_functions.Bidder(signal_count[b], mids, uncs, bids[:,b,:].T) for b in range(n_bidders)])

    for _ in range(iterations):

        for b_id in profs.argsort():
            opt_bids, opt_profs = zip(*map(lambda m: verify_bid_funcs.calc_best_resp_b_m_2(b_id, m, mids, uncs, sample_rel_mids, sample_uncs, opt_bidders_set, n_samples, max_uncertainty, value_bounds, threshold), range(len(mids))))
            opt_bidders_set.bidders[b_id] = bid_functions.Bidder(signal_count[b_id], mids, uncs, np.array(opt_bids).T)
            #print(f'{np.array(opt_profs).shape}')
            profs[b_id] = np.array(opt_profs).sum()

        opt_bids, opt_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_exp_prof_m(m, mids, plot_uncs, sample_rel_mids, sample_uncs, opt_bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))
        write_bidder(opt_bidders_set, f'br_{iterations}_thresh_{threshold * 100}_reps_{n_samples}_{"_".join([str(i) for i in signal_count])}.csv')
        
    bids, bid_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_exp_prof_m(m, mids, plot_uncs, sample_rel_mids, sample_uncs, bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))

    plot_exp_profit(mids, plot_uncs, bids, bid_profs, signal_count, True)
    plot_exp_profit(mids, plot_uncs, opt_bids, opt_profs, signal_count, True, f'br_{iterations}')

def write_bidder(bid_set, filename):
    newline = '' if os.name == 'nt' else None
    with open(filename, 'a+', newline=newline) as csvfile:
        writer = csv.writer(csvfile)
        for bidder in bid_set:
            writer.writerow([bidder.signals])
            writer.writerow(bidder.midpoints)
            if bidder.signals == 1:
                writer.writerow([bidder.uncertainty])
                writer.writerow(bidder.bids)
            else:
                writer.writerow(bidder.uncertainties)
                writer.writerows(bidder.bids)

def read_file(filename, n_bidders):
    
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        bidder_sets = bid_functions.BidderSets()
        bidders = []
        for row in reader:
            signals = int(row[0])
            midpoints = np.array(next(reader),dtype=np.float64)
            if signals == 1:
                uncertainty = next(reader)
                bids = np.array(next(reader), dtype=np.float64)
                bidders.append(bid_functions.SingleSignalBidder(signals, midpoints, uncertainty, bids))
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
                bidders.append(bid_functions.Bidder(signals, midpoints, uncertainties, bids))
                
                if len(bidders) == n_bidders:
                    bidder_sets.append(bid_functions.BidderSet(bidders))
                    bidders = []
    return bidder_sets

def expected_profits(mids, uncs, bidders, max_uncertainty, value_bounds, n_samples = 1000):
    n_signals = bidders.signals()

    sample_rel_mids, sample_uncs = verify_bid_funcs.sample_rel_sufficient_stats(n_signals, n_samples, max_uncertainty)
    pool = Pool(6)
    return zip(*pool.map(lambda m: expected_profit_m(m, mids, uncs, sample_rel_mids, sample_uncs, bidders, n_samples, max_uncertainty, value_bounds), range(len(mids))))

def best_responses(mids, uncs, bidders, max_uncertainty, value_bounds, n_samples = 1000):
    n_signals = bidders.signals()

    sample_rel_mids, sample_uncs = verify_bid_funcs.sample_rel_sufficient_stats(n_signals, n_samples, max_uncertainty)
    pool = Pool(6)
    return zip(*pool.map(lambda m: verify_bid_funcs.calc_opts_m(m, mids, uncs, sample_rel_mids, sample_uncs, bidders, n_samples, max_uncertainty, value_bounds), range(len(mids))))

def best_responses_iterative(mids, uncs, bidders, max_uncertainty, value_bounds, n_samples = 1000):
    n_signals = bidders.signals()

    sample_rel_mids, sample_uncs = verify_bid_funcs.sample_rel_sufficient_stats(n_signals, n_samples, max_uncertainty)
    pool = Pool(6)
    return zip(*pool.map(lambda m: verify_bid_funcs.calc_opts_iterative_m(m, mids, uncs, sample_rel_mids, sample_uncs, bidders, n_samples, max_uncertainty, value_bounds), range(len(bidders[0].midpoints))))


def plot_exp_profit(mids, uncs, bids, bid_profs, signal_counts, save=False, prefix=''):
    n_uncs = len(uncs)
    signal_str = ",".join([str(s) for s in signal_counts])
    fig, axs = plt.subplots(n_uncs,2, figsize=(16, 9))
    fig.suptitle(f'{signal_str}')
    for unc_idx in range(n_uncs):
        unc = uncs[unc_idx]
        axb = axs[unc_idx][0] if n_uncs > 1 else axs[0]
        axp = axs[unc_idx][1] if n_uncs > 1 else axs[1]
        axb.set_ylabel(f'unc {unc}')
        axb.set_xlim([500 - unc / 2, 9500 + unc / 2])
        axp.set_xlim([500 - unc / 2, 9500 + unc / 2])
        axb.set_ylim([-600,500])
        axp.set_ylim([-10,80])
        for b_id in range(4):
            axb.plot(mids, np.array(bids)[:,b_id,unc_idx] - mids, label=f'Bidder {b_id + 1}')
            axp.plot(mids, np.array(bid_profs)[:,b_id,unc_idx], label=f'Bidder {b_id + 1}')
        axb.legend()
    
    if save:
        plt.savefig(f'{prefix}_{signal_str}.png', dpi=600)
    else:
        plt.show()

def plot_exp_profit_3d(mids, uncs, bids, bid_profs):
    
    fig = plt.figure()
    fig.suptitle(f'Expected Profits')
    for b_id in range(4):
        ax1 = fig.add_subplot(2, 2, b_id + 1, projection='3d')
        ax1.set_title(f'Bidder {b_id}')

        m_idx, u_idx = np.meshgrid(range(len(mids)), range(len(uncs)))
        profs = np.array(bid_profs)[:,b_id,:][m_idx,u_idx]
        surf = ax1.plot_surface(np.array(mids)[m_idx], np.array(uncs)[u_idx], profs, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        #ax1 = axs[b_id // 2][b_id % 2]
        fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.show()

def plot_results_3d_save(mids, uncs, bids, bid_profs, signal_counts):
    signal_str = ",".join([str(s) for s in signal_counts])
    m_idx, u_idx = np.meshgrid(range(len(mids)), range(len(uncs)))
    xlim = np.min(mids), np.max(mids)
    ylim = np.min(uncs), np.max(uncs)

    fig = plt.figure()
    title = f'{signal_str} Expected Profit'
    fig.suptitle(title)
    plt.tight_layout()
    plot_axes = []
    for b_id in range(4):
        
        ax1 = fig.add_subplot(2,2,b_id + 1, projection='3d')
        plot_axes.append(ax1)
        ax1.set_xlim(*xlim)
        ax1.set_ylim(*ylim)
        ax1.set_zlim(-20, 70)

        profs = np.array(bid_profs)[:,b_id,:][m_idx,u_idx]
        #surf = ax1.plot_surface(np.array(mids)[m_idx], np.array(uncs)[u_idx], profs, rstride=5, cstride=100, cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=0, vmax=80)
        ax1.plot_wireframe(np.array(mids)[m_idx], np.array(uncs)[u_idx], profs, rstride=5, cstride=50)

    def rotate(angle):
        for ax in plot_axes:
            ax.view_init(30, angle)

    anim = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
    anim.save(f"{title}.gif", dpi=100, writer=PillowWriter(fps=15))

    fig.clear()
    title = f'{signal_str} Bids'
    fig.suptitle(f'{signal_str} Bidder Exp Profit')
    plt.tight_layout()
    plot_axes = []
    for b_id in range(4):
        
        
        ax1 = fig.add_subplot(2,2,b_id + 1, projection='3d')
        plot_axes.append(ax1)
        ax1.set_xlim(*xlim)
        ax1.set_ylim(*ylim)
        ax1.set_zlim(*xlim)

        bidder_bids = np.array(bids)[:,b_id,:][m_idx,u_idx]
        #surf = ax1.plot_surface(np.array(mids)[m_idx], np.array(uncs)[u_idx], profs, rstride=5, cstride=100, cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=0, vmax=80)
        ax1.plot_wireframe(np.array(mids)[m_idx], np.array(uncs)[u_idx], bidder_bids, rstride=5, cstride=50)

    anim = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
    anim.save(f"{title}.gif", dpi=100, writer=PillowWriter(fps=15))

def test_pathos(mids, uncs, max_uncertainty, value_bounds, n_samples, bidders_set, sample_rel_mids, sample_uncs):
    pool = Pool(6)
    return zip(*pool.map(lambda m: verify_bid_funcs.calc_opts_m(m, mids, uncs, sample_rel_mids, sample_uncs, bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))

def test_single_thread(mids, uncs, max_uncertainty, value_bounds, n_samples, bidders_set, sample_rel_mids, sample_uncs):
    return zip(*map(lambda m: verify_bid_funcs.calc_opts_m(m,mids, uncs, sample_rel_mids, sample_uncs, bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))

def test_partial_numba(mids, uncs, max_uncertainty, value_bounds, n_samples, bidders_set, sample_rel_mids, sample_uncs):
    return zip(*map(lambda m: verify_bid_funcs.calc_opts_m_numba(m,mids, uncs, sample_rel_mids, sample_uncs, bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))

def test_pathos_numba_equal():
    n_samples = 1000
    signal_count = (2,2,2,2)
    max_uncertainty = 1000
    value_bounds = (500, 9500)
    n_bidders = len(signal_count)

    n_uncs = 61
    uncs = np.linspace(0, max_uncertainty, n_uncs)

    n_mids = 61
    mids = np.linspace(value_bounds[0] - max_uncertainty/2, value_bounds[1] + max_uncertainty/2, n_mids)
    
    file_name = f'F:/Dropbox (Chapman)/Costly Signals/sealed_bid_model/bid_funcs_{"".join([str(i) for i in signal_count])}.csv'
    bidders_set = bid_functions.read_file(file_name)
    
    sample_rel_mids, sample_uncs = verify_bid_funcs.sample_rel_sufficient_stats(signal_count, n_samples, max_uncertainty)

    bids, profs, opt_bids, opt_profs = test_pathos(mids, uncs, max_uncertainty, value_bounds, n_samples, bidders_set, sample_rel_mids, sample_uncs)
    bids2, profs2, opt_bids2, opt_profs2 = test_partial_numba(mids, uncs, max_uncertainty, value_bounds, n_samples, bidders_set, sample_rel_mids, sample_uncs)
    print(f'Bid diff = {(np.array(bids) - np.array(bids2)).sum()}')
    print(f'prof diff = {(np.array(profs) - np.array(profs2)).sum()}')
    print(f'opt_bid diff = {(np.array(opt_bids) - np.array(opt_bids2)).sum()}')
    print(f'opt_prof diff = {(np.array(opt_profs) - np.array(opt_profs2)).sum()}')

def plot_bidder_set(bidder_set, uncs, mids, n_samples):
    fig, axs = plt.subplots(len(uncs), 2)
    fig.set_size_inches(16,10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    colors = ['blue','green','yellow','orange']

    sample_rel_mids, sample_uncs = verify_bid_funcs.sample_rel_sufficient_stats(bidder_set.signals(), n_samples)

    bids, bid_profs = zip(*map(lambda m: verify_bid_funcs.calc_exp_prof_m(m, mids, uncs, sample_rel_mids, sample_uncs, bidder_set, n_samples), range(len(mids))))
    bids = np.array(bids)
    bid_profs = np.array(bid_profs)

    for u in range(len(uncs)):
        ax_bid = axs[u][0]
        ax_bid.set_ylim(-600, 300)
        ax_prof = axs[u][1]
        for bidder_idx in range(4):
            ax_bid.plot(mids, bids[:,bidder_idx, u] - mids, color=colors[bidder_idx])
            ax_prof.plot(mids, bid_profs[:,bidder_idx, u], color=colors[bidder_idx])


def main():
    #single_signal_case()
    #for sigs in [(2,2,2,2), (2,3,4,5), (2,2,2,5), (3,3,4,4), (2,5,5,5)]:
    for sigs in [(1,1,1,1)]:
    #    #ga_expected_profits(sigs, 10000, np.array([100,500,900]), np.linspace(0, 10000, 61))
        ga_best_responses_5(sigs, 1000000, mids = np.linspace(0, 10000, 121), uncs = np.linspace(0,1000,2), plot_uncs = np.array([1000]), threshold = 0.02, iterations = 4)
        # if increasing sample size doesnt work then next step might be to extend bid function precision through best response.

    

if __name__ == "__main__":
    main()
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    # main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.dump_stats('parital_numba_profile')