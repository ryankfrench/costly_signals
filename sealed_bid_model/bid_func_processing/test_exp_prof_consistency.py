import numpy as np
import importlib.util
import ryan_ga_verify as ryan
import sim_combiner
import bid_functions
import verify_bid_funcs
import matplotlib.pyplot as plt
from pathos.multiprocessing import Pool


def main():
    bidder_iters = ryan.read_file("F:\\Dropbox (Chapman)\\Costly Signals\\sealed_bid_model\\bid_func_processing\\br_25_thresh_2.0_reps_100000_2_2_2_2_241_mids\\br_100_thresh_2.0_reps_100000_2_2_2_2.csv", 4)
    bidder_set = bidder_iters.bid_sets[0]
    n_samples = 100000
    mids = np.linspace(0,10000,61)
    uncs = [100, 500, 900]

    pool = Pool(8)
    orderings = []
    for _ in range(1000):
        sample_rel_mids, sample_uncs = verify_bid_funcs.sample_rel_sufficient_stats(bidder_set.signals(), n_samples)
        bids, bid_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_exp_prof_m(m, mids, uncs, sample_rel_mids, sample_uncs, bidder_set, n_samples), range(len(mids))))
        bids = np.array(bids)
        bid_profs = np.array(bid_profs)
        profs = np.array(bid_profs).sum(axis=0).sum(axis=1)
        orderings.append(profs.argsort())
    
    plt.matshow(orderings)
    plt.show()

if __name__ == "__main__":
    main()