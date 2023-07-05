import numpy as np
import importlib.util
import ryan_ga_verify as ryan
import sim_combiner
import bid_functions
import verify_bid_funcs
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from pathos.multiprocessing import ProcessingPool as Pool
from celluloid import Camera

def main():
    n_samples = 1000000
    signal_count = (2,2,2,2)
    max_uncertainty = 1000
    value_bounds = (500, 9500)
    n_bidders = len(signal_count)

    uncs = np.array([100,500,900])

    file_name = "br_20_thresh_2.0_reps_1000000_2_2_2_2_nmids_121"

    bidder_iters = ryan.read_file(f"{file_name}.csv", 4)

    mids = bidder_iters.bid_sets[0][0].midpoints

    ga_bidders = bid_functions.read_file(f'F:/Dropbox (Chapman)/Costly Signals/sealed_bid_model/bid_funcs_{"".join([str(i) for i in signal_count])}.csv')
    bidder_iters.bid_sets.insert(0, ga_bidders)

    sample_rel_mids, sample_uncs = verify_bid_funcs.sample_rel_sufficient_stats(signal_count, n_samples, max_uncertainty)

    fig, axs = plt.subplots(len(uncs), 2)
    fig.set_size_inches(16,10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    colors = ['blue','green','yellow','orange']
    camera = Camera(fig)

    pool = Pool(6)

    iteration = 1
    for bidder_set in list(bidder_iters):
        bids, bid_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_exp_prof_m(m, mids, uncs, sample_rel_mids, sample_uncs, bidder_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))
        bids = np.array(bids)
        bid_profs = np.array(bid_profs)

        for u in range(len(uncs)):
            ax_bid = axs[u][0]
            ax_bid.set_ylim(-600, 300)
            ax_prof = axs[u][1]
            for bidder_idx in range(4):
                ax_bid.plot(mids, bids[:,bidder_idx, u] - mids, color=colors[bidder_idx])
                ax_prof.plot(mids, bid_profs[:,bidder_idx, u], color=colors[bidder_idx])

        axs[0][0].text(0.5, 1.01, f'{iteration}/{len(bidder_iters.bid_sets)}', transform=axs[0][0].transAxes)
        camera.snap()
        iteration += 1
        
        # for u in range(3):
        #     for i in range(2):
        #         axs[u][i].clear()

    anim = camera.animate()
    plt.close()

    anim.save(f"{file_name}.gif", dpi=300, writer=PillowWriter(fps=1))

if __name__ == "__main__":
    main()