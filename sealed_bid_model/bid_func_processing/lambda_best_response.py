import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import bid_functions
import verify_bid_funcs
import ryan_ga_verify as ryan
from celluloid import Camera

'''
Instead of using calc_opts_iter_m_numba I am going to use calc_exp_profit because I don't want the other bid functions optimizing against the worst
bidder. I will instead let the worst bidder move first.
'''
def ga_best_responses_lambda(signal_count, n_samples, uncs = None, mids = None, plot_uncs = None, threshold = 0.02, iterations = 1, pools = 22):
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

    out_file = f'br_{iterations}_thresh_{threshold * 100}_reps_{n_samples}_{"_".join([str(i) for i in signal_count])}_nmids_{len(mids)}'
    file_name = f'F:/Dropbox (Chapman)/Costly Signals/sealed_bid_model/bid_funcs_{"".join([str(i) for i in signal_count])}.csv'
    bidders_set = bid_functions.read_file(file_name)

    sample_rel_mids, sample_uncs = verify_bid_funcs.sample_rel_sufficient_stats(signal_count, n_samples, max_uncertainty)
    pool = Pool(pools)
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

    fig, axs = plt.subplots(len(plot_uncs), 2)
    fig.set_size_inches(16,10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    camera = Camera(fig)
    for iteration in range(iterations):
        print(f'**** ITERATION {iteration} ****')
        for b_id in profs.argsort():
            opt_bids, opt_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_best_resp_b_m_2(b_id, m, mids, uncs, sample_rel_mids, sample_uncs, opt_bidders_set, n_samples, max_uncertainty, value_bounds, threshold, verify_bid_funcs.find_opt_bid_single), range(len(mids))))
            if signal_count[b_id] == 1:
                opt_bidders_set.bidders[b_id] = bid_functions.SingleSignalBidder(signal_count[b_id], mids, max_uncertainty, np.array(opt_bids)[:,-1])
            else:
                opt_bidders_set.bidders[b_id] = bid_functions.Bidder(signal_count[b_id], mids, uncs, np.array(opt_bids).T)
            #print(f'{np.array(opt_profs).shape}')
            profs[b_id] = np.array(opt_profs).sum()
        
        ryan.write_bidder(opt_bidders_set, f'{out_file}.csv')
        opt_bids, opt_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_exp_prof_m(m, mids, plot_uncs, sample_rel_mids, sample_uncs, opt_bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))
        plot_bidders(mids, plot_uncs, opt_bids, opt_profs, axs, camera, iteration, iterations)
        
    bids, bid_profs = zip(*pool.map(lambda m: verify_bid_funcs.calc_exp_prof_m(m, mids, plot_uncs, sample_rel_mids, sample_uncs, bidders_set, n_samples, max_uncertainty, value_bounds), range(len(mids))))

    ryan.plot_exp_profit(mids, plot_uncs, bids, bid_profs, signal_count, True)
    ryan.plot_exp_profit(mids, plot_uncs, opt_bids, opt_profs, signal_count, True, f'br_{iterations}')

    anim = camera.animate()
    anim.save("{out_file}.gif", dpi=300, writer=PillowWriter(fps=1))

def plot_bidders(mids, plot_uncs, opt_bids, opt_profs, axs, camera, iteration, iterations):
    colors = ['blue','green','yellow','orange']
    opt_bids = np.array(opt_bids)
    opt_profs = np.array(opt_profs)

    for u in range(len(plot_uncs)):
        ax_bid = axs[u][0] if len(plot_uncs) > 1 else axs[0]
        ax_bid.set_ylim(-600, 450)
        ax_prof = axs[u][1] if len(plot_uncs) > 1 else axs[1]
        for bidder_idx in range(4):
            ax_bid.plot(mids, opt_bids[:,bidder_idx, u] - mids, color=colors[bidder_idx])
            ax_prof.plot(mids, opt_profs[:,bidder_idx, u], color=colors[bidder_idx])

    iter_axs = axs[0][0] if len(plot_uncs) > 1 else axs[0]
    iter_axs.text(0.5, 1.01, f'{iteration}/{iterations}', transform=iter_axs.transAxes)
    camera.snap()

def main():
    #for sigs in [(2,2,2,2), (2,3,4,5), (2,2,2,5), (3,3,4,4), (2,5,5,5)]:
    # for sigs in [(2,2,2,2)]:
    #     ga_best_responses_lambda(sigs, 1000000, mids = np.linspace(0, 10000, 121), plot_uncs = np.array([100, 500, 900]), threshold = 0.02, iterations = 100)
    for sigs in [(1,1,1,1)]:
        ga_best_responses_lambda(sigs, 1000000, mids = np.linspace(0, 10000, 241), uncs = np.linspace(0,1000, 2), plot_uncs = np.array([1000]), threshold = 0.03, iterations = 6, pools = 6)



if __name__ == "__main__":
    main()