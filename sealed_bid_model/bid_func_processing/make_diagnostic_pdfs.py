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
from bid_functions import *

colors = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "indigo",
    "magenta",
    "cyan",
    "black",
]

def main():
    bidder_sets = BidderSets()
    filenames = get_filenames()
    for filename in filenames:
        bidder_sets.append(read_file(filename))
    save_within_plots(bidder_sets)
    # save_me_constant_plots(bidder_sets)
    # save_them_constant_plots(bidder_sets)
    # save_all_same_comparison_plots(bidder_sets)

def get_filenames():
    return pathlib.Path.cwd().glob("br_bids_*.csv")

def save_within_plots(bid_sets):
    for bid_set in bid_sets:
        with PdfPages('br_plots' + bid_set.signal_string() + '.pdf') as pdf:
            bid_func_plots(bid_set, pdf)
            bid_slices_plots(bid_set, pdf)


def bid_func_plots(bid_set, pdf):
    i = 0
    for bidder in bid_set:
        if bidder.signals > 1:
            plot_contour(bidder, pdf, i)
            plot_midpoint_slices(bidder, pdf, i)
            plot_uncertainty_slices(bidder, pdf, i)
        else:
            fig, ax = plt.subplots()
            ax.set_xlabel('midpoint')
            ax.set_ylabel('bid')
            ax.set_title('Bidder=' + str(i) + ' Signal=' + str(bidder.signals))
            plt.plot(bidder.midpoints, bidder.bids, linewidth=2)
            plt.ylim(0, 10000)
            pdf.savefig()
            plt.close()
        i += 1

def plot_contour(bidder, pdf, i):
    fig, ax = plt.subplots()
    ax.set_xlabel('midpoint')
    ax.set_ylabel('uncertainty')
    bid_mat = bidder.bids
    mid_mesh, u_mesh = np.meshgrid(bidder.midpoints, bidder.uncertainties)
    bid_mat = np.where(mid_mesh + u_mesh/2 > 500, bid_mat, np.nan)
    bid_mat = np.where(mid_mesh - u_mesh/2 < 9500, bid_mat, np.nan)
    cont = plt.contour(bidder.midpoints, bidder.uncertainties, bid_mat, levels=np.array([550,600,1000,2000,3000,4000,5000,6000,7000,8000,9000,9400,9450]))
    ax.clabel(cont, inline=1, fontsize=10)
    ax.set_title('Bidder=' + str(i) + ' Signal=' + str(bidder.signals))
    pdf.savefig()
    plt.close()

def plot_midpoint_slices(bidder, pdf, i):
    fig, ax = plt.subplots()
    ax.set_xlabel('midpoint')
    ax.set_ylabel('bid')
    ax.set_title('Bidder=' + str(i) + ' Signal=' + str(bidder.signals))
    plt.ylim(0,10000)
    i = 0
    for unc in range(100, 901, 200):
        plt.plot(bidder.midpoints[bidder.valid_midpoints(unc)], bidder[unc][bidder.valid_midpoints(unc)], linewidth=2, label=unc)
        i += 1
    plt.legend()
    pdf.savefig()
    plt.close()

def plot_uncertainty_slices(bidder, pdf, i):
    fig, ax = plt.subplots()
    ax.set_xlabel('uncertainty')
    ax.set_ylabel('bid')
    ax.set_title('Bidder=' + str(i) + ' Signal=' + str(bidder.signals))
    plt.ylim(0,10000)
    i = 0
    for mid in [500,1000,2500,5000,7500,9000,9500]:
        uncs = bidder.uncertainties
        bids = bidder[:,mid]
        bids = np.where(mid + uncs/2 > 500, bids, np.nan)
        bids = np.where(mid - uncs/2 < 9500, bids, np.nan)
        plt.plot(uncs, bids, linewidth=2, label=mid)
        i += 1
    plt.legend()
    pdf.savefig()
    plt.close()

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def bid_slices_plots(bid_set, pdf):
    for unc in [0, 100,300,500,700,900,1000]:
        i = 0
        n_plots = 0
        fig, ax = plt.subplots()
        ax.set_xlabel('midpoint')
        ax.set_ylabel('bid')
        ax.set_title('uncertainty=' + str(unc))
        plt.ylim(0,10000)
        plt.xlim(0,10000)
        for bidder in bid_set:
            if bidder.signals > 1:
                valid_mids = bidder.midpoints[bidder.valid_midpoints(unc)]
                plt.plot(valid_mids, bidder[unc][bidder.valid_midpoints(unc)],
                         linewidth=2,
                         label='bidder=' + str(i) + ', ' 'signals=' + str(bidder.signals))
                n_plots += 1
            if bidder.signals == 1 and unc == 1000:
                plt.plot(moving_average(bidder.midpoints,n=3), moving_average(np.sort(bidder.bids),n=3),
                         linewidth=2,
                         label='bidder=' + str(i) + ', ' 'signals=' + str(bidder.signals))
                n_plots += 1
            i += 1
        if n_plots > 0:
            plt.legend()
            pdf.savefig()
        plt.close()

    for unc in [0,100,300,500,700,900,1000]:
        i = 0
        n_plots = 0
        fig, ax = plt.subplots()
        ax.set_xlabel('midpoint')
        ax.set_ylabel('relative bid')
        ax.set_title('uncertainty=' + str(unc))
        plt.ylim(-1000,500)
        plt.xlim(0,10000)
        for bidder in bid_set:
            if bidder.signals > 1:
                valid_mids = bidder.midpoints[bidder.valid_midpoints(unc)]
                plt.plot(moving_average(valid_mids, n=3), moving_average(bidder[unc][bidder.valid_midpoints(unc)] - valid_mids, n=3),
                         linewidth=2,
                         label='bidder=' + str(i) + ', ' 'signals=' + str(bidder.signals))
                n_plots += 1
            if bidder.signals == 1 and unc == 1000:
                plt.plot(moving_average(bidder.midpoints, n=5), moving_average(np.sort(bidder.bids) - bidder.midpoints, n=5),
                         linewidth=2,
                         label='bidder=' + str(i) + ', ' 'signals=' + str(bidder.signals))
                n_plots += 1
            i += 1
        if n_plots > 0:
            plt.legend()
            pdf.savefig()
        plt.close()

    for mid in [100,250, 500, 800, 1000, 1500, 3000, 4000, 5000, 6000, 7000, 8500,9000, 9200, 9500, 9750, 9900]:
        i = 0
        fig, ax = plt.subplots()
        ax.set_xlabel('uncertainty')
        ax.set_ylabel('bid')
        ax.set_title('midpoint=' + str(mid))
        if mid <= 1000:
            plt.ylim(400, 1100)
        elif mid >= 9500:
            plt.ylim(8900, 9600)
        else:
            plt.ylim(mid-600, mid+100)
        plt.xlim(0,1000)
        for bidder in bid_set:
            if bidder.signals > 1:
                plt.plot(moving_average(bidder.uncertainties[bidder.valid_uncertainties(mid)],n=3), moving_average(bidder[:,mid][bidder.valid_uncertainties(mid)],n=3),
                         linewidth=1,
                         label=str(bidder.signals) + ' signals')
            else:
                plt.scatter([bidder.uncertainty], [bidder[mid]], label=str(bidder.signals) + ' signal')
            i += 1
        plt.plot(np.linspace(max(0,max(2*(500-mid), 2*(mid-9500))), 1000, 101), np.clip(mid - np.linspace(max(0,max(2*(500-mid),2*(mid-9500))),1000,101)/2, 500, 9500), 'k--', linewidth=0.5, label = 'true value limit')
        plt.plot(np.linspace(max(0,max(2*(500-mid), 2*(mid-9500))), 1000, 101), np.clip(mid + np.linspace(max(0,max(2*(500-mid),2*(mid-9500))),1000,101)/2, 500, 9500), 'k--', linewidth=0.5)
        plt.legend()
        pdf.savefig()
        plt.close()

def save_comparison_plots(bidder_sets, signal_list, title):
    bidders = [bidder_sets[i].get_by_signal(signal_list[i])[0] for i in range(len(bidder_sets))]
    bid_set = BidderSet(bidders)
    with PdfPages(title + '.pdf') as pdf:
        print(title)
        bid_func_plots(bid_set, pdf)
        bid_slices_plots(bid_set, pdf)

def save_me_constant_plots(bidder_sets):
    for me in range(1,6):
        subset = []
        me_sigs = []
        them_sigs = []
        for them in range(1,6):
            subset.append(bidder_sets.select((me, them, them, them)))
            me_sigs.append(me)
            them_sigs.append(them)
        save_comparison_plots(subset, me_sigs, 'me_'+''.join([str(s) for s in me_sigs]) + '_them_' + ''.join([str(s) for s in them_sigs]))

def save_them_constant_plots(bidder_sets):
    for them in range(1,6):
        subset = []
        me_sigs = []
        them_sigs = []
        for me in range(1,6):
            subset.append(bidder_sets.select((me, them, them, them)))
            me_sigs.append(me)
            them_sigs.append(them)
        save_comparison_plots(subset, me_sigs, 'me_'+''.join([str(s) for s in me_sigs]) + '_them_' + ''.join([str(s) for s in them_sigs]))

def save_all_same_comparison_plots(bidder_sets):
    subset = []
    us_sigs = []
    them_sigs = []
    for us in range(1,6):
        subset.append(bidder_sets.select((us, us, us, us)))
        us_sigs.append(us)
    save_comparison_plots(subset, us_sigs, 'us_'+''.join([str(s) for s in us_sigs]))

if __name__ == "__main__":
    main()
