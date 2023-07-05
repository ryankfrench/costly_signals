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

good_bidsets = [(1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3), (4, 4, 4, 4), (5, 5, 5, 5), (2,3,4,5)]
good_midpoints = [800, 1000, 3000, 5000, 7000]

def main():
    filenames = get_filenames()
    bidder_sets = BidderSets()
    for filename in filenames:
        bidder_sets.append(read_file(filename))
    save_within_plots(bidder_sets)

def get_filenames():
    return pathlib.Path.cwd().glob("br_bids_*.csv")

def save_within_plots(bid_sets):
    for bid_set in bid_sets:
        if tuple(bid_set.signals()) not in good_bidsets:
            continue
        bid_func_plots(bid_set)
        bid_slices_plots(bid_set)


def bid_func_plots(bid_set):
    i = 0
    for bidder in bid_set:
        if bidder.signals > 1:
            plot_contour(bidder, i, bid_set)
        else:
            fig, ax = plt.subplots()
            ax.set_xlabel('midpoint')
            ax.set_ylabel('bid')
            ax.set_title('Bidder=' + str(i) + ' Signal=' + str(bidder.signals))
            plt.plot(bidder.midpoints, bidder.bids, linewidth=2)
            plt.ylim(0, 10000)
            plt.savefig('bid_func_'+str(i)+'_'+''.join([str(s) for s in bid_set.signals()])+'.png')
            plt.close()
        i += 1

def plot_contour(bidder, i, bid_set):
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
    plt.savefig('bid_func_'+str(i)+'_'+''.join([str(s) for s in bid_set.signals()])+'.png')
    plt.close()

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def bid_slices_plots(bid_set):
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
        if mid in good_midpoints:
            plt.savefig('bid_slice_unc_'+str(mid)+'_'+''.join([str(s) for s in bid_set.signals()])+'.png')
        plt.close()

    for mid in [100,250, 500, 800, 1000, 1500, 3000, 4000, 5000, 6000, 7000, 8500,9000, 9200, 9500, 9750, 9900]:
        i = 0
        fig, ax = plt.subplots()
        ax.set_xlabel('range')
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
                plt.plot(moving_average(1000 - bidder.uncertainties[bidder.valid_uncertainties(mid)],n=3), moving_average(bidder[:,mid][bidder.valid_uncertainties(mid)],n=3),
                         linewidth=1,
                         label=str(bidder.signals) + ' signals')
            else:
                plt.scatter([bidder.uncertainty], [bidder[mid]], label=str(bidder.signals) + ' signal')
            i += 1
        plt.plot(1000 - np.linspace(max(0,max(2*(500-mid), 2*(mid-9500))), 1000, 101), np.clip(mid - np.linspace(max(0,max(2*(500-mid),2*(mid-9500))),1000,101)/2, 500, 9500), 'k--', linewidth=0.5, label = 'true value limit')
        plt.plot(1000 - np.linspace(max(0,max(2*(500-mid), 2*(mid-9500))), 1000, 101), np.clip(mid + np.linspace(max(0,max(2*(500-mid),2*(mid-9500))),1000,101)/2, 500, 9500), 'k--', linewidth=0.5)
        plt.legend()
        if mid in good_midpoints:
            plt.savefig('bid_slice_range_'+str(mid)+'_'+''.join([str(s) for s in bid_set.signals()])+'.png')
        plt.close()

if __name__ == "__main__":
    main()
