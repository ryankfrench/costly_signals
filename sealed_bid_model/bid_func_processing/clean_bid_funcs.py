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

def main():
    filenames = get_filenames()
    for filename in filenames:
        bidder_set = read_file(filename)
        n_iters = 50
        for j in range (len(bidder_set)):
            for i in range(n_iters):
                thresh = 1000*((n_iters-1-i)/(n_iters-1)) + 1000*(i/(n_iters-1))
                if bidder_set[j].signals==1:
                    pass
                    # clean_single_bid(bidder_set[j], thresh)
                else:
                    clean_bid_func(bidder_set[j], (0,1), thresh)
                    clean_bid_func(bidder_set[j], (0,-1), thresh)
                    clean_bid_func(bidder_set[j], (1,0), thresh)
                    clean_bid_func(bidder_set[j], (-1,0), thresh)
        write_file(bidder_set, "clean_bid_funcs_" + bidder_set.signal_string() + ".csv")


def get_filenames():
    return pathlib.Path.cwd().glob("bid_funcs_*.csv")


if __name__ == "__main__":
    main()
