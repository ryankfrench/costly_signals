import os
import csv
import numpy as np

def read_bid_func(data, n, max_uncertainty):
    midpoints = [float(n) for n in data[0]]
    if n == 1:
        bids = [max(500, float(n)) for n in data[1]]
        return {"midpoints": midpoints, "uncertainties": [max_uncertainty], "bids": np.array([bids])}
    else:
        uncertainties = np.array([float(n) for n in data[1]])
        bids = np.array([float(n) for n in data[2]])
        bid_mat = []
        for i in range(0, len(bids), len(midpoints)):
            bid_mat.append(np.array(bids[i:i + len(midpoints)]))
        bid_mat = np.array(bid_mat)
        bid_mat = np.maximum(bid_mat, 500)
        return {"midpoints": midpoints, "uncertainties": uncertainties, "bids": bid_mat}


def tail(f, lines=1, _buffer=4098):
    """Tail a file and get X lines from the end"""
    lines_found = []
    block_counter = -1
    while len(lines_found) < lines:
        try:
            f.seek(block_counter * _buffer, os.SEEK_END)
        except IOError:  # either file is too small, or too many lines requested
            f.seek(0)
            lines_found = f.readlines()
            break
        lines_found = f.readlines()
        block_counter -= 1
    return lines_found[-lines:]


def main():
    max_signals = 5
    signal_count_sets = []
    for i in range(1, max_signals+1):
        for j in range(i, max_signals+1):
            for k in range(j, max_signals+1):
                for l in range(k, max_signals+1):
                    signal_count_sets.append((i,j,k,l))
    in_dir = '/media/kajames/Cheese4Lyfe/'
    out_dir = '.'
    for signal_counts in signal_count_sets:
        process_signal_count_set(signal_counts, in_dir, out_dir)


def process_signal_count_set(signal_counts, in_dir, out_dir):
    n_lines = [2 + (s>1) for s in signal_counts]
    if signal_counts == (1,1,1,1):
        n_period_means = 199
    else:
        n_period_means = 5
    n_tot_lines = sum(n_lines) + 1  # extra line for profits
    signal_str = '_'.join([str(s) for s in signal_counts])
    max_uncertainty = 1000
    with open(in_dir + '/' + 'common_value_' + signal_str + '.csv', 'r') as csvfile:
        data = tail(csvfile, n_period_means * n_tot_lines)
        data = [line.split(',') for line in data]
    bid_dict = {}
    cur_ind = 0
    for per in range(n_period_means):
        for i in range(len(signal_counts)):
            next_ind = cur_ind + n_lines[i]
            if per == 0:
                bid_dict[i] = read_bid_func(data[cur_ind:next_ind], signal_counts[i], max_uncertainty)
            else:
                bid_dict[i]['bids'] =  bid_dict[i]['bids'] + read_bid_func(data[cur_ind:next_ind], signal_counts[i], max_uncertainty)['bids']
            cur_ind = next_ind
        if per == 0:
            bid_dict["profits"] = np.array([float(n) for n in data[cur_ind]])
        else:
            bid_dict["profits"] += np.array([float(n) for n in data[cur_ind]])
        cur_ind += 1
    for i in range(len(signal_counts)):
        bid_dict[i]['bids'] /= n_period_means
    bid_dict['profits'] /= n_period_means
    bid_dict["signals"] = signal_counts
    with open(out_dir + '/bid_funcs_'+''.join([str(i) for i in bid_dict["signals"]]) + ".csv", 'w') as outfile:
        writer = csv.writer(outfile)
        for i in range(0, len(bid_dict["signals"])):
            writer.writerow([bid_dict['signals'][i]])
            writer.writerow(bid_dict[i]['midpoints'])
            writer.writerow(bid_dict[i]['uncertainties'])
            for j in range(len(bid_dict[i]['uncertainties'])):
                writer.writerow(bid_dict[i]['bids'][j])

if __name__ == "__main__":
    main()
