import csv
import numpy as np

def read_bid_func(reader, n, max_uncertainty):
    midpoints = [float(n) for n in next(reader)]
    if n == 1:
        bids = [float(n) for n in next(reader)]
        return {"midpoints": midpoints, "uncertainties": [max_uncertainty], "bids": [bids]}
    else:
        uncertainties = np.array([float(n) for n in next(reader)])
        bids = np.array([float(n) for n in next(reader)])
        bid_mat = []
        for i in range(0, len(bids), len(midpoints)):
            bid_mat.append(np.array(bids[i:i + len(midpoints)]))
        bid_mat = np.array(bid_mat)
        mid_mesh, u_mesh = np.meshgrid(midpoints, uncertainties)
        bid_mat = bid_mat*(mid_mesh + u_mesh/2 > 500) + 500*(mid_mesh + u_mesh/2 <= 500)
        bid_mat = bid_mat*(mid_mesh - u_mesh/2 < 9500) + 9500*(mid_mesh - u_mesh/2 >= 9500)
        return {"midpoints": midpoints, "uncertainties": uncertainties, "bids": bid_mat}


def write_eq(func_dict, writer):
    writer.writerow(["", "", "midpoint"])
    writer.writerow(["", ""] + func_dict["midpoints"])
    for i in range(len(func_dict["uncertainties"])):
        label = ""
        if i == 0:
            label = "uncertainty"
        writer.writerow([label, func_dict["uncertainties"][i]] + func_dict["bids"][i])


def main():
    with open('combined_res.csv') as csvfile:
        reader = csv.reader(csvfile)
        bid_funcs = []
        max_uncertainty = 1000
        for row in reader:
            signals = [int(float(n)) for n in row]
            bid_dict = {}
            for i in range(len(signals)):
                bid_dict[i] = read_bid_func(reader, signals[i], max_uncertainty)
            bid_dict["profits"] = [float(n) for n in next(reader)]
            bid_dict["signals"] = signals
            bid_funcs.append(bid_dict)
    for bid_func in bid_funcs:
        with open('bid_funcs_'+''.join([str(i) for i in bid_func["signals"]]) + ".csv", 'w') as outfile:
            writer = csv.writer(outfile)
            # writer.writerow(["player"] + list(range(len(bid_func["signals"]))))
            # writer.writerow(["signals"] + list(bid_func["signals"]))
            for i in range(0, len(bid_func["signals"])):
                writer.writerow([bid_func['signals'][i]])
                writer.writerow(bid_func[i]['midpoints'])
                writer.writerow(bid_func[i]['uncertainties'])
                for j in range(len(bid_func[i]['uncertainties'])):
                    writer.writerow(bid_func[i]['bids'][j])

if __name__ == "__main__":
    main()
