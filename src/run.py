from preprocessing import run_deepmatcher
from pprint import pprint
import workloads as wl
import pandas as pd
import matplotlib.pyplot as plt

def plot_bargraph(data, filename):
    data.sort()
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(data)
    plt.savefig(filename + ".png")

def main():
    predictions = run_deepmatcher("../data/itunes-amazon", epochs = 2)
    workload = wl.Workload(pd.read_csv("../data/itunes-amazon/test.csv"), "left_Genre", "right_Genre", 
                            predictions, label_column = "label", multiple_sens_attr = True, 
                            delimiter = ",", single_fairness = True)
    k_combs = workload.create_k_combs(2)

    values_acc_par = workload.is_fair(k_combs, "accuracy_parity")
    values_miss_rate_par = workload.is_fair(k_combs, "misclassification_rate_parity")

    plot_bargraph(values_acc_par, "accuracy_parity_single")
    plot_bargraph(values_miss_rate_par, "misclassification_rate_parity_single")
    

main()