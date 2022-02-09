from preprocessing import run_deepmatcher
from pprint import pprint
import workloads as wl
import pandas as pd
import FairEM as fem
import matplotlib.pyplot as plt

def plot_bargraph(data, filename):
    data.sort()
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(data)
    plt.savefig(filename + ".png")

def run_one_workload():
    predictions = run_deepmatcher("../data/itunes-amazon", epochs = 2)
    workload = wl.Workload(pd.read_csv("../data/itunes-amazon/test.csv"), "left_Genre", 
                            "right_Genre", predictions, label_column = "label", 
                            multiple_sens_attr = True, delimiter = ",", single_fairness = True)
    k_combs = workload.create_k_combs(2)
    return [workload], k_combs

def run_multiple_workloads():
    predictions = run_deepmatcher("../data/itunes-amazon", epochs = 7)
    workload1 = wl.Workload(pd.read_csv("../data/itunes-amazon/test1.csv"), "left_Genre", 
                            "right_Genre", predictions, label_column = "label", 
                            multiple_sens_attr = True, delimiter = ",", single_fairness = True)
    workload2 = wl.Workload(pd.read_csv("../data/itunes-amazon/test2.csv"), "left_Genre", 
                            "right_Genre", predictions, label_column = "label", 
                            multiple_sens_attr = True, delimiter = ",", single_fairness = True)
    
    workloads = [workload1, workload2]
    k_combs = workload1.create_k_combs(2)
    return workloads, k_combs

    
def main():
    workloads, k_combs = run_one_workload()
    alpha = 0.2
    threshold = 0.6
    fairEM = fem.FairEM(workloads, alpha, threshold, single_fairness=True)

    for measure in ["accuracy_parity", "misclassification_rate_parity"]:
        for aggregate in ["max", "min", "max_minus_min", "average", "distribution"]:
            res = fairEM.is_fair(k_combs, measure, aggregate)
            print(measure, aggregate, res)
        
    # values_acc_par = workload.fairness(k_combs, "accuracy_parity")
    # values_miss_rate_par = workload.fairness(k_combs, "misclassification_rate_parity")

    # plot_bargraph(values_acc_par, "accuracy_parity_single")
    # plot_bargraph(values_miss_rate_par, "misclassification_rate_parity_single")
    

main()