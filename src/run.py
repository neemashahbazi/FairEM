from preprocessing import run_deepmatcher
from pprint import pprint
from create_multiple_workloads import create_workloads_from_file
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
                            multiple_sens_attr = True, delimiter = ",", single_fairness = True,
                            k_combinations=2)
    return [workload]

def run_multiple_workloads(num_of_workloads):
    create_workloads_from_file("../data/itunes-amazon", "test.csv", number_of_workloads = num_of_workloads)
    predictions = run_deepmatcher("../data/itunes-amazon", epochs = 2)
    workloads = []
    for i in range(0, num_of_workloads):
        test_file = "test" + str(i) + ".csv"
        workload_i = wl.Workload(pd.read_csv("../data/itunes-amazon/" + test_file), "left_Genre", 
                            "right_Genre", predictions, label_column = "label", 
                            multiple_sens_attr = True, delimiter = ",", single_fairness = False,
                            k_combinations=1)
        workloads.append(workload_i)
    return workloads

    
def main():
    workloads = run_multiple_workloads(40)
    alpha = 0.05
    threshold = 0.6
    
    fairEM = fem.FairEM(workloads, alpha, threshold, "../data/itunes-amazon", "test.csv", single_fairness=True)

    # for measure in ["accuracy_parity", "misclassification_rate_parity"]:
    #     # for aggregate in ["max", "min", "max_minus_min", "average", "distribution"]:
    #     for aggregate in ["distribution"]:
    #         res = fairEM.is_fair(measure, aggregate)
    #         print(measure, aggregate)
    #         pprint(res)
        
    # values_acc_par = workload.fairness(k_combs, "accuracy_parity")
    # values_miss_rate_par = workload.fairness(k_combs, "misclassification_rate_parity")

    # plot_bargraph(values_acc_par, "accuracy_parity_single")
    # plot_bargraph(values_miss_rate_par, "misclassification_rate_parity_single")

    # subgroups_list = [x for x in k_combs]
    
    # print("LIST of SUBGROUPS = ", subgroups_list)
    # print("k_combs", k_combs)

    is_fair_distribution = fairEM.is_fair("misclassification_rate_parity", "distribution")

    pprint(is_fair_distribution)


    # for subgroup in is_fair_distribution:
    #     if not is_fair_distribution[subgroup]:
    #         fairEM.distance_analysis(subgroup)

    
    
    # print("TYPE = ", type(k_combs))
    # print("DISTRIBUTION = ", is_fair_distribution)
    # subgroup_indices_unfair = [x for x in range(len(k_combs)) if not is_fair_distribution[x]]
    # unfair_index = subgroup_indices_unfair[0]
    # print("UNFAIR INDEX = ", unfair_index)
    # print("ALL UNFAIR = ", subgroup_indices_unfair)


    # fairEM.create_fairness_per_bin(subgroups_list, unfair_index, k_combs)



    

main()