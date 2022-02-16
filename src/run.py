from preprocessing import run_deepmatcher
from pprint import pprint
from create_multiple_workloads import create_workloads_from_file
import workloads as wl
import pandas as pd
import FairEM as fem
import matplotlib.pyplot as plt

def plot_bargraph(data, filename, title=""):
    data.sort()
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(data)
    plt.title(title)
    plt.savefig(filename + ".png")
    plt.close()

def plot_bins_to_conf_matrix(bins_to_conf_matrix, subgroup, title, location):
    keys = []
    vals = []
    for key in sorted(bins_to_conf_matrix):
        keys.append(key)
        val = (bins_to_conf_matrix[key][0] + bins_to_conf_matrix[key][2]) / sum(bins_to_conf_matrix[key])
        vals.append(val)        

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(keys, vals)
    subgroup = subgroup.replace("/","") # to deal with cases like Hip/Hop 
    subgroup = subgroup.replace("\\","")
    title += subgroup
    plt.title(title)
    ax.set_xticks(keys)
    plt.savefig(location + title + ".png")
    plt.close()            

def run_one_workload(epochs=10, single_fairness=True):
    predictions = run_deepmatcher("../data/itunes-amazon", epochs = epochs)
    workload = wl.Workload(pd.read_csv("../data/itunes-amazon/test.csv"), "left_Genre", 
                            "right_Genre", predictions, label_column = "label", 
                            multiple_sens_attr = True, delimiter = ",", single_fairness = single_fairness,
                            k_combinations=1)
    return [workload]

def run_multiple_workloads(num_of_workloads, epochs=10, k_combs = 1):
    create_workloads_from_file("../data/itunes-amazon", "test.csv", number_of_workloads = num_of_workloads)
    predictions = run_deepmatcher("../data/itunes-amazon", epochs = epochs)
    workloads = []
    for i in range(0, num_of_workloads):
        test_file = "test" + str(i) + ".csv"
        workload_i = wl.Workload(pd.read_csv("../data/itunes-amazon/" + test_file), "left_Genre", 
                            "right_Genre", predictions, label_column = "label", 
                            multiple_sens_attr = True, delimiter = ",", single_fairness = True,
                            k_combinations=k_combs)
        workloads.append(workload_i)
    return workloads

def experiment_one(epochs):
    workloads = run_one_workload(epochs=epochs, single_fairness=True)
    fairEM = fem.FairEM(workloads, alpha=0.05, directory="../data/itunes-amazon", 
                        full_workload_test="test.csv", single_fairness=True)

    binary_fairness = []
    measures = ["accuracy_parity", "statistical_parity", \
                "true_positive_rate_parity", "false_positive_rate_parity", \
                "false_negative_rate_parity", "true_negative_rate_parity", \
                "negative_predictive_value_parity", "false_discovery_rate_parity", \
                "false_omission_rate_parity"]
    aggregate = "distribution"
    for measure in measures:
        is_fair = fairEM.is_fair(measure, aggregate)
        binary_fairness.append(is_fair)

    actual_fairness = []
    for measure in measures:
        is_fair = fairEM.is_fair(measure, aggregate, real_distr = True)
        actual_fairness.append(is_fair)

    attribute_names = []
    for k_comb in workloads[0].k_combs_to_attr_names:
        attribute_names.append(workloads[0].k_combs_to_attr_names[k_comb])

    print("ATTRIBUTE NAMES = ", attribute_names)


    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plt.imshow(binary_fairness, cmap='bwr', interpolation='nearest')
    title = "Experiment 1: \nBinary Fairness Values For 1-subgroups and Single Fairness and 1 workload"
    plt.title(title)
    plt.savefig("../experiments/" + title.replace("\n","") + ".png")
    plt.close()

    print("BINARY = ", binary_fairness)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plt.imshow(actual_fairness, cmap='bwr', interpolation='nearest')
    title = "Experiment 1: \nActual Fairness Values For 1-subgroups and Single Fairness and 1 workload"
    plt.title(title)
    plt.savefig("../experiments/" + title.replace("\n","") + ".png")
    plt.close()

    print("Actual = ", actual_fairness)


def experiment_two(epochs):
    workloads = run_one_workload(epochs=epochs, single_fairness=False)
    fairEM = fem.FairEM(workloads, alpha=0.05, directory="../data/itunes-amazon", 
                        full_workload_test="test.csv", single_fairness=False)

    binary_fairness = []
    measures = ["accuracy_parity", "statistical_parity", "true_positive_rate_parity"]
    aggregate = "distribution"
    for measure in measures:
        is_fair = fairEM.is_fair(measure, aggregate)
        binary_fairness.append(is_fair)

    actual_fairness = []
    for measure in measures:
        is_fair = fairEM.is_fair(measure, aggregate, real_distr = True)
        actual_fairness.append(is_fair)

    

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plt.imshow(binary_fairness, cmap='bwr', interpolation='nearest')
    title = "Experiment 2: \nBinary Fairness Values For 1-subgroups and Pairwise Fairness and 1 workload"
    plt.title(title)
    plt.savefig("../experiments/" + title.replace("\n","") + ".png")
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plt.imshow(actual_fairness, cmap='bwr', interpolation='nearest')
    title = "Experiment 2: \nActual Fairness Values For 1-subgroups and Pairwise Fairness and 1 workload"
    plt.title(title)
    plt.savefig("../experiments/" + title.replace("\n","") + ".png")
    plt.close()

def experiment_three(single_fairness, epochs):
    workloads = run_multiple_workloads(num_of_workloads=40, epochs=epochs)
    fairEM = fem.FairEM(workloads, alpha=0.05, directory="../data/itunes-amazon", 
                        full_workload_test="test.csv", single_fairness=single_fairness)

    fairness = []
    measures = ["accuracy_parity"]
    aggregate = "distribution"
    for measure in measures:
        is_fair = fairEM.is_fair(measure, aggregate)
        fairness.append(is_fair)

    fairness_values = [list(x.values()) for x in fairness]
    
    print(fairness_values)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plt.imshow(fairness_values, cmap='bwr', interpolation='nearest')
    title = "Experiment 3: \nBinary Fairness Values For 1-subgroups and Single Fairness and 40 workloads" if single_fairness else "Experiment 3: \nBinary Fairness Values For 1-subgroups and Pairwise Fairness and 40 workloads"
    plt.title(title)
    plt.savefig("../experiments/" + title.replace("\n","") + ".png")
    plt.close()

# TODO: FINISH THE COLORING NICELY
# def experiment_four(epochs):
#     measures = ["accuracy_parity"]
#     aggregates = ["max", "min", "max_minus_min", "average"]
#     for measure in measures:
#         k_comb_VS_fairness = []
#         for k_comb in [1, 2]:
#             curr_fairness = []
#             workloads = run_one_workload(epochs=epochs)
#             fairEM = fem.FairEM(workloads, alpha=0.05, directory="../data/itunes-amazon", 
#                             full_workload_test="test.csv", single_fairness=False)

#             for aggregate in aggregates:
#                 curr_fairness.append(fairEM.is_fair(measure, aggregate))
            
#             k_comb_VS_fairness.append(curr_fairness)

#         fig, ax = plt.subplots(1, 1, figsize=(7, 5))
#         plt.imshow(k_comb_VS_fairness, cmap='bwr', interpolation='nearest')
#         title = "Experiment 4: \n1-comb and 2-comb VS AGG Functions and measure = " + measure
#         plt.title(title)
#         ax.
#         plt.savefig("../experiments/" + title.replace("\n","") + ".png")
#         plt.close()

def experiment_five(epochs):
    workloads = run_multiple_workloads(num_of_workloads=40, epochs=epochs)
    fairEM = fem.FairEM(workloads, alpha=0.05, directory="../data/itunes-amazon", 
                        full_workload_test="test.csv", single_fairness=False)

    # fixed one measure
    measure = ["accuracy_parity"]
    is_fair_distribution = fairEM.is_fair(measure, "distribution")

    for subgroup in is_fair_distribution:
        if not is_fair_distribution[subgroup]:
            bins_to_conf_matrix = fairEM.distance_analysis(subgroup)
            plot_bins_to_conf_matrix(bins_to_conf_matrix, subgroup, title="Experiment 5: ", location="../experiments/")


    
            



    
def main():
    # workloads = run_multiple_workloads(40)
    # alpha = 0.05
    # threshold = 0.6
    
    # fairEM = fem.FairEM(workloads, alpha, threshold, "../data/itunes-amazon", "test.csv", single_fairness=True)

    # for measure in ["accuracy_parity"]:
    #     # for aggregate in ["max", "min", "max_minus_min", "average", "distribution"]:
    #     for aggregate in ["distribution"]:
    #         res = fairEM.is_fair(measure, aggregate)
    #         print(measure, aggregate)
    #         pprint(res)
        
    # values_acc_par = workload.fairness(k_combs, "accuracy_parity")
    
    # plot_bargraph(values_acc_par, "accuracy_parity_single")
    
    # subgroups_list = [x for x in k_combs]
    
    # print("LIST of SUBGROUPS = ", subgroups_list)
    # print("k_combs", k_combs)

    # is_fair_distribution = fairEM.is_fair("accuracy_rate_parity", "distribution")

    # pprint(is_fair_distribution)


    # for subgroup in is_fair_distribution:
    #     if not is_fair_distribution[subgroup]:
    #         bins_to_conf_matrix = fairEM.distance_analysis(subgroup)
    #         plot_bins_to_conf_matrix(bins_to_conf_matrix, subgroup)

    
    
    # print("TYPE = ", type(k_combs))
    # print("DISTRIBUTION = ", is_fair_distribution)
    # subgroup_indices_unfair = [x for x in range(len(k_combs)) if not is_fair_distribution[x]]
    # unfair_index = subgroup_indices_unfair[0]
    # print("UNFAIR INDEX = ", unfair_index)
    # print("ALL UNFAIR = ", subgroup_indices_unfair)


    # fairEM.create_fairness_per_bin(subgroups_list, unfair_index, k_combs)


    experiment_one(epochs=2)
    # experiment_two(epochs=2)
    # experiment_three(single_fairness=True, epochs=10)
    # experiment_three(single_fairness=False, epochs=10)
    # experiment_four(epochs=2)
    # experiment_five(epochs=2)
    

main()