from preprocessing import run_deepmatcher, jsonl_to_predictions
from pprint import pprint
from create_multiple_workloads import create_workloads_from_file
import workloads as wl
import pandas as pd
import FairEM as fem
import numpy as np
import matplotlib.pyplot as plt

def plot_bargraph(data, filename, title=""):
    data.sort()
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(data)
    plt.title(title)
    plt.savefig(filename + ".png")
    plt.close()

def plot_bins_to_conf_matrix(axarr, bins_to_conf_matrix, subgroup, title, location, ax):
    keys = []
    vals = []
    for key in sorted(bins_to_conf_matrix):
        keys.append(key)
        val = (bins_to_conf_matrix[key][0] + bins_to_conf_matrix[key][2]) / sum(bins_to_conf_matrix[key])
        vals.append(val)        

    # fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    axarr.plot(keys, vals)
    subgroup = subgroup.replace("/","") # to deal with cases like Hip/Hop 
    subgroup = subgroup.replace("\\","")
    title += subgroup
    plt.title(title)
    axarr.set_xticks(keys)
    plt.savefig(location + title + ".png")
    plt.close()            

def plot_results_in_2d_heatmap(dataset, data, xlabels, ylabels, title, shrink=0.5):
    fig, ax = plt.subplots(figsize=(20,14))
    im = ax.imshow(data)
    fig.colorbar(im, shrink=0.5)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(labels=xlabels)
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(labels=ylabels)

    plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
            rotation_mode="anchor")
        
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig("../experiments/" + dataset + "/" + title.replace("\n","") + ".png")

def run_one_workload(model, epochs=10, single_fairness=True):
    if model == "deepmatcher":
        predictions = run_deepmatcher("../data/itunes-amazon", epochs = epochs)
    elif model == "ditto":
        predictions = jsonl_to_predictions("../data/itunes-amazon/", "ditto_out_test.jsonl")

    workload = wl.Workload(pd.read_csv("../data/itunes-amazon/test.csv"), "left_Genre", 
                            "right_Genre", predictions, label_column = "label", 
                            multiple_sens_attr = True, delimiter = ",", single_fairness = single_fairness,
                            k_combinations=1)
    return [workload]

def run_multiple_workloads(model, num_of_workloads, epochs=10, k_combs = 1, single_fairness = True):
    # create_workloads_from_file("../data/itunes-amazon", "test.csv", number_of_workloads = num_of_workloads)
    workloads = []
    for i in range(0, num_of_workloads):
        test_file = "test" + str(i) + ".csv"
        ditto_file = "output_iTunes-Amazon" + str(i) + ".jsonl"
        if model == "deepmatcher":
            predictions = run_deepmatcher("../data/itunes-amazon", test=test_file, epochs = epochs)
        elif model == "ditto":
            predictions = jsonl_to_predictions("../data/itunes-amazon/ditto_out", ditto_file)
        workload_i = wl.Workload(pd.read_csv("../data/itunes-amazon/" + test_file), "left_Genre", 
                            "right_Genre", predictions, label_column = "label", 
                            multiple_sens_attr = True, delimiter = ",", single_fairness = single_fairness,
                            k_combinations=k_combs)
        workloads.append(workload_i)
    return workloads

def experiment_one(model, epochs):
    workloads = run_one_workload(model, epochs=epochs, single_fairness=True)
    fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/itunes-amazon", 
                        full_workload_test="test.csv", threshold=0.1, single_fairness=True)

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


    title = model + "0.1" + "Experiment 1: \nActual Fairness Values For 1-subgroups and Single Fairness and 1 workload"
    plot_results_in_2d_heatmap(actual_fairness, attribute_names, 
                                measures, title)
    title = model + "0.1" + "Experiment 1: \nBinary Fairness Values For 1-subgroups and Single Fairness and 1 workload"
    plot_results_in_2d_heatmap(binary_fairness, attribute_names, 
                                measures, title)
    
def experiment_two(model, epochs):
    workloads = run_one_workload(model, epochs=epochs, single_fairness=False)
    fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/itunes-amazon", 
                        full_workload_test="test.csv", single_fairness=False)

    binary_fairness = []
    measures = ["accuracy_parity", "statistical_parity", 
                "true_positive_rate_parity", "false_positive_rate_parity",
                "false_negative_rate_parity", "true_negative_rate_parity"]
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

    title = model + "Experiment 2: \nBinary Fairness Values For 1-subgroups and Pairwise Fairness and 1 workload"
    plot_results_in_2d_heatmap(binary_fairness, attribute_names, 
                                measures, title)
    title = model + "Experiment 2: \nActual Fairness Values For 1-subgroups and Pairwise Fairness and 1 workload"
    plot_results_in_2d_heatmap(actual_fairness, attribute_names, 
                                measures, title)

def experiment_three(model, single_fairness, epochs):
    workloads = run_multiple_workloads(model, num_of_workloads=40, epochs=epochs, single_fairness=single_fairness)
    fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/itunes-amazon", 
                        full_workload_test="test.csv", single_fairness=single_fairness)

    fairness = []
    if single_fairness:
        measures = ["accuracy_parity", "statistical_parity", \
                    "true_positive_rate_parity", "false_positive_rate_parity", \
                    "false_negative_rate_parity", "true_negative_rate_parity", \
                    "negative_predictive_value_parity", "false_discovery_rate_parity", \
                    "false_omission_rate_parity"]
    else:
        measures = ["accuracy_parity", "statistical_parity", \
                    "true_positive_rate_parity", \
                    "false_positive_rate_parity", \
                    "false_negative_rate_parity", \
                    "true_negative_rate_parity"]
    aggregate = "distribution"
    fairness_keys = None
    for measure in measures:
        is_fair = fairEM.is_fair(measure, aggregate)
        if fairness_keys == None:
            fairness_keys = is_fair.keys()
        fairness.append(is_fair)

    fairness_values = []
    
    for x in fairness:
        curr = []
        for key in fairness_keys:
            curr.append(x[key])
        fairness_values.append(curr)
    
    
    title = model + "Experiment 3: \nBinary Fairness Values For 1-subgroups and Single Fairness and 40 workloads" if single_fairness else model + "Experiment 3: \nBinary Fairness Values For 1-subgroups and Pairwise Fairness and 40 workloads"
    plot_results_in_2d_heatmap(fairness_values, fairness_keys, 
                                measures, title)

def experiment_four(model, epochs):
    measures = ["accuracy_parity", "statistical_parity", \
                "true_positive_rate_parity", "false_positive_rate_parity", \
                "false_negative_rate_parity", "true_negative_rate_parity", \
                "negative_predictive_value_parity", "false_discovery_rate_parity", \
                "false_omission_rate_parity"]
    aggregates = ["max", "min", "max_minus_min", "average"]
    

    f, axarr = plt.subplots(3,3,figsize=(11,10))
    i = 0
    
    workloads = run_one_workload(model, epochs=epochs)
    fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/itunes-amazon", 
                    full_workload_test="test.csv", single_fairness=False)

    k_combs_ylabel = ["1-comb", "2-comb"]
    for measure in measures:
        k_comb_VS_fairness = []
        for k_comb in [1, 2]:
            curr_fairness = []
            for aggregate in aggregates:
                curr_fairness.append(fairEM.is_fair(measure, aggregate))
            
            k_comb_VS_fairness.append(curr_fairness)

        x = int(i / 3)
        y = i % 3
        
        axarr[x][y].set_title(measure)
        axarr[x][y].imshow(k_comb_VS_fairness)
        axarr[x][y].set_xticks(np.arange(len(aggregates)))
        axarr[x][y].set_xticklabels(labels=aggregates)
        axarr[x][y].set_yticks(np.arange(len(k_combs_ylabel)))
        axarr[x][y].set_yticklabels(labels=k_combs_ylabel)
        
        plt.setp(axarr[x][y].get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # print("k_comb_VS_fairness = ", k_comb_VS_fairness)
        i += 1    

    f.tight_layout()
    plt.savefig("../experiments/" + model + "Experiment 4: 1-comb and 2-comb VS AGG Functions for each measure.png")
        
def experiment_five(model, epochs, one_workload=True, single_fairness=True):
    if one_workload:
        workloads = run_one_workload(model, epochs=epochs, single_fairness=single_fairness)
    else:
        workloads = run_multiple_workloads(num_of_workloads=40, epochs=epochs, single_fairness=single_fairness)
    
    fairEM = fem.FairEM(workloads, alpha=0.05, directory="../data/itunes-amazon", 
                        full_workload_test="test.csv", single_fairness=single_fairness)

    # fixed one measure
    measure = "accuracy_parity"
    is_fair_distribution = fairEM.is_fair(measure, "distribution")
    
    multiple_bins_to_conf_matrix = []
    for subgroup in is_fair_distribution:
        print("is_fair_distribution[subgroup] = ", is_fair_distribution[subgroup])
        if not is_fair_distribution[subgroup]:
            print("NOT FAIR = ", subgroup)
            bins_to_conf_matrix = fairEM.distance_analysis(subgroup)
            multiple_bins_to_conf_matrix.append(bins_to_conf_matrix)

    f, axarr = plt.subplots(1,len(multiple_bins_to_conf_matrix),figsize=(11,10))
    for i in range(len(multiple_bins_to_conf_matrix)):
        bins_to_conf_matrix = multiple_bins_to_conf_matrix[i]
        plot_bins_to_conf_matrix(axarr[0][i], bins_to_conf_matrix, subgroup, title="Experiment 5: ", location="../experiments/")
    
def run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, epochs=10, single_fairness=True):
    if model == "deepmatcher":
        predictions = run_deepmatcher("../data/" + dataset + "/", 
                                        train="train.csv", 
                                        validation="valid.csv",
                                        test="test.csv",
                                        epochs = epochs)
    elif model == "ditto":
        predictions = jsonl_to_predictions("../data/" + dataset + "/", "ditto_out_test.jsonl")
    workload = wl.Workload(pd.read_csv("../data/" + dataset + "/test.csv"), 
                            left_sens_attribute, right_sens_attribute,
                            predictions, label_column = "label", 
                            multiple_sens_attr = True, delimiter = ",", 
                            single_fairness = single_fairness, k_combinations=1)
    return [workload]

def experiment_one(model, dataset, left_sens_attribute, right_sens_attribute, epochs=10, single_fairness=True, threshold=0.2):
    workloads = run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, 
                                epochs=epochs, single_fairness=single_fairness)
    fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/" + dataset, 
                        full_workload_test="test.csv", threshold=threshold, single_fairness=single_fairness)

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
    attribute_names = []
    for k_comb in workloads[0].k_combs_to_attr_names:
        attribute_names.append(workloads[0].k_combs_to_attr_names[k_comb])

    title = "Exp1: " + dataset + " " + model + " with " + str(threshold) + " threshold \nBinary Fairness Values For 1-subgroups and Single Fairness and 1 workload"
    plot_results_in_2d_heatmap(dataset, binary_fairness, attribute_names, 
                                measures, title)


def experiment_two(model, dataset, left_sens_attribute, right_sens_attribute, epochs=10, single_fairness=False, threshold=0.2):
    workloads = run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, 
                                epochs=epochs, single_fairness=single_fairness)
    fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/" + dataset, 
                        full_workload_test="test.csv", threshold=threshold, single_fairness=single_fairness)

    binary_fairness = []
    measures = ["accuracy_parity", "statistical_parity", \
                "true_positive_rate_parity", "false_positive_rate_parity", \
                "false_negative_rate_parity", "true_negative_rate_parity"]
    aggregate = "distribution"
    for measure in measures:
        is_fair = fairEM.is_fair(measure, aggregate)
        binary_fairness.append(is_fair)
    attribute_names = []
    for k_comb in workloads[0].k_combs_to_attr_names:
        attribute_names.append(workloads[0].k_combs_to_attr_names[k_comb])

    title = "Exp2: " + dataset + " " + model + " with " + str(threshold) + " threshold\nBinary Fairness Values For 1-subgroups and Pairwise Fairness and 1 workload"
    plot_results_in_2d_heatmap(dataset, binary_fairness, attribute_names, 
                                measures, title)


def experiment_four(left_sens_attribute, right_sens_attribute, epochs=10, single_fairness=False):
    workloads = run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, 
                                epochs=epochs, single_fairness=single_fairness)
    fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/" + dataset, 
                        full_workload_test="test.csv", threshold=0.1, single_fairness=single_fairness)
    
    measures = ["accuracy_parity", "statistical_parity", \
                "true_positive_rate_parity", "false_positive_rate_parity", \
                "false_negative_rate_parity", "true_negative_rate_parity", \
                "negative_predictive_value_parity", "false_discovery_rate_parity", \
                "false_omission_rate_parity"]
    aggregates = ["max", "min", "max_minus_min", "average"]
    

    f, axarr = plt.subplots(3,3,figsize=(11,10))
    i = 0
    
    workloads = run_one_workload(model, epochs=epochs)
    fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/itunes-amazon", 
                    full_workload_test="test.csv", single_fairness=False)

    k_combs_ylabel = ["1-comb", "2-comb"]
    for measure in measures:
        k_comb_VS_fairness = []
        for k_comb in [1, 2]:
            curr_fairness = []
            for aggregate in aggregates:
                curr_fairness.append(fairEM.is_fair(measure, aggregate))
            
            k_comb_VS_fairness.append(curr_fairness)

        x = int(i / 3)
        y = i % 3
        
        axarr[x][y].set_title(measure)
        axarr[x][y].imshow(k_comb_VS_fairness)
        axarr[x][y].set_xticks(np.arange(len(aggregates)))
        axarr[x][y].set_xticklabels(labels=aggregates)
        axarr[x][y].set_yticks(np.arange(len(k_combs_ylabel)))
        axarr[x][y].set_yticklabels(labels=k_combs_ylabel)
        
        plt.setp(axarr[x][y].get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # print("k_comb_VS_fairness = ", k_comb_VS_fairness)
        i += 1    

    f.tight_layout()
    plt.savefig("../experiments/" + dataset + "/" + model + "Experiment 4: 1-comb and 2-comb VS AGG Functions for each measure.png")
    



def full_experiment_one(threshold=0.2):
    experiment_one(model="ditto", dataset="dblp-acm", 
                    left_sens_attribute="left_venue", 
                    right_sens_attribute="right_venue", 
                    epochs=2, single_fairness=True,
                    threshold=threshold)
    experiment_one(model="deepmatcher", dataset="dblp-acm", 
                    left_sens_attribute="left_venue", 
                    right_sens_attribute="right_venue", 
                    epochs=2, single_fairness=True,
                    threshold=threshold)
    experiment_one(model="ditto", dataset="itunes-amazon", 
                    left_sens_attribute="left_Genre", 
                    right_sens_attribute="right_Genre", 
                    epochs=2, single_fairness=True,
                    threshold=threshold)
    experiment_one(model="deepmatcher", dataset="itunes-amazon", 
                    left_sens_attribute="left_Genre", 
                    right_sens_attribute="right_Genre", 
                    epochs=2, single_fairness=True,
                    threshold=threshold)
    experiment_one(model="ditto", dataset="shoes", 
                    left_sens_attribute="left_locale", 
                    right_sens_attribute="right_locale", 
                    epochs=2, single_fairness=True,
                    threshold=threshold)
    experiment_one(model="deepmatcher", dataset="shoes", 
                    left_sens_attribute="left_locale", 
                    right_sens_attribute="right_locale", 
                    epochs=2, single_fairness=True,
                    threshold=threshold)

def full_experiment_two(threshold=0.2):
    experiment_two(model="ditto", dataset="dblp-acm", 
                    left_sens_attribute="left_venue", 
                    right_sens_attribute="right_venue", 
                    epochs=2, single_fairness=False,
                    threshold=threshold)
    experiment_two(model="deepmatcher", dataset="dblp-acm", 
                    left_sens_attribute="left_venue", 
                    right_sens_attribute="right_venue", 
                    epochs=2, single_fairness=False,
                    threshold=threshold)
    experiment_two(model="ditto", dataset="itunes-amazon", 
                    left_sens_attribute="left_Genre", 
                    right_sens_attribute="right_Genre", 
                    epochs=2, single_fairness=False,
                    threshold=threshold)
    experiment_two(model="deepmatcher", dataset="itunes-amazon", 
                    left_sens_attribute="left_Genre", 
                    right_sens_attribute="right_Genre", 
                    epochs=2, single_fairness=False,
                    threshold=threshold)
    experiment_two(model="ditto", dataset="shoes", 
                    left_sens_attribute="left_locale", 
                    right_sens_attribute="right_locale", 
                    epochs=2, single_fairness=False,
                    threshold=threshold)
    experiment_two(model="deepmatcher", dataset="shoes", 
                    left_sens_attribute="left_locale", 
                    right_sens_attribute="right_locale", 
                    epochs=2, single_fairness=False,
                    threshold=threshold)

def main():
    full_experiment_one(threshold=0.1)
    full_experiment_one(threshold=0.2)
    full_experiment_two(threshold=0.1)
    full_experiment_two(threshold=0.2)
    
    


    # workloads = run_multiple_workloads(num_of_workloads=40, epochs=2)
    # alpha = 0.05
    # threshold = 0.2
    
    # fairEM = fem.FairEM(workloads, alpha, "../data/itunes-amazon", "test.csv", threshold, single_fairness=False)

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

    # experiment_one_dblp(model = "ditto", epochs=2)
    # experiment_one_dblp(model = "deepmatcher", epochs=2)

    
    
    
    
    
    

    # experiment_one_shoes(model = "deepmatcher", epochs=2)
    
    # experiment_one(model = "deepmatcher", epochs=2)
    # experiment_one(model = "ditto", epochs=2)
    
    # experiment_two(model = "deepmatcher", epochs=2)
    # experiment_two(model = "ditto", epochs=2)
    
    # experiment_three(model = "deepmatcher", single_fairness=True, epochs=2)
    # experiment_three(model = "deepmatcher", single_fairness=False, epochs=2)
    # experiment_three(model = "ditto", single_fairness=True, epochs=2)
    # experiment_three(model = "ditto", single_fairness=False, epochs=2)

    # experiment_four(model = "ditto", epochs=2)
    # experiment_four(model = "deepmatcher", epochs=2)
    
    # experiment_five(epochs=2, one_workload=True, single_fairness=True)
    

main()