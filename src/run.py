from preprocessing import run_deepmatcher, jsonl_to_predictions, deepmatcher_output_to_predictions
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

def plot_bins_to_conf_matrix(ax, bins_to_conf_matrix, subgroup):
    keys = []
    vals = []
    for key in sorted(bins_to_conf_matrix):
        keys.append(key)
        val = (bins_to_conf_matrix[key][0] + bins_to_conf_matrix[key][2]) / sum(bins_to_conf_matrix[key])
        vals.append(val)        

    # fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(keys, vals)
    subgroup = subgroup.replace("/","") # to deal with cases like Hip/Hop 
    subgroup = subgroup.replace("\\","")
    ax.set_title(subgroup)
    ax.set_xticks(keys)

def plot_results_in_2d_heatmap(dataset, data, xlabels, ylabels, title, shrink=0.5):
    fig, ax = plt.subplots(figsize=(20,14))
    im = ax.imshow(data, vmin=0, vmax=1)
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


def experiment_three(dataset, model, single_fairness, epochs):
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
    plot_results_in_2d_heatmap(dataset, fairness_values, fairness_keys, 
                                measures, title)
       
def experiment_five(model, epochs, one_workload=True, single_fairness=True):
    dataset = "itunes-amazon"
    left_sens_attribute = "left_Genre"
    right_sens_attribute = "right_Genre"
    
    if one_workload:
        workloads = run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, 
                                epochs=epochs, single_fairness=single_fairness)
    else:
        workloads = run_multiple_workloads(num_of_workloads=40, epochs=epochs, single_fairness=single_fairness)
    
    fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/" + dataset, 
                        full_workload_test="test.csv", single_fairness=single_fairness)

    # fixed one measure
    measure = "accuracy_parity"
    is_fair_distribution = fairEM.is_fair(measure, "distribution")

    attribute_names = []
    for k_comb in workloads[0].k_combs_to_attr_names:
        attribute_names.append(workloads[0].k_combs_to_attr_names[k_comb])

    multiple_bins_to_conf_matrix = []
    unfair_subgroups = []

    for i in range(len(is_fair_distribution)):
        subgroup = attribute_names[i]
        val = is_fair_distribution[i]
        if not val:
            bins_to_conf_matrix = fairEM.distance_analysis(subgroup)
            multiple_bins_to_conf_matrix.append(bins_to_conf_matrix)
            unfair_subgroups.append(subgroup)

    f, axarr = plt.subplots(2, int(len(multiple_bins_to_conf_matrix) / 2),figsize=(11,10))
    for i in range(len(multiple_bins_to_conf_matrix)):
        x = int(i / 2)
        y = int(i % 2)
        bins_to_conf_matrix = multiple_bins_to_conf_matrix[i]
        plot_bins_to_conf_matrix(axarr[y][x], bins_to_conf_matrix, unfair_subgroups[i])
    plt.savefig("../experiments/itunes-amazon/" + "Exp5: " + model + " Distance Explainability" + ".png")
    plt.close()            

    
def run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, epochs=10, single_fairness=True, k_combinations=1, delimiter = ','):
    if model == "deepmatcher":
        predictions = deepmatcher_output_to_predictions("../data/" + dataset + "/", "deepmatcher_out_15.txt")
    elif model == "ditto":
        predictions = jsonl_to_predictions("../data/" + dataset + "/", "ditto_out_test.jsonl")
    workload = wl.Workload(pd.read_csv("../data/" + dataset + "/test.csv"), 
                            left_sens_attribute, right_sens_attribute,
                            predictions, label_column = "label", 
                            multiple_sens_attr = True, delimiter = delimiter, 
                            single_fairness = single_fairness, k_combinations=k_combinations)
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

def experiment_four(dataset, model, left_sens_attribute, right_sens_attribute, epochs, k_combinations):
    measures = ["accuracy_parity", "statistical_parity", \
                "true_positive_rate_parity", "false_positive_rate_parity", \
                "false_negative_rate_parity", "true_negative_rate_parity", \
                "negative_predictive_value_parity", "false_discovery_rate_parity", \
                "false_omission_rate_parity"]
    aggregates = ["max", "min", "max_minus_min", "average"]
    

    f, axarr = plt.subplots(3,3,figsize=(11,10))
    
    k_combs_ylabel = [str(x)+"-comb" for x in range(1, k_combinations+1)]
    k_combs = [x for x in range(1, k_combinations+1)]
    k_combs_fairness_measure = []
    for i in range(len(k_combs)):
        k_comb = k_combs[i]
        k_comb_label = k_combs_ylabel[i]
        k_comb_VS_fairness = []
        workloads = run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, epochs=epochs, single_fairness=True, k_combinations=k_comb)
        fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/" + dataset, 
                    full_workload_test="test.csv", single_fairness=False)

        for measure in measures:
            curr_fairness = []
            for aggregate in aggregates:
                curr_fairness.append(fairEM.is_fair(measure, aggregate))
            
            k_comb_VS_fairness.append(curr_fairness)
    
        k_combs_fairness_measure.append(k_comb_VS_fairness)

    for i in range(len(measures)):
        k_comb_VS_fairness = []
        for j in range(len(k_combs)):
            curr_fairness = []
            for k in range(len(aggregates)):
                curr_fairness.append(k_combs_fairness_measure[j][i][k])
            
            k_comb_VS_fairness.append(curr_fairness)
    

        x = int(i / 3)
        y = i % 3
        
        axarr[x][y].set_title(measures[i])
        axarr[x][y].imshow(k_comb_VS_fairness, vmin=0, vmax=1)
        axarr[x][y].set_xticks(np.arange(len(aggregates)))
        axarr[x][y].set_xticklabels(labels=aggregates)
        axarr[x][y].set_yticks(np.arange(len(k_combs_ylabel)))
        axarr[x][y].set_yticklabels(labels=k_combs_ylabel)
        
        plt.setp(axarr[x][y].get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

    f.tight_layout()
    plt.savefig("../experiments/" + dataset + "/Exp4: " + model + " General Model Fairness.png")


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

def full_experiment_four():
    experiment_four(dataset = "itunes-amazon", model = "ditto", 
                    left_sens_attribute="left_Genre", 
                    right_sens_attribute="right_Genre",
                    epochs=15, k_combinations=2)
    experiment_four(dataset = "itunes-amazon", model = "deepmatcher", 
                    left_sens_attribute="left_Genre", 
                    right_sens_attribute="right_Genre",
                    epochs=15, k_combinations=2)
    experiment_four(dataset = "dblp-acm", model = "ditto", 
                    left_sens_attribute="left_venue", 
                    right_sens_attribute="right_venue",
                    epochs=15, k_combinations=1)
    experiment_four(dataset = "dblp-acm", model = "deepmatcher", 
                    left_sens_attribute="left_venue", 
                    right_sens_attribute="right_venue",
                    epochs=15, k_combinations=1)
    experiment_four(dataset = "shoes", model = "ditto", 
                    left_sens_attribute="left_locale", 
                    right_sens_attribute="right_locale",
                    epochs=15, k_combinations=1)
    experiment_four(dataset = "shoes", model = "deepmatcher", 
                    left_sens_attribute="left_locale", 
                    right_sens_attribute="right_locale",
                    epochs=15, k_combinations=1)

def input_distributions(dataset, left_sens_attribute, right_sens_attribute):
    # model is irrelevant
    workloads = run_one_workload("deepmatcher", dataset, left_sens_attribute, right_sens_attribute, 
                                epochs=15, single_fairness=True)
    attr_value_frequency = {}
    workload = workloads[0]
    for attr_val in workload.sens_attr_vals:
        attr_value_frequency[attr_val] = 0
    
    for idx, row in workload.df.iterrows():
        left = row[workload.sens_att_left]
        right = row[workload.sens_att_right]
        if workload.multiple_sens_attr:
            for item in left.split(workload.delimiter):
                sens_att = item.strip()
                attr_value_frequency[sens_att] += 1
            for item in right.split(workload.delimiter):
                sens_att = item.strip()
                attr_value_frequency[sens_att] += 1
        else:
            sens_att = left.strip()
            attr_value_frequency[sens_att] += 1
            sens_att = right.strip()
            attr_value_frequency[sens_att] += 1
    
    pprint(attr_value_frequency)
    
def case_study():
    model = "deepmatcher"
    dataset = "shoes"
    left_sens_attribute = "left_locale"
    right_sens_attribute = "right_locale"
    workloads = run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, epochs=15, single_fairness=True, k_combinations=1, delimiter=';')
    fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/" + dataset, 
                    full_workload_test="test.csv", single_fairness=False)

    pprint(fairEM.workloads[0].sens_attr_vals)


def debugging():
    model = "ditto"
    dataset = "itunes-amazon"
    left_sens_attribute = "left_Genre"
    right_sens_attribute = "right_Genre"
    single_fairness=True
    threshold=0.2

    workloads = run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, 
                                epochs=15, single_fairness=single_fairness)
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

    workload = workloads[0]
    print("len(workload.k_combs) = ", len(workload.k_combs))
    print("workload.k_combs = ", workload.k_combs)
    print("len(attribute_names) = ", len(attribute_names))
    print("attribute_names = ", attribute_names)
    print("len(workload.sens_attr_vals) = ", len(workload.sens_attr_vals))
    print("workload.sens_attr_vals = ", workload.sens_attr_vals)


    experiment_one(model="ditto", dataset="dblp-acm", 
                    left_sens_attribute="left_venue", 
                    right_sens_attribute="right_venue", 
                    epochs=2, single_fairness=True,
                    threshold=threshold)
    
def main():
    # debugging()

    # case_study()
    # full_experiment_one(threshold=0.1)
    full_experiment_one(threshold=0.2)
    # full_experiment_two(threshold=0.1)
    # full_experiment_two(threshold=0.2)
    # full_experiment_four()
    # experiment_five(model = "ditto", epochs=15)
    # experiment_five(model = "deepmatcher", epochs=15)

    # experiment_three(dataset = "itunes-amazon", model = "ditto", single_fairness=True, epochs=2)
    
    # input_distributions("itunes-amazon", "left_Genre", "right_Genre")
    # input_distributions("dblp-acm", "left_venue", "right_venue")
    # input_distributions("shoes", "left_locale", "right_locale")
    
    
    
    
    


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


    # experiment_three(model = "deepmatcher", single_fairness=True, epochs=2)
    # experiment_three(model = "deepmatcher", single_fairness=False, epochs=2)
    # experiment_three(model = "ditto", single_fairness=True, epochs=2)
    # experiment_three(model = "ditto", single_fairness=False, epochs=2)

    
    # experiment_five(epochs=2, one_workload=True, single_fairness=True)
    

    

main()