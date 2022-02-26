from preprocessing import run_deepmatcher, jsonl_to_predictions, deepmatcher_output_to_predictions
from pprint import pprint
import workloads as wl
import pandas as pd
import FairEM as fem
import numpy as np
import matplotlib.pyplot as plt
from measures import *

def make_acronym(word, delim):
    res = ""
    for spl in word.split(delim):
        res += spl[0].capitalize()
    return res

def plot_bargraph(data, filename, title=""):
    data.sort()
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(data)
    plt.title(title)
    plt.savefig(filename + ".png")
    plt.close()

# def plot_bins_to_conf_matrix(ax, bins_to_conf_matrix, subgroup):
#     keys = []
#     vals = []
#     for key in sorted(bins_to_conf_matrix):
#         keys.append(key)
#         val = (bins_to_conf_matrix[key][0] + bins_to_conf_matrix[key][2]) / sum(bins_to_conf_matrix[key])
#         vals.append(val)        

#     ax.plot(keys, vals)
#     subgroup = subgroup.replace("/","") # to deal with cases like Hip/Hop 
#     subgroup = subgroup.replace("\\","")
#     ax.set_title(subgroup)
#     ax.set_xticks(keys)

def plot_results_in_2d_heatmap(dataset, data, xlabels, ylabels, title, figsize = (10,7), x_font = 9, y_font = 12):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(labels=xlabels, fontdict={'fontsize': x_font})
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(labels=ylabels, fontdict={'fontsize': y_font})

    plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
            rotation_mode="anchor")
        
    # ax.set_title(title, fontdict={'fontsize': 24})
    fig.tight_layout()
    plt.savefig("../experiments/" + dataset + "/" + title.replace("\n","") + ".png", dpi=100)
    plt.close()

def run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, 
                    epochs=10, single_fairness=True, 
                    k_combinations=1, delimiter = ',',
                    test_file = "test.csv"):
    if model == "deepmatcher":
        predictions = deepmatcher_output_to_predictions("../data/" + dataset + "/", "deepmatcher_out_15.txt")
    elif model == "ditto":
        predictions = jsonl_to_predictions("../data/" + dataset + "/", "ditto_out_test.jsonl")
    workload = wl.Workload(pd.read_csv("../data/" + dataset + "/" + test_file), 
                            left_sens_attribute, right_sens_attribute,
                            predictions, label_column = "label", 
                            multiple_sens_attr = True, delimiter = delimiter, 
                            single_fairness = single_fairness, k_combinations=k_combinations)
    return [workload]

def run_multiple_workloads(dataset, model, num_of_workloads=40, epochs=10, k_combs = 1, single_fairness = True, delimiter=",", others=True):
    workloads = []
    if dataset == "itunes-amazon":
        left_sens_attribute = "left_Genre"
        right_sens_attribute = "right_Genre"
    elif dataset == "dblp-acm":
        left_sens_attribute = "left_venue"
        right_sens_attribute = "right_venue"
    elif dataset == "shoes":
        left_sens_attribute = "left_locale"
        right_sens_attribute = "right_locale"

    
    for i in range(0, num_of_workloads):
        test_file = "test_others" + str(i) + ".csv" if others== True else "test" + str(i) + ".csv"
        ditto_file = "ditto_out_test_" + str(i) + ".jsonl"
        deepmatcher_file = "deepmatcher_out_15_" + str(i) + ".txt"

        if model == "deepmatcher":
            predictions = deepmatcher_output_to_predictions("../data/" + dataset + "/multiple_workloads/", deepmatcher_file)
        elif model == "ditto":
            predictions = jsonl_to_predictions("../data/" + dataset + "/multiple_workloads/", ditto_file)
        workload_i = wl.Workload(pd.read_csv("../data/" + dataset + "/multiple_workloads/" + test_file), 
                                left_sens_attribute, right_sens_attribute,
                                predictions, label_column = "label", 
                                multiple_sens_attr = True, delimiter = delimiter, 
                                single_fairness = single_fairness, k_combinations=k_combs)

        workloads.append(workload_i)
    return workloads

def plot_distances_all(distances_all, label, model, distance_to_bin, measure = "accuracy"):
    plot = {}
    ls_distances = list(distances_all.keys())
    ls_distances.sort()
    for distance in ls_distances:
        binn = distance_to_bin[distance]
        if binn not in plot:
            plot[binn] = [0,0,0,0]
        plot[binn] = list(np.add(plot[binn], distances_all[distance]))

    pprint(plot)
    for binn in plot:
        TP, FP, TN, FN = tuple(plot[binn])
        if measure == "true_positive_rate":
            plot[binn] = TPR(TP, FP, TN, FN)
        elif measure == "accuracy":
            plot[binn] = AP(TP, FP, TN, FN)

    pprint(plot)
    
    lists = sorted(plot.items()) 
    x, y = zip(*lists)

    plt.plot(x, y)
      
def create_distance_to_bin(distances, n = 4):
    distance_to_bin = {}
    x = np.array_split(distances, n)
    x = [list(y) for y in x]
    for i in range(len(x)):
        for a in x[i]:
            distance_to_bin[a] = i
    
    return distance_to_bin

def experiment_one(model, dataset, left_sens_attribute, right_sens_attribute, 
                    epochs=10, single_fairness=True, 
                    threshold=0.2, test_file = "test.csv"):
    if dataset == "itunes-amazon" or dataset == "shoes":
        test_file = "test_others.csv"
    workloads = run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, 
                                epochs=epochs, single_fairness=single_fairness, test_file=test_file)
    
    fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/" + dataset, 
                        full_workload_test=test_file, threshold=threshold, single_fairness=single_fairness)

    binary_fairness = []
    measures = ["accuracy_parity", "statistical_parity", \
                "true_positive_rate_parity", "false_positive_rate_parity", \
                "false_negative_rate_parity", "true_negative_rate_parity", \
                "negative_predictive_value_parity", "false_discovery_rate_parity", \
                "false_omission_rate_parity"]
    measures_acronymes = [make_acronym(x, "_") for x in measures]
    
    aggregate = "distribution"
    for measure in measures:
        is_fair = fairEM.is_fair(measure, aggregate)
        binary_fairness.append(is_fair)
    attribute_names = []
    for k_comb in workloads[0].k_combs_to_attr_names:
        attribute_names.append(workloads[0].k_combs_to_attr_names[k_comb])
    
    if dataset == "dblp-acm":
        attribute_names = [make_acronym(x, " ") for x in attribute_names]
        figsize = (8,8)
        x_font = 12
        y_font = 12
    elif dataset == "itunes-amazon":
        figsize = (10, 7)
        x_font = 9
        y_font = 12        
    elif dataset == "shoes":
        figsize = (8,6)
        x_font = 12
        y_font = 12

    title = "Exp1: " + dataset + " " + model + " with " + str(threshold) + " threshold \nBinary Fairness Values For 1-subgroups and Single Fairness and 1 workload"
    if dataset == "itunes-amazon" or dataset == "shoes":
        title += " with others column"
    plot_results_in_2d_heatmap(dataset, binary_fairness, attribute_names, 
                                measures_acronymes, title, 
                                figsize, x_font, y_font)

def experiment_two(model, dataset, left_sens_attribute, right_sens_attribute, epochs=10, single_fairness=False, threshold=0.2):
    test_file = "test.csv" if dataset == "dblp-acm" else "test_others.csv"
    workloads = run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, 
                                epochs=epochs, single_fairness=single_fairness, 
                                test_file=test_file)
    fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/" + dataset, 
                        full_workload_test="test.csv", threshold=threshold, single_fairness=single_fairness)

    binary_fairness = []
    measures = ["accuracy_parity", "statistical_parity", \
                "true_positive_rate_parity", "false_positive_rate_parity", \
                "false_negative_rate_parity", "true_negative_rate_parity"]
    measures_acronymes = [make_acronym(x, "_") for x in measures]
    
    aggregate = "distribution"
    for measure in measures:
        is_fair = fairEM.is_fair(measure, aggregate)
        binary_fairness.append(is_fair)
    attribute_names = []
    for k_comb in workloads[0].k_combs_to_attr_names:
        curr_attr_name = workloads[0].k_combs_to_attr_names[k_comb]
        curr_attr_name = curr_attr_name.replace("|", " | ") 
        attribute_names.append(curr_attr_name)

    if dataset == "dblp-acm":
        attribute_names = [make_acronym(x, " ") for x in attribute_names]
        figsize = (8,8)
        x_font = 12
        y_font = 12
    elif dataset == "itunes-amazon":
        figsize = (10, 7)
        x_font = 9
        y_font = 12        
    elif dataset == "shoes":
        figsize = (8,6)
        x_font = 12
        y_font = 12

    title = "Exp2: " + dataset + " " + model + " with " + str(threshold) + " threshold\nBinary Fairness Values For 1-subgroups and Pairwise Fairness and 1 workload"
    if dataset == "itunes-amazon" or dataset == "shoes":
        title += " with others column"
    plot_results_in_2d_heatmap(dataset, binary_fairness, attribute_names, 
                                measures_acronymes, title, 
                                figsize, x_font, y_font)

def experiment_three(dataset, model, single_fairness=True, epochs=15, others=True):
    workloads = run_multiple_workloads(dataset, model, epochs=epochs, single_fairness=single_fairness, others=others)
    fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/" + dataset + "/", 
                        full_workload_test="test.csv", single_fairness=single_fairness)

    binary_fairness = []
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

    measures_acronymes = [make_acronym(x,"_") for x in measures]
    
    for measure in measures:
        is_fair = fairEM.is_fair(measure, aggregate)
        binary_fairness.append(is_fair)
    attribute_names = []
    for k_comb in workloads[0].k_combs_to_attr_names:
        attribute_names.append(workloads[0].k_combs_to_attr_names[k_comb])

    subgroup_to_isfair = {}
    
    for measure in measures:
        is_fair = fairEM.is_fair(measure, aggregate) # returns a dictionary
        for subgroup in is_fair:
            if subgroup not in subgroup_to_isfair:
                subgroup_to_isfair[subgroup] = []
            subgroup_to_isfair[subgroup].append(is_fair[subgroup])
        

    subgroups = []
    values = []
    for subgroup in subgroup_to_isfair:
        subgroups.append(subgroup)
        values.append(subgroup_to_isfair[subgroup])

    if dataset == "dblp-acm":
        subgroups = [make_acronym(x, " ") for x in subgroups]
        figsize = (8,8)
        x_font = 12
        y_font = 12
    elif dataset == "itunes-amazon":
        figsize = (10, 7)
        x_font = 9
        y_font = 12        
    elif dataset == "shoes":
        figsize = (8,6)
        x_font = 12
        y_font = 12

    title = "Exp3: " + dataset + " " + model +" \nBinary Fairness Values For 1-subgroups and Single Fairness and 40 workloads" if single_fairness else \
            "Exp3: " + dataset + model + "\nBinary Fairness Values For 1-subgroups and Pairwise Fairness and 40 workloads"
    plot_results_in_2d_heatmap(dataset, np.transpose(values), subgroups, 
                                measures_acronymes, title,
                                figsize, x_font, y_font)
       
def experiment_four(dataset, model, left_sens_attribute, right_sens_attribute, epochs, k_combinations, test_file="test.csv"):
    if dataset == "itunes-amazon" or dataset == "shoes":
        test_file = "test_others.csv"
    
    measures = ["accuracy_parity", "statistical_parity", \
                "true_positive_rate_parity", "false_positive_rate_parity", \
                "false_negative_rate_parity", "true_negative_rate_parity", \
                "negative_predictive_value_parity", "false_discovery_rate_parity", \
                "false_omission_rate_parity"]
    aggregates = ["max", "min", "max_minus_min", "average"]
    

    f, axarr = plt.subplots(3,3,figsize=(6,5))
    
    k_combs_ylabel = [str(x)+"-comb" for x in range(1, k_combinations+1)]
    k_combs = [x for x in range(1, k_combinations+1)]
    k_combs_fairness_measure = []
    for i in range(len(k_combs)):
        k_comb = k_combs[i]
        k_comb_label = k_combs_ylabel[i]
        k_comb_VS_fairness = []
        workloads = run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, 
                                    epochs=epochs, single_fairness=True, k_combinations=k_comb,
                                    test_file=test_file)
        fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/" + dataset, 
                    full_workload_test=test_file, single_fairness=False)

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
        
        axarr[x][y].set_title(measures[i], fontdict={'fontsize': 8})
        axarr[x][y].imshow(k_comb_VS_fairness, cmap="RdYlGn", vmin=0, vmax=1)
        axarr[x][y].set_xticks(np.arange(len(aggregates)))
        axarr[x][y].set_xticklabels(labels=aggregates, fontdict={'fontsize': 8})
        axarr[x][y].set_yticks(np.arange(len(k_combs_ylabel)))
        axarr[x][y].set_yticklabels(labels=k_combs_ylabel, fontdict={'fontsize': 8})
        
        plt.setp(axarr[x][y].get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

    f.tight_layout()
    plt.savefig("../experiments/" + dataset + "/Exp4: " + model + " General Model Fairness.png")
    plt.close()

def experiment_five(model, epochs, one_workload=True, single_fairness=True, measure = "accuracy"):
    dataset = "itunes-amazon"
    left_sens_attribute = "left_Genre"
    right_sens_attribute = "right_Genre"
    
    if one_workload:
        workloads = run_one_workload(model, dataset, left_sens_attribute, right_sens_attribute, 
                                epochs=epochs, single_fairness=single_fairness, 
                                k_combinations=1, delimiter=',', test_file = "test_others.csv")
    else:
        workloads = run_multiple_workloads(num_of_workloads=40, epochs=epochs, single_fairness=single_fairness)
    
    fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/" + dataset, 
                        full_workload_test="test.csv", single_fairness=single_fairness)

    is_fair_distribution = fairEM.is_fair(measure + "_parity", "distribution")

    attribute_names = []
    for k_comb in workloads[0].k_combs_to_attr_names:
        attribute_names.append(workloads[0].k_combs_to_attr_names[k_comb])

    multiple_bins_to_conf_matrix = []
    unfair_subgroups = []

    for i in range(len(is_fair_distribution)):
        subgroup = attribute_names[i]
        val = is_fair_distribution[i]
        if not val:
            bins_to_conf_matrix = fairEM.distance_analysis_unfair(subgroup)
            bins_to_conf_matrix = {k: v for k, v in sorted(bins_to_conf_matrix.items())}
            multiple_bins_to_conf_matrix.append(bins_to_conf_matrix)
            unfair_subgroups.append(subgroup)

    fairEM.distance_analysis_all()
    labels = ["general"]
    conf_matrices = []
    conf_matrices.append(fairEM.distances_all)
    distances_list = list(fairEM.distances_all.keys())
    distances_list.sort()
    distance_to_bin = create_distance_to_bin(distances_list)

    for i in range(len(unfair_subgroups)):
        labels.append(unfair_subgroups[i])
        conf_matrices.append(multiple_bins_to_conf_matrix[i])

    for i in range(len(conf_matrices)):
        plot_distances_all(conf_matrices[i], labels[i], model, distance_to_bin, measure)

    plt.legend(labels, bbox_to_anchor=[0.0, 1.0], loc='upper left', prop={'size': 6})
    plt.xticks(range(0,4))
    title = "Exp5: " + model + " Distance Explainability for " + measure
    
    plt.savefig("../experiments/itunes-amazon/" + title + ".png")

    plt.close()      


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

def full_experiment_three():
    experiment_three("shoes", "ditto")
    experiment_three("shoes", "deepmatcher")
    experiment_three("itunes-amazon", "ditto")
    experiment_three("itunes-amazon", "deepmatcher")
    experiment_three("dblp-acm", "ditto", others=False)
    experiment_three("dblp-acm", "deepmatcher", others=False)

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

def full_experiment_five():
    experiment_five(model="deepmatcher", epochs=15, one_workload=True, single_fairness=True, measure = "accuracy")
    experiment_five(model="ditto", epochs=15, one_workload=True, single_fairness=True, measure = "accuracy")
    experiment_five(model="deepmatcher", epochs=15, one_workload=True, single_fairness=True, measure = "true_positive_rate")
    experiment_five(model="ditto", epochs=15, one_workload=True, single_fairness=True, measure = "true_positive_rate")
    

    
    
def main():
   
    full_experiment_one()
    full_experiment_two()
    full_experiment_three()
    full_experiment_four()
    full_experiment_five()

main()