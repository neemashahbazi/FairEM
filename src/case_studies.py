from preprocessing import run_deepmatcher, jsonl_to_predictions, deepmatcher_output_to_predictions
from pprint import pprint
from create_multiple_workloads import create_workloads_from_file
from run import *
import workloads as wl
import pandas as pd
import FairEM as fem
import numpy as np
import matplotlib.pyplot as plt

def find_small_freqs_vals(dataset, left_sens_attribute, right_sens_attribute):
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
    
    sorted_attr_value_frequency = sorted(attr_value_frequency.items(), key=lambda x: x[1])
    pprint(sorted_attr_value_frequency)
    small_freq_vals = set()
    
    for subgroup_freq in sorted_attr_value_frequency:
        if subgroup_freq[1] < 10:
            small_freq_vals.add(subgroup_freq[0])
    
    return small_freq_vals



def case_study_itunes_amazon(dataset = "itunes-amazon", model="deepmatcher",
                            left_sens_attribute = "left_Genre",
                            right_sens_attribute = "right_Genre", delimiter=",",
                            single_fairness = True, k_combinations = 1,
                            threshold = 0.2):
    small_freq_vals = find_small_freqs_vals(dataset, left_sens_attribute, right_sens_attribute)
    pprint(small_freq_vals)

    # df = pd.read_csv("../data/itunes-amazon/test.csv", index_col = "id")
    # for idx, row in df.iterrows():
    #     left = row[left_sens_attribute]
    #     right = row[right_sens_attribute]
    #     new_left = ""
    #     new_right = ""
    #     for item in left.split(delimiter):
    #         item = item.strip()
    #         if item not in small_freq_vals:
    #             new_left += item + delimiter
    #         else:
    #             new_left += "other" + delimiter
    #     new_left = new_left[:-1]
    #     for item in right.split(delimiter):
    #         item = item.strip()
    #         if item not in small_freq_vals:
    #             new_right += item + delimiter
    #         else:
    #             new_right += "other" + delimiter
    #     new_right = new_right[:-1]
        
    #     print("new_left = ", new_left)
    #     print("new_right = ", new_right)
    #     df.at[idx,left_sens_attribute] = new_left
    #     df.at[idx,right_sens_attribute] = new_right
    
    # df.to_csv("../data/itunes-amazon/test_others.csv")

    # for model in ["deepmatcher", "ditto"]:
    #     if model == "deepmatcher":
    #         predictions = deepmatcher_output_to_predictions("../data/" + dataset + "/", "deepmatcher_out_15.txt")
    #     elif model == "ditto":
    #         predictions = jsonl_to_predictions("../data/" + dataset + "/", "ditto_out_test.jsonl")
    #     workload = wl.Workload(pd.read_csv("../data/" + dataset + "/test_others.csv"), 
    #                             left_sens_attribute, right_sens_attribute,
    #                             predictions, label_column = "label", 
    #                             multiple_sens_attr = True, delimiter = delimiter, 
    #                             single_fairness = single_fairness, k_combinations=1)
    #     workloads = [workload]
    #     fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/" + dataset, 
    #                         full_workload_test="test_others.csv", threshold=threshold, single_fairness=single_fairness)

    #     binary_fairness = []
    #     measures = ["accuracy_parity", "statistical_parity", \
    #                 "true_positive_rate_parity", "false_positive_rate_parity", \
    #                 "false_negative_rate_parity", "true_negative_rate_parity", \
    #                 "negative_predictive_value_parity", "false_discovery_rate_parity", \
    #                 "false_omission_rate_parity"]
    #     aggregate = "distribution"
    #     for measure in measures:
    #         is_fair = fairEM.is_fair(measure, aggregate)
    #         binary_fairness.append(is_fair)
    #     attribute_names = []
    #     for k_comb in workloads[0].k_combs_to_attr_names:
    #         attribute_names.append(workloads[0].k_combs_to_attr_names[k_comb])

    #     title = "OTHERS_Exp1: " + dataset + " " + model + " with " + str(threshold) + " threshold \nBinary Fairness Values For 1-subgroups and Single Fairness and 1 workload"
    #     plot_results_in_2d_heatmap(dataset, binary_fairness, attribute_names, 
    #                                 measures, title)
   
def case_study_shoes(dataset = "shoes", model="deepmatcher",
                            left_sens_attribute = "left_locale",
                            right_sens_attribute = "right_locale", delimiter=",",
                            single_fairness = True, k_combinations = 1,
                            threshold = 0.2):
    # small_freq_vals = find_small_freqs_vals(dataset, left_sens_attribute, right_sens_attribute)
    # # pprint(small_freq_vals)

    # df = pd.read_csv("../data/shoes/test.csv", index_col = "id")
    # for idx, row in df.iterrows():
    #     left = row[left_sens_attribute]
    #     right = row[right_sens_attribute]
    #     new_left = ""
    #     new_right = ""
    #     for item in left.split(delimiter):
    #         item = item.strip()
    #         if item not in small_freq_vals:
    #             new_left += item + delimiter
    #         else:
    #             new_left += "other" + delimiter
    #     new_left = new_left[:-1]
    #     for item in right.split(delimiter):
    #         item = item.strip()
    #         if item not in small_freq_vals:
    #             new_right += item + delimiter
    #         else:
    #             new_right += "other" + delimiter
    #     new_right = new_right[:-1]
        
    #     print("new_left = ", new_left)
    #     print("new_right = ", new_right)
    #     df.at[idx,left_sens_attribute] = new_left
    #     df.at[idx,right_sens_attribute] = new_right
    
    # df.to_csv("../data/shoes/test_others.csv")

    for model in ["deepmatcher", "ditto"]:
        if model == "deepmatcher":
            predictions = deepmatcher_output_to_predictions("../data/" + dataset + "/", "deepmatcher_out_15.txt")
        elif model == "ditto":
            predictions = jsonl_to_predictions("../data/" + dataset + "/", "ditto_out_test.jsonl")
        workload = wl.Workload(pd.read_csv("../data/" + dataset + "/test_others.csv"), 
                                left_sens_attribute, right_sens_attribute,
                                predictions, label_column = "label", 
                                multiple_sens_attr = True, delimiter = delimiter, 
                                single_fairness = single_fairness, k_combinations=1)
        workloads = [workload]
        fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/" + dataset, 
                            full_workload_test="test_others.csv", threshold=threshold, single_fairness=single_fairness)

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

        title = "OTHERS_Exp1: " + dataset + " " + model + " with " + str(threshold) + " threshold \nBinary Fairness Values For 1-subgroups and Single Fairness and 1 workload"
        plot_results_in_2d_heatmap(dataset, binary_fairness, attribute_names, 
                                    measures, title)

case_study_itunes_amazon()   
# case_study_shoes()