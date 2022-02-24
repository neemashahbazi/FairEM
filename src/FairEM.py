import itertools
import numpy
import numpy as np
import scipy.stats as stats
import pandas as pd
import workloads as wl 
from statsmodels.stats.weightstats import ztest
from utils import calculate_distance
from preprocessing import run_deepmatcher, jsonl_to_predictions, deepmatcher_output_to_predictions
from pprint import pprint

class FairEM:
    # the input is a list of objects of class Workload
    # alpha is used for the Z-Test 
    def __init__(self, model, workloads, alpha, directory, full_workload_test, threshold=0.2, single_fairness=True, 
                ditto_out_test_full="ditto_out_test.jsonl", deepmatcher_out_test_full="deepmatcher_out_15.txt"):
        self.model = model
        self.workloads = workloads
        self.alpha = alpha
        self.threshold = threshold
        self.single_fairness = single_fairness
        self.ditto_out_test_full = ditto_out_test_full
        self.deepmatcher_out_test_full = deepmatcher_out_test_full
        self.full_workload_distance = self.distance_analysis_prepro(directory, full_workload_test, epochs=2)

        self.TP = 0
        self.FP = 1
        self.TN = 2
        self.FN = 3
        

    # creates a two dimensional matrix, subgroups x workload fairness value
    # used only for distribution
    def separate_distributions_from_workloads(self, subgroups, workloads_fairness):
        num_of_subgroups = len(subgroups)
        subgroup_precisions = []
        for i in range(num_of_subgroups):
            subgroup_precisions.append([])
        for i in range(num_of_subgroups):
            for workload_distr in workloads_fairness:
                subgroup_precisions[i].append(workload_distr[i])
        return subgroup_precisions

    # true would mean something is good, i.e. is fair
    # so for accuracy if x0 - avg(x) > -threshold, this is good
    # if we want a measure to be as low as possible, 
    # then x0 - avg(x) < threshold
    def is_fair_measure_specific(self, measure, workload_fairness):
        if measure == "accuracy_parity" or \
            measure == "statistical_parity" or \
            measure == "true_positive_rate_parity" or \
            measure == "true_negative_rate_parity" or \
            measure == "positive_predictive_value_parity" or \
            measure == "negative_predictive_value_parity":
            return workload_fairness >= -self.threshold
        if measure == "false_positive_rate_parity" or \
            measure == "false_negative_rate_parity" or \
            measure == "false_discovery_rate_parity" or \
            measure == "false_omission_rate_parity":
            return workload_fairness <= self.threshold

    
    def is_fair(self, measure, aggregate, real_distr = False):
        if len(self.workloads) == 1:
            workload_fairness = self.workloads[0].fairness(self.workloads[0].k_combs, measure, aggregate)
            if aggregate is not "distribution":
                return self.is_fair_measure_specific(measure, workload_fairness)
            else:
                if real_distr:
                    return workload_fairness
                else:
                    return [self.is_fair_measure_specific(measure, subgroup_fairness) \
                            for subgroup_fairness in workload_fairness]
        else:
            workloads_fairness = []
            for workload in self.workloads:
                workloads_fairness.append(workload.fairness(workload.k_combs, measure, aggregate))

            # general idea of how the entity matching is performed
            if aggregate is not "distribution":
                p_value = ztest(workloads_fairness, value=self.threshold)[1]
                return p_value <= self.alpha
            # specific for each measure
            else:
                subgroup_to_list_of_fairneses = {}
                for i in range(len(self.workloads)):
                    workload = self.workloads[i]
                    fairnesses = workloads_fairness[i]

                    k_combs_list = [x for x in workload.k_combs_to_attr_names]

                    for j in range(len(fairnesses)):
                        subgroup_index = k_combs_list[j]
                        subgroup_name = workload.k_combs_to_attr_names[subgroup_index]
                        if subgroup_name not in subgroup_to_list_of_fairneses:
                            subgroup_to_list_of_fairneses[subgroup_name] = []
                        subgroup_to_list_of_fairneses[subgroup_name].append(fairnesses[j])
                    
                subroups_is_fair = {}
                for subgroup in subgroup_to_list_of_fairneses:
                    if len(subgroup_to_list_of_fairneses[subgroup]) >= 30: #limit for a valid z-test
                        p_value = ztest(subgroup_to_list_of_fairneses[subgroup], value = self.threshold)[1]
                        subroups_is_fair[subgroup] = (p_value <= self.alpha)
                
                return subroups_is_fair

    def distance_analysis_prepro(self, directory, full_workload_test, epochs):
        if self.model == "deepmatcher":
            predictions = deepmatcher_output_to_predictions(directory, self.deepmatcher_out_test_full)
        elif self.model == "ditto":
            predictions = jsonl_to_predictions(directory, self.ditto_out_test_full)
            
        workload = wl.Workload(pd.read_csv(directory + "/" + full_workload_test), self.workloads[0].sens_att_left, 
                                self.workloads[0].sens_att_left, predictions, 
                                label_column = self.workloads[0].label_column,
                                multiple_sens_attr = self.workloads[0].multiple_sens_attr,
                                delimiter = self.workloads[0].delimiter, 
                                single_fairness = self.workloads[0].single_fairness,
                                k_combinations = 2)
        return workload


    # unfair_subgroup is given as a string
    def distance_analysis(self, unfair_subgroup, number_of_bins=5):

        workload = self.full_workload_distance
        print("UNFAIR_SUBGROUP = ", unfair_subgroup)
        
        fairness_per_distance = {}

        for idx, row in workload.df.iterrows():
            distance = None
            if workload.single_fairness:
                distance_left = calculate_distance(unfair_subgroup, row[workload.sens_att_left], ",")
                distance_right = calculate_distance(unfair_subgroup, row[workload.sens_att_right], ",")
                distance = min(distance_left, distance_right)

            else:
                unfair_subgroup_separated = unfair_subgroup.split("|")
                distance_left = calculate_distance(unfair_subgroup_separated[0], row[workload.sens_att_left], ",")
                distance_right = calculate_distance(unfair_subgroup_separated[1], row[workload.sens_att_right], ",")
                distance = distance_left + distance_right
                
                distance_left = calculate_distance(unfair_subgroup_separated[1], row[workload.sens_att_left], ",")
                distance_right = calculate_distance(unfair_subgroup_separated[0], row[workload.sens_att_right], ",")

                distance = min(distance, distance_left + distance_right)
                
            if distance not in fairness_per_distance:
                fairness_per_distance[distance] = [0, 0, 0, 0]

            if workload.prediction[idx]:
                if row[workload.label_column]: #ground truth
                    fairness_per_distance[distance][self.TP] += 1
                else:
                    fairness_per_distance[distance][self.FP] += 1
            else:
                if row[workload.label_column]:
                    fairness_per_distance[distance][self.FN] += 1
                else:
                    fairness_per_distance[distance][self.TN] += 1

        bins_to_conf_matrix = self.bin_fairness_per_distance(fairness_per_distance)
        pprint(bins_to_conf_matrix)
        return bins_to_conf_matrix

    def bin_fairness_per_distance(self, fairness_per_distance, number_of_bins=5):
        freq = []
        for distance in fairness_per_distance:
            total_with_distance = sum(fairness_per_distance[distance])
            for i in range(total_with_distance):
                freq.append(distance)
        bins = pd.qcut(freq, q = number_of_bins, labels=False, duplicates = 'drop')

        distance_to_bin = {}
        for i in range(len(freq)):
            distance_to_bin[freq[i]] = bins[i]
        
        bins_to_conf_matrix = {}

        for distance in distance_to_bin:
            binn = distance_to_bin[distance]
            if binn not in bins_to_conf_matrix:
                bins_to_conf_matrix[binn] = [0, 0, 0, 0]
            bins_to_conf_matrix[binn] = [x + y for x, y in zip(bins_to_conf_matrix[binn], fairness_per_distance[distance])]

        return bins_to_conf_matrix