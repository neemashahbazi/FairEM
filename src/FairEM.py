import itertools
import numpy
import numpy as np
import scipy.stats as stats
import pandas as pd
import workloads as wl 
from statsmodels.stats.weightstats import ztest
from utils import calculate_distance, f1_score
from preprocessing import run_deepmatcher
from pprint import pprint

class FairEM:
    # the input is a list of objects of class Workload
    # alpha is used for the Z-Test 
    def __init__(self, workloads, alpha, threshold, directory, full_workload_test, single_fairness=True):
        self.workloads = workloads
        self.alpha = alpha
        self.threshold = threshold
        self.single_fairness = single_fairness

        self.full_workload_distance = self.distance_analysis_prepro(directory, full_workload_test)

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

    def is_fair(self, measure, aggregate):
        if len(self.workloads) == 1:
            workload_fairness = self.workloads[0].fairness(workloads[0].k_combs, measure, aggregate)
            if aggregate is not "distribution":
                return workload_fairness > self.threshold
            else:
                return [subgroup_fairness > self.threshold for subgroup_fairness in workload_fairness]
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

    def distance_analysis_prepro(self, directory, full_workload_test):
        predictions = run_deepmatcher(directory, epochs = 2)
        workload = wl.Workload(pd.read_csv(directory + "/" + full_workload_test), self.workloads[0].sens_att_left, 
                                self.workloads[0].sens_att_left, predictions, 
                                label_column = self.workloads[0].label_column,
                                multiple_sens_attr = self.workloads[0].multiple_sens_attr,
                                delimiter = self.workloads[0].delimiter, 
                                single_fairness = self.workloads[0].single_fairness,
                                k_combinations = 2)
        return workload


    # unfair_subgroup is given as a string
    def distance_analysis(self, unfair_subgroup):

        workload = self.full_workload_distance
        print("UNFAIR_SUBGROUP = ", unfair_subgroup)
        
        fairness_per_distance = {}

        for idx, row in workload.df.iterrows():
            distance_left = calculate_distance(unfair_subgroup, row[workload.sens_att_left], ",")
            distance_right = calculate_distance(unfair_subgroup, row[workload.sens_att_right], ",")
            distance = min(distance_left, distance_right)

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

        pprint(fairness_per_distance)
        return fairness_per_distance