import itertools

import numpy
import numpy as np
import scipy.stats as stats
import pandas as pd
import sys
from statsmodels.stats.weightstats import ztest
from utils import calculate_distance
from pprint import pprint

class FairEM:
    # the input is a list of objects of class Workload
    # alpha is used for the Z-Test 
    def __init__(self, workloads, alpha, threshold, single_fairness=True):
        self.workloads = workloads
        self.alpha = alpha
        self.threshold = threshold
        self.single_fairness = single_fairness

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

