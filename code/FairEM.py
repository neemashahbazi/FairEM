import itertools

import numpy
import numpy as np
import scipy.stats as stats
import pandas as pd
import sys

class FairER:
    # the input is a list of objects of class Workload
    # alpha is used for One-Way Anova Test 
    def __init__(self, workloads, alpha, threshold, single_fairness=True):
        self.workloads = workloads
        self.alpha = alpha
        self.threshold = threshold
        self.single_fairness = single_fairness

        # subgroups is a list containing encodings
        def is_fair(self, subgroups, measure):
            if len(self.workloads) == 1:
                accuracy = [abs(self.calculate_measure(subgroup, self.workloads[0], measure)) for subgroup in subgroups]
                self.fairness = accuracy
                return [val < self.threshold for val in accuracy]
            else:
                arr = []
                for subgroup in subgroups:
                    subgroup_accuracy = [self.calculate_measure(subgroup, workload, measure) for workload in self.workload]
                    p_value = ztest(subgroup_accuracy, value=self.threshold)[1]
                    # if p_value <= alpha then the null hypothesis is not rejected, and the
                    # algorithm is fair
                    arr.append(p_value <= self.alpha)
                return arr
