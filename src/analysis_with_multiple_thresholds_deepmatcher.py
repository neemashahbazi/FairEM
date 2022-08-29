from preprocessing import deepmatcher_output_to_predictions
from run import plot_results_in_2d_heatmap, make_acronym
import workloads as wl
import pandas as pd
import FairEM as fem
import matplotlib.pyplot as plt
import numpy as np


def run_dm():
    dataset = "itunes-amazon"
    model = "deepmatcher"
    for dm_threshold in [0.2, 0.4, 0.6, 0.8, 0.9]:
        path = "../data/" + dataset + "/"
        filename = "deepmatcher_out_15_" + str(dm_threshold) + ".txt"

        predictions = deepmatcher_output_to_predictions(path, filename)

        test_file = "test_others.csv"
        left_sens_attribute = "left_Genre"
        right_sens_attribute = "right_Genre"
        workload = wl.Workload(pd.read_csv("../data/" + dataset + "/" + test_file), 
                                left_sens_attribute, right_sens_attribute,
                                predictions, label_column = "label", 
                                multiple_sens_attr = True, delimiter = ",", 
                                single_fairness = True, k_combinations=1)
        workloads = [workload]
        fairEM = fem.FairEM(model, workloads, alpha=0.05, directory="../data/" + dataset, 
                            full_workload_test=test_file, threshold=0.2, single_fairness=True)

        binary_fairness = []
        actual_fairness = []
        measures = ["accuracy_parity", "statistical_parity", \
                    "true_positive_rate_parity", "false_positive_rate_parity", \
                    "false_negative_rate_parity", "true_negative_rate_parity", \
                    "negative_predictive_value_parity", "false_discovery_rate_parity", \
                    "false_omission_rate_parity"]
        measures_acronymes = [make_acronym(x, "_") for x in measures]
        
        aggregate = "distribution"
        for measure in measures:
            is_fair = fairEM.is_fair(measure, aggregate)
            is_fair_real = fairEM.is_fair(measure, aggregate, real_distr = True)
            binary_fairness.append(is_fair)
            actual_fairness.append(is_fair_real)

        

        attribute_names = []
        for k_comb in workloads[0].k_combs_to_attr_names:
            attribute_names.append(workloads[0].k_combs_to_attr_names[k_comb])
        

        if dataset == "itunes-amazon":
            figsize = (10, 7)
            x_font = 9
            y_font = 12        
        
        title = "Exp1: Deepmatcher threshold " + str(dm_threshold) + " with others column"
        plot_results_in_2d_heatmap(dataset, binary_fairness, attribute_names, 
                                    measures_acronymes, title, 
                                    figsize, x_font, y_font)
            
# run_dm()

thresholds = [0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
labels = ["AP","SP","TPRP","FPRP","FNRP","TNRP","NPVP","FDRP","FORP"]
unfairnessesAP =   [0, 1, 2, 1, 0, 0]
unfairnessesSP =   [0, 0, 0, 0, 0, 0]
unfairnessesTPRP = [7, 7, 4, 9, 7, 0]
unfairnessesFPRP = [2, 2, 2, 2, 0, 0]
unfairnessesFNRP = [7, 7, 4, 9, 7, 0]
unfairnessesTNRP = [2, 2, 2, 2, 0, 0]
unfairnessesNPVP = [2, 1, 3, 0, 0, 0]
unfairnessesFDRP = [1, 0, 0, 0, 0, 0]
unfairnessesFORP = [1, 1, 3, 0, 0, 0]

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(unfairnessesAP, linestyle=":")
ax.plot(unfairnessesSP, linestyle="-")
ax.plot(unfairnessesTPRP, linestyle="-.")
ax.plot(unfairnessesFPRP, linestyle=":")
ax.plot(unfairnessesFNRP, linestyle="-")
ax.plot(unfairnessesTNRP, linestyle="-.")
ax.plot(unfairnessesNPVP, linestyle="-")
ax.plot(unfairnessesFDRP, linestyle=":")
ax.plot(unfairnessesFORP, linestyle="-.")
ax.set_xticks(np.arange(len(thresholds)))
ax.set_xticklabels(labels=thresholds)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.set_xlabel("Deepmatcher threshold", fontdict={'fontsize': 16})
ax.set_ylabel("Number of Unfairnesses", fontdict={'fontsize': 16})
ax.legend(labels, bbox_to_anchor=[0.0, 1.0], loc='upper left')


plt.savefig("../experiments/itunes-amazon/Deepmatcher different thresholds analysis.png")
plt.close()