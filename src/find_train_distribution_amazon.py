from preprocessing import run_deepmatcher, jsonl_to_predictions, deepmatcher_output_to_predictions
from pprint import pprint
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
    workloads[0].df = pd.read_csv("../data/" + dataset + "/train.csv", index_col = "id")
    workload = workloads[0]
    
    workload.sens_attr_vals = workload.find_all_sens_attr()
    workload.sens_att_to_index = workload.create_sens_att_to_index()
    workload.name_to_encode = {}
    workload.encoding = workload.encode()


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
    
    sorted_attr_value_frequency = sorted(attr_value_frequency.items(), key=lambda x: x[1], reverse=True)
    pprint(sorted_attr_value_frequency)

find_small_freqs_vals("itunes-amazon", "left_Genre", "right_Genre")