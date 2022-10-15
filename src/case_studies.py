# from preprocessing import run_deepmatcher, jsonl_to_predictions, deepmatcher_output_to_predictions
from pprint import pprint
from run import *
import workloads as wl
import pandas as pd
import FairEM as fem
import numpy as np
import matplotlib.pyplot as plt


def find_small_freqs_vals(dataset, left_sens_attribute, right_sens_attribute,test_file):
    # model is irrelevant
    workloads = run_one_workload("DeepMatcher", dataset, left_sens_attribute, right_sens_attribute, test_file,
                                  single_fairness=True)
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


def case_study_itunes_amazon(dataset="iTunes-Amazon", model="DeepMatcher",
                             left_sens_attribute="left_Genre",
                             right_sens_attribute="right_Genre", delimiter=",",
                             single_fairness=True, k_combinations=1,
                             threshold=0.2):
    small_freq_vals = find_small_freqs_vals(dataset, left_sens_attribute, right_sens_attribute)
    pprint(small_freq_vals)
    dir = "../data/FairEM/DeepMatcher/" + dataset + "/test.csv"

    df = pd.read_csv(dir, index_col="id")
    for idx, row in df.iterrows():
        left = row[left_sens_attribute]
        right = row[right_sens_attribute]
        new_left = ""
        new_right = ""
        for item in left.split(delimiter):
            item = item.strip()
            if item not in small_freq_vals:
                new_left += item + delimiter
            else:
                new_left += "other" + delimiter
        new_left = new_left[:-1]
        for item in right.split(delimiter):
            item = item.strip()
            if item not in small_freq_vals:
                new_right += item + delimiter
            else:
                new_right += "other" + delimiter
        new_right = new_right[:-1]

        print("new_left = ", new_left)
        print("new_right = ", new_right)
        df.at[idx, left_sens_attribute] = new_left
        df.at[idx, right_sens_attribute] = new_right

    df.to_csv("../data/FairEM/DeepMatcher/" + dataset + "/test_others.csv")


def case_study_shoes(dataset="Shoes",
                     left_sens_attribute="left_locale",
                     right_sens_attribute="right_locale", delimiter=","):
    small_freq_vals = find_small_freqs_vals(dataset, left_sens_attribute, right_sens_attribute,test_file="test_others.csv")

    dir = "../data/FairEM/DeepMatcher/" + dataset + "/test_others.csv"
    df = pd.read_csv(dir, index_col="id")

    for idx, row in df.iterrows():
        left = row[left_sens_attribute]
        right = row[right_sens_attribute]
        new_left = ""
        new_right = ""
        for item in left.split(delimiter):
            item = item.strip()
            if item not in small_freq_vals:
                new_left += item + delimiter
            else:
                new_left += "other" + delimiter
        new_left = new_left[:-1]
        for item in right.split(delimiter):
            item = item.strip()
            if item not in small_freq_vals:
                new_right += item + delimiter
            else:
                new_right += "other" + delimiter
        new_right = new_right[:-1]

        print("new_left = ", new_left)
        print("new_right = ", new_right)
        df.at[idx, left_sens_attribute] = new_left
        df.at[idx, right_sens_attribute] = new_right

    df.to_csv("../data/FairEM/DeepMatcher/" + dataset + "/test_others_.csv")


# case_study_itunes_amazon()
case_study_shoes()
case_study_shoes(dataset="Cameras")

