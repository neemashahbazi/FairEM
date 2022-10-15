from pprint import pprint

import pandas as pd


def find_all_sens_attr(df, sens_att_left, sens_att_right, multiple_sens_attr, delimiter):
    sens_att = set()
    for it_index, row in df.iterrows():
        left = row[sens_att_left]
        right = row[sens_att_right]
        if multiple_sens_attr:
            for item in left.split(delimiter):
                sens_att.add(item.strip())
            for item in right.split(delimiter):
                sens_att.add(item.strip())
        else:
                sens_att.add(left.strip())
                sens_att.add(right.strip())

    sens_attr_vals = list(sens_att)
    sens_attr_vals.sort()
    return sens_attr_vals


def find_small_freqs_vals(delimiter, multiple_sens_attr, sens_att_left, sens_att_right, df):
    sens_attr_vals = find_all_sens_attr(df, sens_att_left, sens_att_right, multiple_sens_attr, delimiter)
    attr_value_frequency = {}
    for attr_val in sens_attr_vals:
        attr_value_frequency[attr_val] = 0

    for idx, row in df.iterrows():
        left = row[sens_att_left]
        right = row[sens_att_right]
        if multiple_sens_attr:
            for item in left.split(delimiter):
                sens_att = item.strip()
                attr_value_frequency[sens_att] += 1
            for item in right.split(delimiter):
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


find_small_freqs_vals(delimiter=',',
                      multiple_sens_attr=True,
                      sens_att_left="left_Genre",
                      sens_att_right="right_Genre",
                      df=pd.read_csv("../data/FairEM/DeepMatcher/iTunes-Amazon/train.csv", index_col="id"))

print("========================================")

find_small_freqs_vals(delimiter=',',
                      multiple_sens_attr=False,
                      sens_att_left="left_venue",
                      sens_att_right="right_venue",
                      df=pd.read_csv("../data/FairEM/DeepMatcher/DBLP-ACM/train_others.csv", index_col="id"))

print("========================================")

find_small_freqs_vals(delimiter=',',
                      multiple_sens_attr=False,
                      sens_att_left="left_ENTRYTYPE",
                      sens_att_right="right_ENTRYTYPE",
                      df=pd.read_csv("../data/FairEM/DeepMatcher/DBLP-Scholar/train_others.csv", index_col="id"))

print("========================================")

find_small_freqs_vals(delimiter=',',
                      multiple_sens_attr=False,
                      sens_att_left="left_batting_style",
                      sens_att_right="right_batting_style",
                      df=pd.read_csv("../data/FairEM/DeepMatcher/Cricket/train_others.csv", index_col="id"))

print("========================================")

find_small_freqs_vals(delimiter=',',
                      multiple_sens_attr=False,
                      sens_att_left="left_locale",
                      sens_att_right="right_locale",
                      df=pd.read_csv("../data/FairEM/DeepMatcher/Shoes/train_others.csv", index_col="id"))

print("========================================")

find_small_freqs_vals(delimiter=',',
                      multiple_sens_attr=False,
                      sens_att_left="left_locale",
                      sens_att_right="right_locale",
                      df=pd.read_csv("../data/FairEM/DeepMatcher/Cameras/train_others.csv", index_col="id"))

print("========================================")
