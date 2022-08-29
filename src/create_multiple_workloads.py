import pandas as pd
import random as rd

def create_workloads_from_file(dataset, path, testcsv, predictions_file, number_of_workloads=40, portion_of_entire_dataset_present_in_workload=0.3):
    df = pd.read_csv(path + "/" + dataset + "/" + testcsv)
    length = len(df)

    with open(path + "/" + dataset + "/" + predictions_file, "r") as preds:
        preds_lines = list(preds)

    for i in range(number_of_workloads):
        dot = predictions_file.find(".")
        res_preds_file = open(path + "/" + dataset + "/multiple_workloads/" + predictions_file[:dot] + "_" + str(i) + predictions_file[dot:], "w")
        new_df = df.iloc[:0,:].copy()
        sample_rows = rd.sample(range(0, length), int(portion_of_entire_dataset_present_in_workload * length))
        for row in sample_rows:
            new_df = new_df.append(df.iloc[[row]], ignore_index=False)
            res_preds_file.write(preds_lines[row])


        new_df.to_csv(path + "/" + dataset + "/multiple_workloads/" + testcsv[:-4] + str(i) + ".csv", index=False)

# following lines need to be run only once
# create_workloads_from_file("shoes", "../data/", "test_others.csv", "ditto_out_test.jsonl")
# create_workloads_from_file("shoes", "../data/", "test_others.csv", "deepmatcher_out_15.txt")
# create_workloads_from_file("itunes-amazon", "../data/", "test_others.csv", "ditto_out_test.jsonl")
# create_workloads_from_file("itunes-amazon", "../data/", "test_others.csv", "deepmatcher_out_15.txt")
# create_workloads_from_file("dblp-acm", "../data/", "test.csv", "ditto_out_test.jsonl")
# create_workloads_from_file("dblp-acm", "../data/", "test.csv", "deepmatcher_out_15.txt")

# create_workloads_from_file("citation", "../data", "test.csv", "deepmatcher_out_15.txt")

