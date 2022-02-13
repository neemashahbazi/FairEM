import pandas as pd
import random as rd

def create_workloads_from_file(path, filename, number_of_workloads=40, portion_of_entire_dataset_present_in_workload=0.3):
    df = pd.read_csv(path + "/" + filename)
    length = len(df)
    for i in range(number_of_workloads):
        new_df = df.iloc[:0,:].copy()
        sample_rows = rd.sample(range(0, length), int(portion_of_entire_dataset_present_in_workload * length))
        for row in sample_rows:
            new_df = new_df.append(df.iloc[[row]], ignore_index=False)

        new_df.to_csv(path + "/" + filename[:4] + str(i) + ".csv", index=False)
create_workloads_from_file("../data/itunes-amazon", "test.csv")
