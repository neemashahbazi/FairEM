import csv
import pandas as pd
from pprint import pprint

count_id = 0

def to_deepmatcher_helper(left_vals, right_vals, table, new_schema, res_filename):
    global count_id
    res_file =  open(res_filename, "w")
    curr_line = "id,"
    for sch_attr in new_schema:
        curr_line += sch_attr + ","
    curr_line = curr_line[:-1]
    curr_line += "\n"
    res_file.write(curr_line)
    for line in table:
        left_id = line[0]
        right_id = line[1]
        label = line[2]
        new_line = [label] + left_vals[left_id] + right_vals[right_id]

        res_file.write(str(count_id) + ",")
        count_id += 1
        for i in range(len(new_line)):
            attr_val = new_line[i]
            if "," in attr_val:
                res_file.write("\"")
                res_file.write(attr_val)
                res_file.write("\"")
            else:
                res_file.write(attr_val)
            if i == len(new_line) - 1:
                res_file.write("\n")
            else:
                res_file.write(",")


def to_deepmatcher_input(dataset_name, tableA, tableB, test, train, valid):
    left_table = open(tableA,  "r")

    schema = left_table.readlines()[0].split(",")[1:] # remove the id
    left_schema = ["left_" + sch.strip() for sch in schema]
    right_schema = ["right_" + sch.strip() for sch in schema]
    new_schema = ["label"] + left_schema + right_schema

    left_table = list(csv.reader(open(tableA)))[1:]
    right_table = list(csv.reader(open(tableB)))[1:]
     
    left_vals = {}
    right_vals = {}

    for left_line in left_table:
        left_id = str(left_line[0])
        left_line = left_line[1:] # remove the id
        left_vals[left_id] = left_line
    for right_line in right_table:
        right_id = str(right_line[0])
        right_line = right_line[1:] # remove the id
        right_vals[right_id] = right_line

    train_table = list(csv.reader(open(train)))[1:]
    test_table = list(csv.reader(open(test)))[1:]
    valid_table = list(csv.reader(open(valid)))[1:]

    to_deepmatcher_helper(left_vals, right_vals, train_table, new_schema, "../data/dblp-acm/train_converted.csv")
    to_deepmatcher_helper(left_vals, right_vals, valid_table, new_schema, "../data/dblp-acm/valid_converted.csv")
    to_deepmatcher_helper(left_vals, right_vals, test_table, new_schema, "../data/dblp-acm/test_converted.csv")

# def to_ditto_input(dataset_name, tableA, tableB, test, train, valid):
    # left_table = open(tableA,  "r")

    # schema = left_table.readlines()[0].split(",")[1:] # remove the id
    # left_table = list(csv.reader(open(tableA)))[1:]
    # right_table = list(csv.reader(open(tableB)))[1:]

    # left_vals = {}
    # right_vals = {}
    
    # for left_line in left_table:
    #     left_id = str(left_line[0])
    #     left_line = left_line[1:] # remove the id
    #     left_vals[left_id] = left_line
    # for right_line in right_table:
    #     right_id = str(right_line[0])
    #     right_line = right_line[1:] # remove the id
    #     right_vals[right_id] = right_line

    # print("left_table = ")
    # pprint(left_table)
    

    # returns train.txt, valid.txt, test.txt

def from_deepmatcher_input_to_ditto_input_helper(path, data_in, data_out):
    df = pd.read_csv(path + data_in, index_col = "id")
    schema = list(df.columns)[1:]
    ditto_schema = [x.replace("left_", "").replace("right_","") for x in schema]
    
    res_file =  open(path + data_out, "w")
    
    for idx, row in df.iterrows():
        label = row["label"]
        ditto_row = ""
        for i in range(len(schema)):
            ditto_row += "COL " + ditto_schema[i] + " "
            ditto_row += "VAL " + str(row[schema[i]]) + " "
            if "left_" in schema[i] and "right_" in schema[i+1]:
                ditto_row += "\t"

        ditto_row += "\t" + str(label)
        res_file.write(ditto_row)
        res_file.write("\n")

def from_deepmatcher_input_to_ditto_input(dataset_name, path, test, train, valid):
    from_deepmatcher_input_to_ditto_input_helper(path, test, "test.txt")
    from_deepmatcher_input_to_ditto_input_helper(path, train, "train.txt")
    from_deepmatcher_input_to_ditto_input_helper(path, valid, "validation.txt")
    
    
# to_deepmatcher_input("dblp-acm", "../data/dblp-acm/tableA.csv", 
#                     "../data/dblp-acm/tableB.csv", 
#                     "../data/dblp-acm/test.csv", 
#                     "../data/dblp-acm/train.csv",
#                     "../data/dblp-acm/valid.csv")

# to_ditto_input("dblp-acm", "../data/dblp-acm/tableA.csv", 
#                     "../data/dblp-acm/tableB.csv", 
#                     "../data/dblp-acm/test.csv", 
#                     "../data/dblp-acm/train.csv",
#                     "../data/dblp-acm/valid.csv")

# from_deepmatcher_input_to_ditto_input("itunes-amazon", 
#                                     "../data/itunes-amazon/",
#                                     "test.csv",
#                                     "train.csv",
#                                     "validation.csv")

from_deepmatcher_input_to_ditto_input("dblp-acm", 
                                    "../data/dblp-acm/",
                                    "test_converted.csv",
                                    "train_converted.csv",
                                    "valid_converted.csv")


def convert_small_workloads(path):
    test = "test"
    for i in range(40):
        test1 = test + str(i)
        from_deepmatcher_input_to_ditto_input_helper(path, test1 + ".csv", test1 + ".txt")
        

# convert_small_workloads("../data/itunes-amazon/")