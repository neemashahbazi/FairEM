import csv
import pandas as pd

def to_deepmatcher_helper(left_vals, right_vals, table, new_schema, res_filename):
    res_file =  open(res_filename, "w")
    for sch_attr in new_schema:
        res_file.write(sch_attr)
        if sch_attr is not "label":
            res_file.write(", ")
        else:
            res_file.write("\n")
    for line in table:
        left_id = line[0]
        right_id = line[1]
        label = line[2]
        new_line = left_vals[left_id] + right_vals[right_id] + [label]

        for attr_val in new_line:
            if "," in attr_val:
                res_file.write("\"")
                res_file.write(attr_val)
                res_file.write("\"")
            else:
                res_file.write(attr_val)
            if attr_val == new_line[len(new_line) - 1]:
                res_file.write("\n")
            else:
                res_file.write(",")


def to_deepmatcher_input(dataset_name, tableA, tableB, test, train, valid):
    left_table = open(tableA,  "r")

    schema = left_table.readlines()[0].split(",")[1:] # remove the id
    left_schema = ["left_" + sch.strip() for sch in schema]
    right_schema = ["right_" + sch.strip() for sch in schema]
    new_schema = left_schema + right_schema + ["label"]

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
    to_deepmatcher_helper(left_vals, right_vals, test_table, new_schema, "../data/dblp-acm/test_converted.csv")
    to_deepmatcher_helper(left_vals, right_vals, valid_table, new_schema, "../data/dblp-acm/valid_converted.csv")


    
    




    
    # return train.csv, valid.csv test.csv

# def to_ditto_input(tableA, tableB, test, train, valid):
    # returns train.txt, valid.txt, test.txt


to_deepmatcher_input("dblp-acm", "../data/dblp-acm/tableA.csv", "../data/dblp-acm/tableB.csv",
                    "../data/dblp-acm/test.csv", "../data/dblp-acm/train.csv",
                    "../data/dblp-acm/valid.csv")