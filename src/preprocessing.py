import deepmatcher as dm
import pandas as pd
import json
import re
import numpy as np


# the following function is used to combine multiple sensitive attributes into one.
# For example, Sex = {Male, Female} and 
# Ethnicity = {white, asian, middle-eastern, black, hispanic} would be
# transformed to Sex_Ethnicity = Sex X Ethinicity (Cartesian product)
def convert_to_level_k_subgroups(directory, table, sens_attributes):
    df = pd.read_csv(directory + "/" + table)
    sens_att_indices = []
    new_sens_attributes = []
    
    new_sens_attribute_name = ""
    for sens_attribute in sens_attributes:
        new_sens_attribute_name += sens_attribute + " X "
    new_sens_attribute_name = new_sens_attribute_name[:-3]

    for index, row in df.iterrows():
        # new_sens_attribute = "\""
        new_sens_attribute = ""
        for sens_attribute in sens_attributes:
            new_sens_attribute += row[sens_attribute] + " "
        new_sens_attribute = new_sens_attribute.strip()
        # new_sens_attribute += "\""
        new_sens_attributes.append(new_sens_attribute)
        
    df[new_sens_attribute_name] = new_sens_attributes
    for sens_attribute in sens_attributes:
        df = df.drop(sens_attribute, 1)
    df.to_csv(directory + "/new_" + table, index = False)

def run_deepmatcher(directory, train="train.csv", validation="validation.csv", test="test.csv", \
    epochs=15, prediction_threshold=0.7):
    """
        Parameters
        ----------
        directory : str
            Location of input .csv files for deepmatcher
        train : str
            Name of training .csv file.
        validation : str
            Name of validation .csv file.
        test : str, optional
            Name of test .csv file.
        epochs : int, optional (default is 10)
            Number of epochs to run deepmatcher.
        prediction_threshold : float, optional (default is 0.7)
            If deepmatcher returns a score bigger than this prediction_threshold, the entity pair is a match.

        Returns
        -------
        An array matching an entity pair from the test dataset
        to a boolean value returned by deepmatcher. 
        """

    train, validation, test = dm.data.process(path=directory, train=train, validation=validation, test=test)

    dm_model = dm.MatchingModel()
    dm_model.run_train(train, validation, best_save_path='best_model.pth', epochs=epochs)
    dm_scores = dm_model.run_prediction(test)
    prediction = [True if dm_scores.iloc[idx]["match_score"] > prediction_threshold else False for idx in range(len(dm_scores))]

    return prediction
    
#def run_ditto( What is the input? Filled in by Nima ): 
 #   returns prediction array

def jsonl_to_predictions(path, file):
    predictions = []
    location = path + file if path.endswith("/") else path + "/" + file
    with open(location, 'r') as json_file:
        json_list = list(json_file)

    for json_line in json_list:
        line = json.loads(json_line)
        predictions.append(line["match"])
        
    return predictions

def deepmatcher_output_to_predictions(path, file):
    predictions = []
    location = path + file if path.endswith("/") else path + "/" + file
    with open(location, 'r') as f:
        lines = list(f)

    for line in lines:
        prediction = line.strip()
        predictions.append(int(prediction))
        
    return predictions


def ditto_format_to_deepmatcher(path, file):
    predictions = []
    location = path + file if path.endswith("/") else path + "/" + file
    with open(location, 'r') as dataset:
        lines = list(dataset)

    left_schema = []
    right_schema = []

    first_line = re.split("COL | VAL", lines[0].split("\t")[0])[1:]
    for i in range(len(first_line)):
        if i % 2 == 0:
            left_schema.append("left_" + first_line[i])
            right_schema.append("right_" + first_line[i])
    labels = []
    csv_values = ""
    
    for k in range(len(lines)):
        line = lines[k]
        spl = line.split("\t")
        csv_values += str(k) + "," + spl[2][:-1] + ","
        
        for i in range(2):
            vals = re.split("COL | VAL", spl[i])[1:]          
            for j in range(0, len(vals)):
                
                # print("j = ", j, "vals[j] = ", vals[j])
                if j % 2 != 0:
                    vals[j] = vals[j].strip()
                    vals[j] = vals[j].replace("\"","")
                    curr_val = "\"" + vals[j] + "\"" + ","
                    csv_values += curr_val
        csv_values = csv_values[:-1]
        csv_values += "\n"

        # break

    title = "id,label,"
    for sch in left_schema:
        title += sch + ","
    for sch in right_schema:
        title += sch + ","
    title = title[:-1]
    title += "\n"

    csv_values = title + csv_values
    
    print(csv_values)

   
# ditto_format_to_deepmatcher("../data/shoes/", "test.txt")
# ditto_format_to_deepmatcher("../data/shoes/", "train.txt")
# ditto_format_to_deepmatcher("../data/shoes/", "valid.txt")

# jsonl_to_predictions("../data/shoes/", "ditto_out_test.jsonl")

def run_deepmatcher_on_existing_datasets(dataset="shoes"):
    predictions = run_deepmatcher("../data/" + dataset +"/",
                    train="train.csv", 
                    validation="valid.csv",
                    test="test.csv",
                    epochs = 15)
    with open("../data/" + dataset + "/deepmatcher_out_15.txt", "w") as f:
        for prediction in predictions:
            f.write("1\n" if prediction else "0\n")

# threshold optimized for max f1 score
def run_deepmatcher_on_existing_datasets_with_match_score_threshold(dataset, threshold):
    df_with_raw_score = pd.read_csv("../data/" + dataset + "/dm_results_with_raw_score.csv")
    predictions = [True if df_with_raw_score.iloc[idx]["match_score"] > threshold else False for idx in range(len(df_with_raw_score))]

    with open("../data/" + dataset + "/deepmatcher_out_15_" + str(threshold) + ".txt", "w") as f:
        for prediction in predictions:
            f.write("1\n" if prediction else "0\n")

# for threshold in [0.2, 0.4, 0.6, 0.8, 0.9]:
#     run_deepmatcher_on_existing_datasets_with_match_score_threshold(dataset="itunes-amazon", threshold = threshold)

# run_deepmatcher_on_existing_datasets_with_match_score_threshold(dataset="dblp-acm", threshold = 0.26)
# run_deepmatcher_on_existing_datasets_with_match_score_threshold(dataset="shoes", threshold = 0.08)

def add_locale_to_shoes_dataset():
    test_demographic_groups = "../data/shoes/test_demographic_groups.txt"
    test_file = "../data/FairEM/DeepMatcher/Shoes/test.csv"
    test_file_with_locale = "../data/FairEM/DeepMatcher/Shoes/test_locale.csv"
    
    left_locale = []
    right_locale = []
    with open(test_demographic_groups) as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace("\"", "")
            line = line[:-1]
            spl = line.split(":")
            left_locale.append(spl[0])
            right_locale.append(spl[1])
    
    df = pd.read_csv(test_file, index_col="id")
    
    df["left_locale"] = left_locale
    df["right_locale"] = right_locale

    df.to_csv(test_file_with_locale, index = "id")

def make_binary_based_on_threshold(dataset, test_file, pred_value="pred", threshold=0.7):
    test_file = "../data/" + dataset + "/" + test_file
    
    df = pd.read_csv(test_file)
    df[pred_value] = np.where(df[pred_value] > threshold, 1, 0)

    df.to_csv(test_file[:-4] + "_binary.csv", index=False)
    
# make_binary_based_on_threshold("itunes-amazon", "HierMatch_pred.csv", "match_score")