import deepmatcher as dm
import pandas as pd
import json
import re


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
    epochs=10, prediction_threshold=0.7):
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

    print("PREDICTION = ", prediction)

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
                curr_val = ""
                if j % 2 != 0:
                    vals[j] = vals[j].strip()
                    if vals[j].startswith("\""):
                        curr_val += vals[j] + ","
                    elif "," in vals[j]:
                        curr_val += "\"" + vals[j] + "\"" + ","
                    else:
                        curr_val += vals[j] + ","
                curr_val += ""

                # print("curr_val = ", curr_val)

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
ditto_format_to_deepmatcher("../data/shoes/", "valid.txt")




# jsonl_to_predictions("../data/shoes/", "ditto_out_test.jsonl")
