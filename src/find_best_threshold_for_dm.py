import deepmatcher as dm
import pandas as pd

def run_deepmatcher(directory, train="train.csv", validation="validation.csv", test="test.csv", \
    epochs=15):
    
    train, validation, test = dm.data.process(path=directory, train=train, validation=validation, test=test)

    dm_model = dm.MatchingModel()
    dm_model.run_train(train, validation, best_save_path='best_model.pth', epochs=epochs)
    dm_scores = dm_model.run_prediction(test)

    dm_scores.to_csv(directory + "dm_results_with_raw_score.csv")
    

def create_pure_dm_scores():
    datasets = ["dblp-acm", "itunes-amazon", "shoes"]
    for dataset in datasets:
        run_deepmatcher("../data/" + dataset + "/",
                        train="train.csv", 
                        validation="valid.csv",
                        test="test.csv",
                        epochs = 15)

# create_pure_dm_scores()

def find_optimal_threshold(test_file, dm_scores):
    df = pd.read_csv(test_file)
    df_dm = pd.read_csv(dm_scores)
    df["match_score"] = df_dm["match_score"]
    
    f1 = 0
    best_threshold = 0

    for i in range(100):
        prediction_threshold = round(i/100, 2)
        binary_prediction = [True if df.iloc[idx]["match_score"] > prediction_threshold else False for idx in range(len(df))]

        df["binary_prediction"] = binary_prediction

        new_f1 = find_f1_score(df)
        if new_f1 > f1:
            f1 = new_f1
            best_threshold = prediction_threshold

    return best_threshold
        

def find_f1_score(df):
    TP = 0
    FP = 1
    TN = 2
    FN = 3
    label_column = "label"
    prediction_column = "binary_prediction"
    for ind, row in df.iterrows():
        pred = row[prediction_column]
        ground_truth = row[label_column]
        if pred:
            if ground_truth:
                TP += 1
            else:
                FP += 1
        else:
            if ground_truth:
                FN += 1
            else:
                TN += 1

    # print(TP, FP, TN, FN)
    
    f1 = TP / (TP + 0.5 * (FP + FN))
    return f1

# datasets = ["dblp-acm", "itunes-amazon", "shoes"]
# for dataset in datasets:
#     thr = find_optimal_threshold("../data/" + dataset + "/test.csv",
#                             "../data/" + dataset + "/dm_results_with_raw_score.csv")
#     print("dataset =", dataset, "threshold =", thr)
