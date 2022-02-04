import deepmatcher as dm


def run_deepmatcher(directory, sensitive_attributes, train="train.csv", validation="validation.csv", test="test.csv", \
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
        prediction_threshold : float, optional (default is 0.8)
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

