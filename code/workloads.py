import numpy as np
from itertools import combinations

class Workload:
    # encoding is a list of vectors. len(encoding) = # of entities pairs
    # each vector is an encoding for that particular entity pair
    def __init__(self, df, sens_att_left, sens_att_right, prediction, label_column="label", multiple_sens_attr = False, delimiter=","):
        self.df = df
        self.label_column = label_column
        self.prediction = prediction
        self.sens_att_left = sens_att_left
        self.sens_att_right = sens_att_right
        self.multiple_sens_attr = multiple_sens_attr
        self.delimiter = delimiter
        self.sens_attr_vals = find_all_sens_attr()
        self.sens_att_to_index = create_sens_att_to_index():
        self.encoding = self.encode()
        
        self.TP = 0
        self.FP = 1
        self.TN = 2
        self.FN = 3

        self.entitites_to_count = self.create_entities_to_count()

    def find_all_sens_attr(self):
        sens_att = set()
        for it_index, row in self.df.iterrows():
            left = row[self.sens_att_left]
            right = row[self.sens_att_right]
            if self.multiple_sens_attr:
                for item in left.split(self.delimiter):
                    sens_att.add(item.strip())
                for item in right.split(self.delimiter):
                    sens_att.add(item.strip())
            else:
                sens_att.add(left.strip())
                sens_att.add(right.strip())

        sens_attr_vals = list(sens_att)
        sens_attr_vals.sort()
        return sens_attr_vals 

    def encode(self):
        encoding = []
        for it_index, row in self.df.iterrows():
            left_att = row[self.sens_att_left]
            right_att = row[self.sens_att_right]
            left_vector = np.zeros(len(self.sens_attr_vals))
            right_vector = np.zeros(len(self.sens_attr_vals))
            if self.multiple_sens_attr:
                for item in left_att.split(self.delimiter):
                    curr_item = item.strip()
                    idx = self.sens_attr_vals.index(curr_item)
                    left_vector[idx] = 1
                for item in right_att.split(self.delimiter):
                    curr_item = item.strip()
                    idx = self.sens_attr_vals.index(curr_item)
                    right_vector[idx] = 1
            else:
                curr_item = left_att.strip()
                idx = self.sens_attr_vals.index(curr_item)
                left_vector[idx] = 1
                curr_item = right_att.strip()
                idx = self.sens_attr_vals.index(curr_item)
                right_vector[idx] = 1
            encoding.append(np.concatenate([left_vector, right_vector]))
        return encoding

    def create_sens_att_to_index(self):
        sens_att_to_index = {}
        for i in range(len(self.sens_attr_vals)):
            att = self.sens_att_vals[i]
            sens_att_to_index[att] = i
        return sens_att_to_index

    def create_hashmap_key(self, row):
        left_att = row[self.sens_att_left]
        right_att = row[self.sens_att_right]
        
        key_left = []
        key_right = []

        if multiple_sens_attr:
            for item in left_att.split(self.delimiter):
                curr_item = item.strip()
                key_left.append(self.sens_att_to_index[curr_item])
            for item in right_att.split(self.delimiter):
                curr_item = item.strip()
                key_right.append(self.sens_att_to_index[curr_item])
            key_left.sort()
            key_right.sort()
        else:
            curr_item = left_att.strip()
            key_left.append(self.sens_att_to_index[curr_item])
            curr_item = right_att.strip()
            key_right.append(self.sens_att_to_index[curr_item])
            

        # -1 added as a delimiter between the left and right keys
        res = key_left + [-1] + key_right if key_left[0] <= key_right[0] else key_right + [-1] + key_left
        return tuple(res)

    def create_entities_to_count(self):
        entitites_to_count = {}
        for ind, row in self.df.iterrows():
            key = self.create_hashmap_key(row)
            entitites_to_count[key] = [0,0,0,0]
            self.fill_entities_to_count(entitites_to_count, key, ind, self.prediction[ind], row[self.label_column])
        return entitites_to_count

    def fill_entities_to_count(self, entitites_to_count, key, ind, pred, ground_truth):
        if pred:
            if ground_truth:
                entitites_to_count[key][self.TP] += 1
            else:
                entitites_to_count[key][self.FP] += 1
        else:
            if ground_truth:
                entitites_to_count[key][self.FN] += 1
            else:
                entitites_to_count[key][self.TN] += 1
    
    def find_border_in_key(self, key):
        return key.index(-1)
    
    def create_k_combs(self, k):
        k_combs = {}
        for entity in self.entitites_to_count:
            number_of_pairs_with_entity = sum(self.entitites_to_count[entity]) + 1 # because of -1 for the border
            border = self.find_border_in_key(entity)
            for comb1 in combinations(entity[:border], k):
                for comb2 in combinations(entity[border+1:], k):
                    k_comb = comb1 + comb2
                    if k_comb not in k_combs:
                        k_combs[k_comb] = number_of_pairs_with_entity
                    else:
                        k_combs[k_comb] = k_combs[k_comb] + number_of_pairs_with_entity
        return k_combs
    