# returns true if all attributes in subgroup1 are present in subgroup2
def clauses_satisfied(subgroup1, subgroup2):
    for i in range(len(subgroup1)):
        if subgroup1[i] == 1 and subgroup2[i] == 0:
            return False
    return True

# converts a k-combination (2-comb (45, 55, 13, 46)) to a subgroup (encoding)
# with 1's at those indices
def comb_to_encoding(combination, full_encoding_len):
    # by definition, 2|len(combination) and 2|full_encoding_len
    boundary = int(len(combination) / 2)
    encoding = np.zeros(full_encoding_len)

    for ind in range(len(combination)):
        if ind < boundary:
            encoding[combination[ind]] = 1
        else:
            encoding[combination[ind] + int(full_encoding_len / 2)] = 1

    return encoding

# given as strings. 
# unfair_subroup is separated by "~", normal_subgroup is just the value of the sens. attribute
def calculate_distance(unfair_subroup, normal_subgroup, normal_subgroup_del):
    unfair_set = set(unfair_subroup.split("~"))

    normal_subgroup_set = set([x.strip() for x in normal_subgroup.split(normal_subgroup_del)])
    distance = 0
    for e1 in unfair_set:
        if e1 not in normal_subgroup_set:
            distance += 1
    for e1 in normal_subgroup_set:
        if e1 not in unfair_set:
            distance += 1
    
    return distance

def f1_score(TP, FP, TN, FN):
    return TP / (TP + 0.5 * (FP + FN))