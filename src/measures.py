from utils import clauses_satisfied

def accuracy_parity_single(workload, subgroup):
    subgroup_encoding = workload.create_subgroup_encoding_from_subgroup_single(subgroup)
    match_TP = match_FP = match_TN = match_FN = 0
    for entity_to_count in workload.entitites_to_count:
        left_entity = list(entity_to_count)[:workload.find_border_in_key(entity_to_count)]
        right_entity = list(entity_to_count)[workload.find_border_in_key(entity_to_count) + 1:]
        left_entity_encoding = workload.create_subgroup_encoding_from_subgroup_single(left_entity)
        right_entity_encoding = workload.create_subgroup_encoding_from_subgroup_single(right_entity)
        # P(sth | g_i) is calculated.
        # g_i = True
        if clauses_satisfied(subgroup_encoding, left_entity_encoding) or clauses_satisfied(subgroup_encoding, right_entity_encoding):
            match_TP += entity_to_count[workload.TP]
            match_TN += entity_to_count[workload.TN]
            match_FP += entity_to_count[workload.FP]
            match_FN += entity_to_count[workload.FN]
    if (match_TP + match_TN + match_FP + match_FN) == 0: # denominator
        return 0
    else:
        return (match_TP + match_TN) / (match_TP + match_TN + match_FP + match_FN)

def accuracy_parity_pairwise(workload, subgroup):
    print("accuracy_parity_pairwise ENTERED")
    encoding1, encoding2 = workload.create_subgroup_encodings_from_subgroup_pairwise(subgroup)
    match_TP = match_FP = match_TN = match_FN = 0
    for entity_to_count in workload.entitites_to_count:
        left_entity = list(entity_to_count)[:workload.find_border_in_key(entity_to_count)]
        right_entity = list(entity_to_count)[workload.find_border_in_key(entity_to_count) + 1:]
        left_entity_encoding = workload.create_subgroup_encoding_from_subgroup_single(left_entity)
        right_entity_encoding = workload.create_subgroup_encoding_from_subgroup_single(right_entity)
        entity_encoding = left_entity_encoding + right_entity_encoding
        
        # P(sth | g_i) is calculated.
        # g_i = True
        if clauses_satisfied(encoding1, entity_encoding) or clauses_satisfied(encoding2, entity_encoding):
            match_TP += entity_to_count[workload.TP]
            match_TN += entity_to_count[workload.TN]
            match_FP += entity_to_count[workload.FP]
            match_FN += entity_to_count[workload.FN]
    if (match_TP + match_TN + match_FP + match_FN) == 0: # denominator
        return 0
    else:
        return (match_TP + match_TN) / (match_TP + match_TN + match_FP + match_FN)
