from preprocessing import run_deepmatcher

# unfair_subgroup is given as a string
def distance_analysis(self, unfair_subgroup, directory, full_workload_test):
    predictions = run_deepmatcher(directory, epochs = 2)
    workload = wl.Workload(pd.read_csv(directory + "/test.csv"), "left_Genre", 
                            "right_Genre", predictions, label_column = "label", 
                            multiple_sens_attr = True, delimiter = ",", single_fairness = True,
                            k_combinations=1)


# input = subgroups passed when testing for fairness
# and subgroup index of a group that turned out to be unfair
def create_fairness_per_bin(self, subgroups, subgroup_index, k_combs, number_of_bins=5):
    unfair_subgroup = subgroups[subgroup_index]
    print("UNFAIR_SUBGROUP", unfair_subgroup)

    # distance_groups is a hashmap where the key is distance,
    # and value is a list of entity pairs (rows from a workload) 
    # having that distance with the unfair_subroup
    distance_groups = self.create_distance_groups(unfair_subgroup)
    # bin_df = self.create_bins_by_frequency(distance_groups, number_of_bins=5)
    # fairness_per_bin = self.calculate_fairness_per_bin(bin_df, number_of_bins)

    # return fairness_per_bin

    # print(distance_groups)



    print("!@#!@#!@$!@$#!$!@$")
    self.get_entity_pairs_from_all_workloads()

def create_distance_groups(self, unfair_subgroup):
    distance_groups = {}
    all_antity_pairs = {}
    for workload in self.workloads:
        unfair_subgroup_encoding = None
        if self.single_fairness:
            unfair_subgroup_encoding = workload.create_subgroup_encoding_from_subgroup_single(unfair_subgroup)
        else:
            unfair_subgroup = workload.create_subgroup_encoding_from_subgroup_single(unfair_subgroup)
        self.add_to_distance_groups_from_workload(workload,distance_groups,
                                                    unfair_subgroup_encoding)
    return distance_groups

def get_entity_pairs_from_all_workloads(self):
    entity_pairs_visited = {}
    for workload in self.workloads:
        for idx, row in workload.df.iterrows():
            entity_pairs_visited[row] = False

    print(entity_pairs_visited)

def add_to_distance_groups_from_workload(self, workload, distance_groups, unfair_subgroup_encoding):
    df = workload.df

    for idx, row in df.iterrows():
        current_encoding = workload.encoding[idx]
        curr_dist = calculate_distance(unfair_subgroup_encoding, current_encoding)
        # create list that is a value in the dict
        if curr_dist not in distance_groups:
            distance_groups[curr_dist] = []
        # a group has a key distance and a value list of strings:
        # each string is a concatenation of the workload index and the index 
        # of the row inside of the workload
        distance_groups[curr_dist].append(str(workload_idx) + "_" + str(idx))

