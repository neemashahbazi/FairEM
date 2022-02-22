from pprint import pprint

def get_id_from_file(fl):
    arr = []
    with open(fl, 'r') as dataset:
        lines = list(dataset)
    lines = lines[1:]

    for line in lines:
        id = line.split(",")[0]
        arr.append(id)
    return arr

arr1 = get_id_from_file("../data/itunes-amazon/test.csv")
arr2 = get_id_from_file("../data/itunes-amazon/train.csv")
arr3 = get_id_from_file("../data/itunes-amazon/validation.csv")

s1 = set(arr1)
s2 = set(arr2)
s3 = set(arr3)

print("S1 = ")
print(s1)
print("S2 = ")
print(s2)
print("S3 = ")
print(s3)