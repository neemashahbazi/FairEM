import pandas as pd

dataset = 'dblp-scholar'
dir = '../data/FairEM/DeepMatcher/' + dataset + '/test.csv'
dir_ = '../data/FairEM/DeepMatcher/' + dataset + '/train_others.csv'

df = pd.read_csv(dir)
df_ = pd.read_csv(dir_)

count = 0
for index, row in df.iterrows():
    if row['label'] == 1:
        count += 1

print('train', df_.shape)
print('test', df.shape)
print('count match', count)
