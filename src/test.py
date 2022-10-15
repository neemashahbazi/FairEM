import pandas as pd

# dir = '../data/FairEM/DeepMatcher/iTunes-Amazon/test_others.csv'
# preds = '../data/FairEM/Ditto/iTunes-Amazon/preds.csv'
# df = pd.read_csv(dir)
# preds = pd.read_csv(preds).values.tolist()
#
# for index, row in df.iterrows():
#     if preds[index][0] == 1 and row['label'] == 0:
#         print('song:', row['left_Song_Name'], '|', row['right_Song_Name'])
#         print('artist:', row['left_Artist_Name'], '|', row['right_Artist_Name'])
#         print('album:', row['left_Album_Name'], '|', row['right_Album_Name'])
#         print('genre:', row['left_Genre'], '|', row['right_Genre'])
#         print('price:', row['left_Price'], '|', row['right_Price'])
#         print('copyright:', row['left_CopyRight'], '|', row['right_CopyRight'])
#         print('duration:', row['left_Time'], '|', row['right_Time'])
#         print('released:', row['left_Released'], '|', row['right_Released'])
#         print("----------------------------------------------------------")

# dir = '../data/FairEM/DeepMatcher/DBLP-ACM/test_others.csv'
# preds = '../data/FairEM/HierMatcher/DBLP-ACM/preds.csv'
# df = pd.read_csv(dir)
# preds = pd.read_csv(preds).values.tolist()
#
# for index, row in df.iterrows():
#     if preds[index][0] ==1 and row['label']==0:
#         print('title:', row['left_title'], '|', row['right_title'])
#         print('author:', row['left_authors'], '|', row['right_authors'])
#         print('venue:', row['left_venue'], '|', row['right_venue'])
#         print('year:', row['left_year'], '|', row['right_year'])
#         print("----------------------------------------------------------")


# dir = '../data/FairEM/DeepMatcher/DBLP-Scholar/test_others.csv'
# preds = '../data/FairEM/GNEM/DBLP-Scholar/preds.csv'
# df = pd.read_csv(dir)
# preds = pd.read_csv(preds).values.tolist()
#
# for index, row in df.iterrows():
#     if  preds[index][0] == row['label']:
#         print('publisher:', row['left_publisher'], '|', row['right_publisher'])
#         print('title:', row['left_title'], '|', row['right_title'])
#         print('author:', row['left_author'], '|', row['right_author'])
#         print('year:', row['left_year'], '|', row['right_year'])
#         print('entry_type:', row['left_ENTRYTYPE'], '|', row['right_ENTRYTYPE'])
#         print('journal:', row['left_journal'], '|', row['right_journal'])
#         print('number:', row['left_number'], '|', row['right_number'])
#         print('volume:', row['left_volume'], '|', row['right_volume'])
#         print('pages:', row['left_pages'], '|', row['right_pages'])
#         print('booktitle:', row['left_booktitle'], '|', row['right_booktitle'])
#         print("----------------------------------------------------------")


dir = '../data/FairEM/DeepMatcher/Cameras/test_others.csv'
preds = '../data/FairEM/DecisionTree/Cameras/preds.csv'
df = pd.read_csv(dir)
preds = pd.read_csv(preds).values.tolist()

for index, row in df.iterrows():
    if preds[index][0] == 0 and row['label'] == 1:
        print('title:', row['left_title'], '||', row['right_title'])

# dir = '../data/FairEM/DeepMatcher/Shoes/test_others.csv'
# preds = '../data/FairEM/GNEM/Shoes/preds.csv'
# df = pd.read_csv(dir)
# preds = pd.read_csv(preds).values.tolist()
#
# for index, row in df.iterrows():
#     if row['left_locale'] != row['right_locale'] and preds[index][0] == row['label']:
#         print('title:', row['left_title'], '|', row['right_title'])


# Ditto considers the structure but it's just another token, it merges everyhitng together, in case
# title: lineage tracing for general data warehouse transformations | data extraction and transformation for the data warehouse
# author: jennifer widom , yingwei cui | case squire
# venue: VLDBJ | SIGMOD
# year: 2003 | 1995

# Ditto
# title: efficient schemes for managing multiversionxml documents | efficient management of multiversion documents by object referencing
# author: shu-yao chien , carlo zaniolo , vassilis j. tsotras | shu-yao chien , vassilis j. tsotras , carlo zaniolo
# venue: VLDBJ | VLDB
# year: 2002 | 2001

# ditto, svm
# title: guest editorial | guest editorial
# author: alon y. halevy | vijay atluri , anupam joshi , yelena yesha
# venue: VLDBJ | VLDBJ
# year: 2002 | 2003


# SVM
# title: guest editorial | guest editorial
# author: alon y. halevy | matthias jarke
# venue: VLDBJ | VLDBJ
# year: 2002 | 1998

# hiermatcher
# title: efficient and cost-effective techniques for browsing and indexing large video databases | effective timestamping in databases
# author: kien a. hua , jung-hwan oh | kristian torp , christian s. jensen , richard thomas snodgrass
# venue: SIGMOD | VLDBJ
# year: 2000 | 2000

# Ditto FP
# title: Canon EF 50mm f/1.4 USM Standard Lens for Canon SLR Cameras - Fixed@en-US Camera Lenses | EISF@en-US || Canon EF 50mm f/1.8 II Lens@en  Canon Lens Standard lens for EOS SLR cameras at Crutchfield.com @en

# MCAN FN
# title: Sony Cyber-shot RX100@en RX100 Prices - CNET@en || Sony Cyber-shot RX100 Zwart - Prijzen @NL Tweakers@NL

# DT FN
# title: Bolso para cámara SLR mediana@es mediana - Case Logic@es || Medium SLR Camera Bag@en Bag - Case Logic@en
