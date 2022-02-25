from pprint import pprint
import pandas as pd

df = pd.read_csv("../data/shoes/test.csv")
left = "left_locale"
right = "right_locale"

for idx, row in df.iterrows():
    if "de" in row[left]:
        df.loc[idx,left] = "de"
    if "de" in row[right]:
        df.loc[idx,right] = "de"
    if "en" in row[left]:
        df.loc[idx,left] = "en"
    if "en" in row[right]:
        df.loc[idx,right] = "en"
    if "fr" in row[left]:
        df.loc[idx,left] = "fr"
    if "fr" in row[right]:
        df.loc[idx,right] = "fr"
    if "es" in row[left]:
        df.loc[idx,left] = "es"
    if "es" in row[right]:
        df.loc[idx,right] = "es"
    if "it" in row[left]:
        df.loc[idx,left] = "it"
    if "it" in row[right]:
        df.loc[idx,right] = "it"
    if "nl" in row[left]:
        df.loc[idx,left] = "nl"
    if "nl" in row[right]:
        df.loc[idx,right] = "nl"
    if "eu" in row[left]:
        df.loc[idx,left] = "eu"
    if "eu" in row[right]:
        df.loc[idx,right] = "eu"
    if "pl" in row[left]:
        df.loc[idx,left] = "pl"
    if "pl" in row[right]:
        df.loc[idx,right] = "pl"
    if "pt" in row[left]:
        df.loc[idx,left] = "pt"
    if "pt" in row[right]:
        df.loc[idx,right] = "pt"
    if "cs" in row[left]:
        df.loc[idx,left] = "cs"
    if "cs" in row[right]:
        df.loc[idx,right] = "cs"
    if "bg" in row[left]:
        df.loc[idx,left] = "bg"
    if "bg" in row[right]:
        df.loc[idx,right] = "bg"
    
        


df.to_csv("modified_shoes.csv", index=False)



df = pd.read_csv("modified_shoes.csv")

pprint(set(df[left].tolist()))
pprint(set(df[right].tolist()))