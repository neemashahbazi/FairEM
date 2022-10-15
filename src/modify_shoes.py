import pandas as pd

df = pd.read_csv("../data/FairEM/DeepMatcher/Shoes/test.csv")
left = "left_company"
right = "right_company"
left_title = "left_title"
right_title = "right_title"

for idx, row in df.iterrows():
    if "nike" in row[left_title].lower():
        df.loc[idx, left] = "nike"
    if "nike" in row[right_title].lower():
        df.loc[idx, right] = "nike"
    if "puma" in row[left_title].lower():
        df.loc[idx, left] = "puma"
    if "puma" in row[right_title].lower():
        df.loc[idx, right] = "puma"
    if "adidas" in row[left_title].lower():
        df.loc[idx, left] = "adidas"
    if "adidas" in row[right_title].lower():
        df.loc[idx, right] = "adidas"
    if "asics" in row[left_title].lower():
        df.loc[idx, left] = "asics"
    if "asics" in row[right_title].lower():
        df.loc[idx, right] = "asics"
    if "patagonia" in row[left_title].lower():
        df.loc[idx, left] = "patagonia"
    if "patagonia" in row[right_title].lower():
        df.loc[idx, right] = "patagonia"
    if "jordan" in row[left_title].lower():
        df.loc[idx, left] = "jordan"
    if "jordan" in row[right_title].lower():
        df.loc[idx, right] = "jordan"
    if "concave" in row[left_title].lower():
        df.loc[idx, left] = "concave"
    if "concave" in row[right_title].lower():
        df.loc[idx, right] = "concave"
    if "keen" in row[left_title].lower():
        df.loc[idx, left] = "keen"
    if "keen" in row[right_title].lower():
        df.loc[idx, right] = "keen"
    if "ariat" in row[left_title].lower():
        df.loc[idx, left] = "ariat"
    if "ariat" in row[right_title].lower():
        df.loc[idx, right] = "ariat"
    if "under armour" in row[left_title].lower():
        df.loc[idx, left] = "under armour"
    if "under armour" in row[right_title].lower():
        df.loc[idx, right] = "under armour"
    if "joma" in row[left_title].lower():
        df.loc[idx, left] = "joma"
    if "joma" in row[right_title].lower():
        df.loc[idx, right] = "joma"
    if "new balance" in row[left_title].lower():
        df.loc[idx, left] = "new balance"
    if "new balance" in row[right_title].lower():
        df.loc[idx, right] = "new balance"
    if "diadora" in row[left_title].lower():
        df.loc[idx, left] = "diadora"
    if "diadora" in row[right_title].lower():
        df.loc[idx, right] = "diadora"
    if "hogan" in row[left_title].lower():
        df.loc[idx, left] = "hogan"
    if "hogan" in row[right_title].lower():
        df.loc[idx, right] = "hogan"
    if "fjallRaven" in row[left_title].lower():
        df.loc[idx, left] = "fjallRaven"
    if "fjallRaven" in row[right_title].lower():
        df.loc[idx, right] = "fjallRaven"

df.to_csv("../data/FairEM/DeepMatcher/Shoes/test_others__.csv", index=False)
