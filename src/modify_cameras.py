import pandas as pd

df = pd.read_csv("../data/FairEM/DeepMatcher/Cameras/test.csv")
left = "left_company"
right = "right_company"
left_title = "left_title"
right_title = "right_title"

for idx, row in df.iterrows():
    if "canon" in row[left_title].lower():
        df.loc[idx, left] = "canon"
    if "canon" in row[right_title].lower():
        df.loc[idx, right] = "canon"
    if "sony" in row[left_title].lower():
        df.loc[idx, left] = "sony"
    if "sony" in row[right_title].lower():
        df.loc[idx, right] = "sony"
    if "leica" in row[left_title].lower():
        df.loc[idx, left] = "leica"
    if "leica" in row[right_title].lower():
        df.loc[idx, right] = "leica"
    if "nikon" in row[left_title].lower():
        df.loc[idx, left] = "nikon"
    if "nikon" in row[right_title].lower():
        df.loc[idx, right] = "nikon"
    if "d-link" in row[left_title].lower():
        df.loc[idx, left] = "d-link"
    if "d-link" in row[right_title].lower():
        df.loc[idx, right] = "d-link"
    if "fujifilm" in row[left_title].lower():
        df.loc[idx, left] = "fujifilm"
    if "fujifilm" in row[right_title].lower():
        df.loc[idx, right] = "fujifilm"
    if "gopro" in row[left_title].lower():
        df.loc[idx, left] = "gopro"
    if "gopro" in row[right_title].lower():
        df.loc[idx, right] = "gopro"
    if "manfrotto" in row[left_title].lower():
        df.loc[idx, left] = "manfrotto"
    if "manfrotto" in row[right_title].lower():
        df.loc[idx, right] = "manfrotto"
    if "transcend" in row[left_title].lower():
        df.loc[idx, left] = "transcend"
    if "transcend" in row[right_title].lower():
        df.loc[idx, right] = "transcend"
    if "arlo" in row[left_title].lower():
        df.loc[idx, left] = "arlo"
    if "arlo" in row[right_title].lower():
        df.loc[idx, right] = "arlo"
    if "logitech" in row[left_title].lower():
        df.loc[idx, left] = "logitech"
    if "logitech" in row[right_title].lower():
        df.loc[idx, right] = "logitech"
    if "garmin" in row[left_title].lower():
        df.loc[idx, left] = "garmin"
    if "garmin" in row[right_title].lower():
        df.loc[idx, right] = "garmin"
    if "sigma" in row[left_title].lower():
        df.loc[idx, left] = "sigma"
    if "sigma" in row[right_title].lower():
        df.loc[idx, right] = "sigma"
    if "kupo" in row[left_title].lower():
        df.loc[idx, left] = "kupo"
    if "kupo" in row[right_title].lower():
        df.loc[idx, right] = "kupo"
    if "panasonic" in row[left_title].lower():
        df.loc[idx, left] = "panasonic"
    if "panasonic" in row[right_title].lower():
        df.loc[idx, right] = "panasonic"
    if "pentax" in row[left_title].lower():
        df.loc[idx, left] = "pentax"
    if "pentax" in row[right_title].lower():
        df.loc[idx, right] = "pentax"
    if "kingston" in row[left_title].lower():
        df.loc[idx, left] = "kingston"
    if "kingston" in row[right_title].lower():
        df.loc[idx, right] = "kingston"
    if "sandisk" in row[left_title].lower():
        df.loc[idx, left] = "sandisk"
    if "sandisk" in row[right_title].lower():
        df.loc[idx, right] = "sandisk"
    if "olympus" in row[left_title].lower():
        df.loc[idx, left] = "olympus"
    if "olympus" in row[right_title].lower():
        df.loc[idx, right] = "olympus"
    if "netgear" in row[left_title].lower():
        df.loc[idx, left] = "netgear"
    if "netgear" in row[right_title].lower():
        df.loc[idx, right] = "netgear"
    if "case logic" in row[left_title].lower():
        df.loc[idx, left] = "case logic"
    if "case logic" in row[right_title].lower():
        df.loc[idx, right] = "case logic"
    if "mefoto" in row[left_title].lower():
        df.loc[idx, left] = "mefoto"
    if "mefoto" in row[right_title].lower():
        df.loc[idx, right] = "mefoto"
    if "veho" in row[left_title].lower():
        df.loc[idx, left] = "veho"
    if "veho" in row[right_title].lower():
        df.loc[idx, right] = "veho"
    if "vigilance" in row[left_title].lower():
        df.loc[idx, left] = "vigilance"
    if "vigilance" in row[right_title].lower():
        df.loc[idx, right] = "vigilance"
    if "vanguard" in row[left_title].lower():
        df.loc[idx, left] = "vanguard"
    if "vanguard" in row[right_title].lower():
        df.loc[idx, right] = "vanguard"
    if "tamron" in row[left_title].lower():
        df.loc[idx, left] = "tamron"
    if "tamron" in row[right_title].lower():
        df.loc[idx, right] = "tamron"
    if "samsung" in row[left_title].lower():
        df.loc[idx, left] = "samsung"
    if "samsung" in row[right_title].lower():
        df.loc[idx, right] = "samsung"
    if "chromo" in row[left_title].lower():
        df.loc[idx, left] = "chromo"
    if "chromo" in row[right_title].lower():
        df.loc[idx, right] = "chromo"
    if "neewer" in row[left_title].lower():
        df.loc[idx, left] = "neewer"
    if "neewer" in row[right_title].lower():
        df.loc[idx, right] = "neewer"




df.to_csv("../data/FairEM/DeepMatcher/Cameras/test_others__.csv", index=False)

# df = pd.read_csv("modified_shoes.csv")
#
# pprint(set(df[left].tolist()))
# pprint(set(df[right].tolist()))
