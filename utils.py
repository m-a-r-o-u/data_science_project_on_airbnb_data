import pandas as pd


def get_dichotomous(data):
    '''Use length < 3, to find dichotomous variables'''

    binaries = []
    for k, v in data.items():
        u = [x for x in v.unique() if pd.notnull(x)]
        if len(u) < 3:
            binaries.append(k)
    return binaries
