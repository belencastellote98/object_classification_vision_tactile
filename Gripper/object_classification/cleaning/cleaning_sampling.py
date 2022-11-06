import pandas as pd
import numpy as np
import matplotlib as plt

def cleaning_function (sampler_txt):

    df = pd.read_csv(sampler_txt, sep=", ")
    df_notime = df.drop(columns=["Time"])
    values = pd.DataFrame(columns = df_notime.columns)
    possible= pd.DataFrame(columns = df.columns)

    flag=0
    count=0
    last_update = df[df.index==0]
    for i in range(len(df)):
        if df_notime.mean(axis=1)[i]!=0:
            flag=1
            possible = df[df.index==i]
            pos_index = i
            count=0
        # elif len(values)>=100:
        #     pass
        elif np.array(possible.Time)-np.array(last_update.Time)<0:
            last_update = df[df.index==i]
        # elif flag==1 and df_notime.mean(axis=1)[i]==0 and np.array(possible.Time)-np.array(last_update.Time)>2: 
        elif flag==1 and df_notime.mean(axis=1)[i]==0 and count>=15 and np.array(possible.Time)-np.array(last_update.Time)>2: 
            #I will have to rethink about the time distance betweeen each measure. maybe depending on the measured object
            flag=0
            values = pd.concat([values, df_notime[df_notime.index==pos_index]])
            last_update=possible
        elif df_notime.mean(axis=1)[i]==0:
            count+=1
    locals()["values_tree_test_full_bottle"]=values.copy()
            
    #values.to_excel("Dataset.xlsx", sheet_name=name)
    return values