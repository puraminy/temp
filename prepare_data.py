import pandas as pd
import json


import csv
from pathlib import Path
from tqdm import tqdm
    
Path("data/").mkdir(parents=True, exist_ok=True)

       
def write_tsv(src, split, sel_cols):
    df = pd.read_csv(src,index_col=0)
    df.iloc[:,:9] = df.iloc[:,:9].apply(lambda col: col.apply(json.loads))    
    skip =0
    fname = "data/" + split + ".tsv"    
    with open(fname, 'w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(["prefix", "input_text", "target_text"])
        for row, cols in tqdm(df.iterrows(), total=df.shape[0]): 
            #row = row + " as a result PersonX want "
            for col in df.columns:
                for c in cols[col]:
                    if len(c) >= 60:
                        skip+=1

                    if ((sel_cols == ["All"] or col in sel_cols)
                        and col != "prefix" 
                        and col != "split" and len(c) < 60):
                        writer.writerow([col, row, c])                

    
print("train set ...")
write_tsv("v4_atomic_trn.csv", "train", ["All"])# ["xWant","xAttr", "xNeed"])
print("test set ...")
write_tsv("v4_atomic_tst.csv", "test", ["All"])#, ["xWant","xAttr", "xNeed"])
print("eval set ...")
write_tsv("v4_atomic_dev.csv", "eval", ["All"])#, ["xWant","xAttr", "xNeed"])


# -



