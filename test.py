# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import logging
import sacrebleu
from pathlib import Path
import pandas as pd
import sys
from simpletransformers.t5 import T5Model, T5Args


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = T5Args()
model_args.max_length = 50
model_args.length_penalty = 2.0
model_args.repetition_penalty = 2.0
model_args.num_beams = 5
model_args.early_stopping = True
#model_args.do_sample = True
#model_args.top_p = 0.3
model_args.num_return_sequences = 5


#model = T5Model("mt5", "persiannlp/mt5-base-parsinlu-opus-translation_fa_en", args=model_args)
model_name = "outputs_t5_small_full_2020/"
task = "xWant"
model = T5Model("t5", model_name, args=model_args)

print("predicting")
#print(model.predict(["xReact: Ali buys a book",
#                     "xReact: Ali fell on his knees"]))


# +
df = pd.read_csv("data/eval.tsv", sep="\t").astype(str)
# Prepare the data for testing
#df = df.groupby('input_text')['target_text'].apply(list)
def my_eval(df, prefix):
    df = df.groupby(['prefix','input_text'], as_index=False).agg({'target_text':lambda x: list(x)})
    truth_values = df.loc[df["prefix"] == prefix]["target_text"].tolist()
    input_values = df.loc[df["prefix"] == prefix]["input_text"].tolist()
    input_values = [prefix + ": " + str(input_text) for input_text in input_values]

    pred_values = model.predict(input_values)
    preds = [item[0] for item in pred_values]
    
    refs = [[] for i in range(5)]
    for item in truth_values:
        for i in range(5):
            if i < len(item):
                refs[i].append(item[i])
            else:
                refs[i].append('')  
    bleu_score = sacrebleu.corpus_bleu(preds, refs)
    return bleu_score, input_values, truth_values, pred_values
    
tasks = list(set(df["prefix"].tolist()))


# +
bleu_score, input_values, truth_values, pred_values = my_eval(df, task)

print("BLEU for " + task + ":" + str(bleu_score.score))

sel_args = ["length_penalty", "repetition_penalty", "num_beams", "do_sample", "top_p", "top_k"]
Path(f'{model_name}/results/').mkdir(exist_ok=True, parents=True)
cc = 1
args = model_args.get_args_for_saving()
res_name = "res_" + task
for key, val in args.items():
    if key in sel_args:
        res_name += "_" + key + "_" + str(val)

with open(f'{model_name}/results/{res_name}.txt', 'w') as f:
        print("BLEU:" + str(bleu_score.score), file=f)
        for inp, truth, pred in zip(input_values, truth_values, pred_values):
            inp = inp.replace("as a result PersonX want","")
            cc += 1
            if cc < 10:
                print(inp,":", truth,"--", pred[0])
            print(inp,":", truth,"--", pred, file =f)
        print("Arguments:", file=f)
        for key, val in args.items():
            print(key, "=", val, file=f)
# -


