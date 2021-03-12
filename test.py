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
# ---

import logging
import sacrebleu
import pandas as pd
import sys
from simpletransformers.t5 import T5Model, T5Args


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = T5Args()
model_args.max_length = 512
model_args.length_penalty = 1
model_args.num_beams = 10

#model = T5Model("mt5", "persiannlp/mt5-base-parsinlu-opus-translation_fa_en", args=model_args)
model_name = "t5-small"
result_name  = "xWant"
model = T5Model("t5", model_name + "/best_model" , args=model_args)

eval_df = pd.read_csv("data/eval.tsv", sep="\t").astype(str)

truth_values = eval_df.loc[eval_df["prefix"] == "xWant"]["target_text"].tolist()
input_values = eval_df.loc[eval_df["prefix"] == "xWant"]["input_text"].tolist()

pred_values = model.predict(input_values)
pred_values


truth_values

len(truth_values)

# +
bleu_score = sacrebleu.corpus_bleu(pred_values, truth_values)
print(bleu_score.score)

with open(model_name + "/results/" + result_name + '.txt', 'w') as f:
        print("BLEU:" + bleu_score.score, file=f)
        for truth, pred in zip(truth_values, pred_values):
            print(truth,"--", pred)
            print(truth,"--", pred, file =f)
# -

print("--------------------------")
print("BLEU: ", bleu_score.score)


print("predicting")
print(model.predict(["PersonX go to the bar","PersonX hunts"]))
sys.exit()


