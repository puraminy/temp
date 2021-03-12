# ---
# jupyter:
#   jupytext:
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
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
#import wandb
#wandb.login()

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_df = pd.read_csv("data/train.tsv", sep="\t").astype(str)
eval_df = pd.read_csv("data/eval.tsv", sep="\t").astype(str)

#train_df = train_df.truncate(after=1000)
#train_df["prefix"] = ""
#eval_df["prefix"] = ""

model_args = T5Args()
model_args.max_seq_length = 59
model_args.train_batch_size = 16 
model_args.eval_batch_size = 16
model_args.num_train_epochs = 1
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 30000
model_args.use_multiprocessing = False
model_args.fp16 = False
model_args.save_steps = 10000
model_args.save_eval_checkpoints = False
model_args.no_cache = True
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.preprocess_inputs = False
model_args.num_return_sequences = 1

model_name = "t5-small"
model_args.output_dir = model_name
#model_args.wandb_project = "mt5"

#model = T5Model("mt5", "google/mt5-small", args=model_args)
model = T5Model("t5", model_name, args=model_args)
model.train_model(train_df, eval_data=eval_df)

# results = model.eval_model(eval_df, verbose=True)


