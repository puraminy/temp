
import logging
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_df = pd.read_csv("data/train.tsv", sep="\t").astype(str)
eval_df = pd.read_csv("data/eval.tsv", sep="\t").astype(str)

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
model_args.do_sample = True
model_args.num_beams = 5
model_args.top_k = 5
model_args.top_p = 0.3
output_dir = "outputs_t5_small_full_2020" 
model_args.output_dir = output_dir

model = T5Model("t5", "t5-small", args=model_args)
model.train_model(train_df, eval_data=eval_df)

# results = model.eval_model(eval_df, verbose=True)


