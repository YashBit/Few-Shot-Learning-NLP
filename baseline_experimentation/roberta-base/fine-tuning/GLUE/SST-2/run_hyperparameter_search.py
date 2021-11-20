"""Run a hyperparameter search on a RoBERTa model fine-tuned on BoolQ.
Example usage:
    python run_hyperparameter_search.py BoolQ/
"""
import argparse
import boolq
import data_utils
import finetuning_utils
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast

"""
    You can use other objective functions than the eval loss
"""
parser = argparse.ArgumentParser(
    description="Run a hyperparameter search for finetuning a RoBERTa model on the BoolQ dataset."
)
parser.add_argument(
    "data_dir",
    type=str,
    help="Directory containing the BoolQ dataset. Can be downloaded from https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip.",
)
args = parser.parse_args()
# Since the labels for the test set have not been released, we will use half of the
# validation dataset as our test dataset for the purposes of this assignment.
train_df = pd.read_json(f"{args.data_dir}/train.jsonl", lines=True, orient="records")
val_df, test_df = train_test_split(
    pd.read_json(f"{args.data_dir}/val.jsonl", lines=True, orient="records"),
    test_size=0.5,
)
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
train_data = boolq.BoolQDataset(train_df, tokenizer)
val_data = boolq.BoolQDataset(val_df, tokenizer)
test_data = boolq.BoolQDataset(test_df, tokenizer)

# Search algorithm


# Try with other search algorithms and see which one work the best for you. 


## TODO: Initialize a transformers.TrainingArguments object here for use in
## training and tuning the model. Consult the assignment handout for some
## sample hyperparameter values.
training_args = TrainingArguments(
    output_dir = '/scratch/yb1025/',
    evaluation_strategy = "epoch",
    num_train_epochs = 3,
    per_device_train_batch_size = 8, 
    # Put a dictionary in the range for the learning rate
)
# Search for atleast 5 trials

## TODO: Initialize a transformers.Trainer object and run a Bayesian
## hyperparameter search for at least 5 trials (but not too many) on the 
## learning rate. Hint: use the model_init() and
## compute_metrics() methods from finetuning_utils.py as arguments to
## trainer.hyperparameter_search(). Use the hp_space parameter to specify
## your hyperparameter search space. (Note that this parameter takes a function
## as its value.)
## Also print out the run ID, objective value,
## and hyperparameters of your best run.

#Initialising the model
trainer = Trainer(
    args = training_args,
    tokenizer = tokenizer,
    train_dataset = train_data,
    eval_dataset = val_data,
    model_init = finetuning_utils.model_init(),
    compute_metrics = finetuning_utils.compute_metrics(),
    evaluation_strategy="epoch"
)

"""
    RAYUNIFORM, UNIFORM RANGE OF VALUES 
    1E-5 - 5E-5
    HYPERPARAMTER SEARCH WILL CHANGE THE HYPER
    MISSING IN HYPERPARAMETER SEARCH:
    1. MANY ARGUMENTS ARE MISSING, check documentation 
"""


#Create a small lambda function. 

BestRun = trainer.hyperparameter_search(
    log_to_file =True,
    backend = "ray",
    #Dictionary which returns the key of 
    hp_space = lambda _: {"learning_rate": tune.uniform(1e-5, 5e-5)},
    search_alg=BayesOptSearch(mode="min"), 
    n_samples = 5, 
    compute_objective = lambda metrics : metrics["eval_loss"]
)
print(BestRun.run_id)
print(BestRun.objective)
print(BestRun.hyperparameters)

"""
RETURN OF THE HYPERPARAMETER ACCESS IT AS AN ATTRIBUTE . 
"""
"""
RUNNING IT ON GREENE: 
1. SCRATCH DIRECTORY
2. REQUEST JOb, already done by sbatch
3. sbatch run_hyperameter_search.slurm
"""


#For the best run: transformers.trainer_utils.BestRun