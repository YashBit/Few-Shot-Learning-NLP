from transformers import RobertaForSequenceClassification
from datasets import load_metric
metric = load_metric('glue', "sst2")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    
    infoDict{
        'accuracy' :  metric.compute(predictions=predictions, references=labels)
    }
    return infoDict


# def compute_metrics(eval_pred):
#     """Computes accuracy, f1, precision, and recall from a 
#     transformers.trainer_utils.EvalPrediction object.
#     """
   

#     ## TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
#     ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.
    
#     logits = eval_pred.predictions.argmax(-1)
#     labels = eval_pred.label_ids
#     infoDict = {
#         'accuracy' : sm.accuracy_score(labels, logits),
#     }
#     return infoDict


def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    ## TODO: Return a pretrained RoBERTa model for sequence classification.
    ## See https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification.
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    model = model.to('cuda')
    return model