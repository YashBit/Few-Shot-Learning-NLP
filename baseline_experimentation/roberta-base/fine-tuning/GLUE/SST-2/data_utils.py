import torch
# Run on Google Collab/Notebook
def encode_data(dataset, tokenizer, max_seq_length=128):
      
    """Featurizes the dataset into input IDs and attention masks for input into a
     transformer-style model.
     NOTE: This method should featurize the entire dataset simultaneously,
     rather than row-by-row.
  Args:
    dataset: A Pandas dataframe containing the data to be encoded.
    tokenizer: A transformers.PreTrainedTokenizerFast object that is used to
      tokenize the data. 
      You can simply call this on the input for the dataset argument
    max_seq_length: Maximum sequence length to either pad or truncate every
      input example to.
  Returns:
    input_ids: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing token IDs for the data.
    attention_mask: A PyTorch.Tensor (with dimensions [len(dataset), max_seq_length])
      containing attention masks for the data.
  """
    ## TODO: Tokenize the questions and passages using both truncation and padding.
    ## Use the tokenizer provided in the argument and see the code comments above for
    ## more details.


    tokenized = tokenizer(dataset['question'].tolist(), dataset['passage'].tolist(), truncation = True, padding = "max_length", max_length = max_seq_length)
    ids = tokenized.get('input_ids')
    mask_lst = tokenized.get('attention_mask')
    input_tensor = torch.LongTensor(ids)
    attention_tensor = torch.LongTensor(mask_lst)  
    return input_tensor, attention_tensor



def extract_labels(dataset):
    """Converts labels into numerical labels.
  Args:
    dataset: A Pandas dataframe containing the labels in the column 'label'.
  Returns:
    labels: A list of integers corresponding to the labels for each example,
      where 0 is False and 1 is True.
  """
    ## TODO: Convert the labels to a numeric format and return as a list.
    bool_list = []
    for index in dataset["label"]:
      if index == True:
        bool_list.append(1)
      elif index == False:
        bool_list.append(0)
    return bool_list