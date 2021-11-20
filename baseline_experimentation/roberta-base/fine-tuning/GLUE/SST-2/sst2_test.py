import pandas as pd
import torch
import unittest

from sst2 import SST2Dataset
from transformers import RobertaTokenizerFast


class TestSST2Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.dataset = pd.DataFrame.from_dict(
            {
                "sentence": ["question 0", "question 1"],
                "idx": [0, 1],
                "label": [0, 1],
            }
        )
        self.max_seq_len = 4
        self.boolq_dataset = SST2Dataset(
            self.dataset, self.tokenizer, self.max_seq_len
        )

    def test_len(self):
        ## TODO: Test that the length of self.boolq_dataset is correct.
        ## len(self.boolq_dataset) should equal len(self.dataset).
        self.assertEqual(len(self.boolq_dataset), len(self.dataset))

    def test_item(self):
        ## TODO: Test that, for each element of self.boolq_dataset, 
        ## the output of __getitem__ (accessible via self.boolq_dataset[idx])
        ## has the correct keys, value dimensions, and value types.
        ## Each item should have keys ["input_ids", "attention_mask", "labels"].
        ## The input_ids and attention_mask values should both have length self.max_seq_len
        ## and type torch.long. The labels value should be a single numeric value.
        labelList = list(self.dataset["label"])
        for items in range(len(self.boolq_dataset)):
            listKeys = list(self.boolq_dataset[items].keys())
            #keys
            self.assertEqual(listKeys[0], "input_ids")
            self.assertEqual(listKeys[1], "attention_mask")
            self.assertEqual(listKeys[2], "labels")
            #lengths
            self.assertEqual(len(self.boolq_dataset[items]["input_ids"]), self.max_seq_len)
            self.assertEqual(len(self.boolq_dataset[items]["attention_mask"]), self.max_seq_len)
            #types
            self.assertEqual(self.boolq_dataset[items]["input_ids"][0].dtype, torch.long)
            self.assertEqual(self.boolq_dataset[items]["attention_mask"][0].dtype, torch.long)
            #labels value

            #If statement with the actual values in the label column
            if labelList[items] == True:
                self.assertEqual(self.boolq_dataset[items]["labels"], 1)  
            elif labelList[items] == False:
                self.assertEqual(self.boolq_dataset[items]["labels"], 0) 
           

            

if __name__ == "__main__":
    unittest.main()