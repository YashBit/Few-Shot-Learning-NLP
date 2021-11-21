import data_utils #Get the file that you want to test
import pandas as pd
import torch
import unittest
import numpy as np
from transformers import RobertaTokenizerFast


"""
CHANGE ALL THE ASSERTIONS TO EXIST WITHOUT THE LOOP, RIGHT NOW LOOP IS ANNOYING, 
ASSERT CAN HANDLE LISTS AS WELL
"""


class TestDataUtils(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.dataset = pd.DataFrame.from_dict(
            {
                "sentence": ["sentence 0", "sentence 1"],
                "idx": [0, 1],
                "label": [0, 1],
            }
        )
        # self.max_seq_len = 4

    def test_sample(self):
        ## An example of a basic unit test, using class variables initialized in
        ## setUpClass().
        # self.assertEqual(self.max_seq_len, 4)
        pass

    def test_encode_data(self):
        ## TODO: Write a unit test that asserts that the dimensions and dtype of the
        ## output of encode_data() are correct.
        ## input_ids should have shape [len(self.dataset), self.max_seq_len] and type torch.long.
        ## attention_mask should have the same shape and type.
        idTensor, attentionTensor = data_utils.encode_data(self.dataset, self.tokenizer)

        # self.assertEqual(list(idTensor.shape),[len(self.dataset), self.max_seq_len])
        # self.assertEqual(list(attentionTensor.shape), [len(self.dataset), self.max_seq_len])
        """
            Only the type fails, not torch.long, but int instead. Solve, for part 1 to be completed. 
        """
        self.assertEqual(idTensor[0].dtype, torch.long)
        self.assertEqual(attentionTensor[0].dtype, torch.long)

    def test_extract_labels(self):
        ## TODO: Write a unit test that asserts that extract_labels() outputs the
        ## correct labels, e.g. [1, 0].
        result = data_utils.extract_labels(self.dataset)
        index = []
        for count in range(len(result)):
            if result[count] == 1:
                self.assertEqual(1, self.dataset["label"][count])
            elif result[count] == 0:
                self.assertEqual(0, self.dataset["label"][count])

if __name__ == "__main__":
    unittest.main()