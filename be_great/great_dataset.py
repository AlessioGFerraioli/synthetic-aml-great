'''

originally from be-great==0.0.4
edited version by Alessio Giuseppe Ferraioli
2024.01.14 last edit 


summary of the edits to the original:
- new version of GReaTDataset._getitem

see readme.md for detailed info


'''

import random
import typing as tp

from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding


class GReaTDataset(Dataset):
    """ GReaT Dataset

    The GReaTDataset overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer from HuggingFace
    """
    def set_tokenizer(self, tokenizer):
        """ Set the Tokenizer

        Args:
            tokenizer: Tokenizer from HuggingFace
        """
        self.tokenizer = tokenizer

    def _getitem(self, key: tp.Union[int, slice, str], decoded: bool = True, **kwargs) -> tp.Union[tp.Dict, tp.List]:
        """ Get Item from Tabular Data

        Get one instance of the tabular data, permuted, converted to text and tokenized.
        If the value of a feature is equal to 0, that feature is ignored and not passed to the tokenizer.

        """
        # If int, what else? 
        row = self._data.fast_slice(key, 1)
        
          
        shuffle_idx = list(range(row.num_columns))
        random.shuffle(shuffle_idx)
        
        bad_word = "0"
        
        text_to_join = []
        for i in shuffle_idx: 
            text = "%s is %s" % (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip())
            if str(row.columns[i].to_pylist()[0]).strip() != bad_word:
                text_to_join.append(text)
        
        shuffled_text = ", ".join(text_to_join)
        tokenized_text = self.tokenizer(shuffled_text)
        return tokenized_text

    
    def __getitems__(self, keys: tp.Union[int, slice, str, list]):
        if isinstance(keys, list):
            return [self._getitem(key) for key in keys]
        else:
            return self._getitem(keys)

@dataclass
class GReaTDataCollator(DataCollatorWithPadding):
    """ GReaT Data Collator

    Overwrites the DataCollatorWithPadding to also pad the labels and not only the input_ids
    """
    def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch
