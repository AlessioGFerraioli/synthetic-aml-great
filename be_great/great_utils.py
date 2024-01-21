'''

from be-great==0.0.4
_convert_text_to_tabular_data_2 function was added, the rest is stock

edited version by alessio
2024.01.14 
on 2024.01.14 it was working 


'''

import typing as tp

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer


def _array_to_dataframe(data: tp.Union[pd.DataFrame, np.ndarray], columns=None) -> pd.DataFrame:
    """ Converts a Numpy Array to a Pandas DataFrame

    Args:
        data: Pandas DataFrame or Numpy NDArray
        columns: If data is a Numpy Array, columns needs to be a list of all column names

    Returns:
        Pandas DataFrame with the given data
    """
    if isinstance(data, pd.DataFrame):
        return data

    assert isinstance(data, np.ndarray), "Input needs to be a Pandas DataFrame or a Numpy NDArray"
    assert columns, "To convert the data into a Pandas DataFrame, a list of column names has to be given!"
    assert len(columns) == len(data[0]), \
        "%d column names are given, but array has %d columns!" % (len(columns), len(data[0]))

    return pd.DataFrame(data=data, columns=columns)


def _get_column_distribution(df: pd.DataFrame, col: str) -> tp.Union[list, dict]:
    """ Returns the distribution of a given column. If continuous, returns a list of all values.
        If categorical, returns a dictionary in form {"A": 0.6, "B": 0.4}

    Args:
        df: pandas DataFrame
        col: name of the column

    Returns:
        Distribution of the column
    """
    if df[col].dtype == "float":
        col_dist = df[col].to_list()
    else:
        col_dist = df[col].value_counts(1).to_dict()
    return col_dist


def _convert_tokens_to_text(tokens: tp.List[torch.Tensor], tokenizer: AutoTokenizer) -> tp.List[str]:
    """ Decodes the tokens back to strings

    Args:
        tokens: List of tokens to decode
        tokenizer: Tokenizer used for decoding

    Returns:
        List of decoded strings
    """
    # Convert tokens to text
    text_data = [tokenizer.decode(t) for t in tokens]

    # Clean text
    text_data = [d.replace("<|endoftext|>", "") for d in text_data]
    text_data = [d.replace("\n", " ") for d in text_data]
    text_data = [d.replace("\r", "") for d in text_data]

    return text_data


def _convert_text_to_tabular_data_2(text: tp.List[str], df_gen: pd.DataFrame) -> pd.DataFrame:
    """ Converts the sentences back to tabular data

    Args:
        text: List of the tabular data in text form
        df_gen: Pandas DataFrame where the tabular data is appended

    Returns:
        Pandas DataFrame with the tabular data from the text appended
    """
    # print(">>entered _convert_text_to_tabular_data")
    
    columns = df_gen.columns.to_list()
    # print(f"columns = {columns}")

    # Convert text to tabular data
    # print("entering for t in text:")
    for t in text:   # "t" is the compound sentence, so i'm navigating the list of compound sentences
        # print(f"t = {t}")
        features = t.split(",") # i divide the compound sentence "t" in elemental sentences, i call them "features" (later shortened to "f")
        #  td = dict.fromkeys(columns)  # create a dictionary in which the keys are the column names, the values are None GREAT ORIGINAL, see line below my version
        td = dict.fromkeys(columns, "0") # my version is to create instead of an empty dictionary, that will be filled with Nones except for the features
        # for which the tokens have created a sentence, I will create a dictionary of zeros, so for the features the tokens have not generated a sentence for
        # i will have a zero instead of a None
        
        # print("td = dict.fromkeys(columns)")
        # print(f"td = {td}")
        
        # Transform all features back to tabular data
        for f in features:  # "f" is the elemental sentence, so for example "status is 1"
            values = f.strip().split(" is ")  # i divide the elemental sentence: "status is 1" gets stored in "values" as values[0]="status", values[1]="1"
            # print('values = f.strip().split(" is ")')
            # print(f'values={values}')
            
            if values[0] in columns: #and not td[values[0]]: # (2024.01.13 18:45 forse è qui che controlla che tutti i values del df siano pieni?
                # print("entered: if values[0] in columns:")
                try:
                    td[values[0]] = [values[1]]
                    # print("td[values[0]] = [values[1]]")
                    # print(f"td[values[0]] = {td[values[0]]}")
                    # print(f"values[1] = {values[1]}")
                except IndexError:
                    #print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                    pass

        # print(f"td at the end of the loop on the elemental sentences of one compound sentence: {td}")
    
        df_gen = pd.concat([df_gen, pd.DataFrame(td)], ignore_index=True, axis=0)
        # print("df_gen = pd.concat([df_gen, pd.DataFrame(td)], ignore_index=True, axis=0)")
        # print(f"df_gen = {df_gen}")
    return df_gen

def _convert_text_to_tabular_data(text: tp.List[str], df_gen: pd.DataFrame) -> pd.DataFrame:
    """ Converts the sentences back to tabular data

    Args:
        text: List of the tabular data in text form
        df_gen: Pandas DataFrame where the tabular data is appended

    Returns:
        Pandas DataFrame with the tabular data from the text appended
    """
    columns = df_gen.columns.to_list()
        
    # Convert text to tabular data
    for t in text:
        features = t.split(",")
        td = dict.fromkeys(columns)  # list of column names
        # print("td = dict.fromkeys(columns)  # list of column names")
        # print(f"td = {td}")
        
        # Transform all features back to tabular data
        for f in features:
            values = f.strip().split(" is ")
            # print('values = f.strip().split(" is ")')
            # print(f'values={values}')
            
            if values[0] in columns: # and not td[values[0]]: # (2024.01.13 18:45 forse è qui che controlla che tutti i values del df siano pieni?
                try:
                    td[values[0]] = [values[1]]
                except IndexError:
                    #print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                    pass
                
        df_gen = pd.concat([df_gen, pd.DataFrame(td)], ignore_index=True, axis=0)
    return df_gen
