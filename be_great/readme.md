This folder contains the scripts from be_great with edits by me to work on AML data.
In order to use them, install [be_great=0.0.4 as explained by the original authors](https://github.com/kathrinse/be_great/tree/main) and replace the original the original .py files with the ones of the same name that are in this folder.


# My Edits and the thought process behind them

This file explains the edits to be_great in order to use it to model the AML dataset.

## Introduction

### The need for edits

The original be_great=0.0.4 package is not suitable to model data such as the AML dataset. 

The principle of be_great is to convert the features of the training data into meaningful sentences, string them together in one long compound sentence for each row and pass them to a large language model ("LLM"), that will learn them in order to generate new sentences.

The AML dataset contains a moderately large number of features (~150) that will be coverted into very long compound sentences;  this causes problems with the training process.  

The Transformer model used by be_great for encoding the sentences into tokens is not designed to work with very long sentences and might raise an error and not work. Even if the Transformer is able to encode the sentences, the amount of tokens generated would be terribly large, rendering the training process prohibitively slow.

Moreover, since in the AML dataset for each row just a handful of mutations are present, most of the compound sentneces would be non-informative and we would be training on a lot of redundant sentences.


<ins>Disclaimer</ins>: I know this uses some patchy techniques and it is not written in a general nor efficient way. The point here was just to make a working prototype, something that would work in my specific case.

I suppose it would not be so difficult to finesse it in order to have it work more generally. However, this should still work for any dataset in which in which the ```0```s mean “no event”.

## Training  
In spite of the title, I did not edited directly the functions regarding the training. 

### Original Workflow
Originally, the training was done on compound sentences, each containing an elemental sentence for each and every column of the data. 

To prepare the data for training (i.e.: creating the compound sentences), the original ```fit``` function converts it into a sligthly modified HugginFace Dataset object. This modified version of the Dataset object is defined in ```great_dataset.py``` and the most important difference is that it contains a custom ```_getitem``` function that overrides the default ```_getitem``` function of the HugginFace Dataset.
The purpose of the custom ```_getitem``` function is to include the permutation step by shuffling the order of the elemental sentences when constructing the compound sentence.

### My Workflow and edits
In my case, I want to create compound sentences with elemental sentences for only the columns of the data in which there is a 1, meaning a “mutation”, or in general an “event”. 
To achieve this, I only need to modify the ```_getitem``` function, by hard-coding a “bad word”, i.e.: a word that if found in the feature value makes the function exclude that feature from the compound sentence. 

In other words, setting the ```bad_word = 0``` means that if a feature value is ```0```, that feature will be excluded by the compound sentence. This is done by a simple if statement that checks if the value is different to the bad-word, and if yes, it appends the corresponding elemental sentence to the compound sentence, else it simply ignores the feature.

In this way, the compound sentences will be made up of of just features that do not take the value ```0```. 

<ins>Disclaimer</ins>: I know this is a patchy way of doing this, but again, I’m justing writing it to work in my specific case; it could be expanded and generalized.
I also know that it would be a hundred times better to not hard code the bad word, but having it as a parameter to pass to the fit function, so it could be specified at runtime, and also if no bad word is passed the fit function would behave as usual. Taking into consideration the fact that a model trained in this way will also need a special function for sampling (```sample2```, see “Sampling”), probably a better solution would be to create a variation for the GReaT class altogether, so that when initializing the model, you could choose whether or not to pass a bad word, and the sample function would work accordingly. 

## Sampling 
### Original Workflow
The original workflow can be summed up as follows:
- initiate an empty dataframe (```df_gen```), with the features as column names
- generate a batch of tokens, convert them into text → we get a list of many compound sentences in the structure “feature4 is x, feature7 is a, feature2 is b”. Note that in my case these sentences include just a very small subset of features. This is due to the fact that I trained the model on short compound setences that included just a small subset of features (only the ones where an event happened i.e. where there was a ```1```) 
- for each compound sentence (labeled ```t``` in the code) initiate an empty dictionary (```td```), with the column names as keys.
- split the compound sentences in elemental sentences (labeled ```f```) of the type “feature4 is x” and then for each elemental sentence save the value of the feature in ```td```, e.g.: “feature4 is x” → ```td[“feature4”] = x for any f```.
- add ```td``` as a row at the bottom of ```df_gen```; in this way, we build the empty dataframe df_gen one row at a time
- when the batch of tokens (= list of compound sentences) is exhausted, before generating a new batch and continuing until we have enough samples (=rows in ```df_gen```), some checks are performed: remove all rows that contain at least one ```None``` or ```Nan```; convert numerical features to ```float```.
- the ```already_generated``` variable, which counts the number of samples generated, gets updated with the number of rows of ```df_gen```; if the number of samples we want to generate is larger than the ones that we have already generated, a new batch of token is produced and the process is repeated until we have enough samples.

### My workflow and edits
The original workflow was based on this simple core idea: the compound sentences will contain the values for all the features and so any value not specified by the compound sentence is treated as a missing value, so as an error. For this reason, variables to collect the generated data are initialized in the default state of being empty, ```None```, and are expected to be completely filled. Each generated row that has not been completely filled is considered an error and so it is deleted. 

In my case, I do not expect the compound sentences to contain all the features, because I trained the model on compound sentences that only contained a small subset of features. For this reason, I will consider the default (better to say “not-specified”) state to be ```0``` instead of ```None```/error. 

The main difference between my code and the original is that I will initialize ```td``` (the dictionary in which I store the values for a given compound sentence, that would later become a row in the dataframe) as all zeros, instead of empty. This will make sure that any feature that is not contained in the compound sentence will be considered as ```0```, meaning “no event”/”no mutation”. 
The only other difference is very marginal: I remove the check that converts numerical data to float, because I want my ```0```s to be treated as strings. This is not a huge deal, since I could simply change their type later, after the sampling process.
I reiterate that these edits to the code are written in order to work on my data, maybe in future I will adapt it to be more general and useful. 


# EEEEEEEEEEEEEEE
### Change everything in just one "sample" method instead of two
All the edits to the sampling process are made by creating ```GReaT.sample2```, a new sampling method of the class ```GReaT``` in ```great.py``` to use instead of ```GReaT.sample```. Another edit is done on the function _convert_text_to_tabular_data in great_utils.py, that gets called by sample and sample2. To mantain the original functionality too, a new function was created, named _convert_text_to_tabular_data_2, that would be called by sample2. 	
