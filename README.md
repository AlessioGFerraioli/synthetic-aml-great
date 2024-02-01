# synthetic-aml-great
Generation and evaluation of synthetic AML patients data with a modified version of [@kathrinse/be-great](https://github.com/kathrinse/be_great/tree/main)

# Installation

In order to use the edited version of be-great present in this repo, install [be_great=0.0.4 as explained by the original authors](https://github.com/kathrinse/be_great/tree/main), then replace the installed .py files with the corresponding ones in this repo.

# Structure of the repo

## be_great/ 

These are the .py files that should replace the ones of the same name in be_great=0.0.4. Refer to the dedicated readme file in the folder for info.

## great_synth.ipynb

This is an example notebook to train a modified great model and sample from it.

### Advices for the training

These are general advices deducted from lots of tests on various datasets; any advice is just a rule of thumbs and should not be treated as gospel.

- __not many epochs__: the model learns very quickly and it is prone to overfitting. As a rule of thumb, training for more than ~250 epochs is usually too much
- __conditioning feature__: the model is trained and samples starting from a specific feature, of which it reproduces the distribution exactly. In general, it works best if this feature has a balanced distribution (for example, a bad choice would be conditioning on a feature that is 90% of the time a value and 10% of the time the other(s) value(s)). For the AML data, I found that conditioning on "time" works best
- __if a features is uninformative, remove it__: the model is able to hold and make sense of the information of a large number of features linked in complex ways, so expecting it to learn everything would still produce somewhat meaningful results. However, if a feature is uninformative (for example, the patient id, or "status" if we assume it is determined by time), removing it allows the model to focus more on what it is important, producing sligthly better results
- __using words instead of abstract values might improve learning__: great is based on pre-trained large language models that learn the data after it has been translated in meaningful english sentences. Since these models a prior understanding of the meaning of the words, translating the abstract values into meaningful words might help the training. For examples, on tests on the "Titanic" dataset, replacing "0/1" with "not survived/survived" appeared to enhance learning. 
- __consider removing duplicates in the training__: if there are duplicate rows in the training data removing them or not before training might produce a significant difference in the production. First of all, if a row is repeated multiple times in the training data it will have more impact in the learning process. Secondly, the model seems to behave differently if there are duplicates in the real data or not, even if it was just one duplicate row versus no duplicate rows. Also, having less duplicate rows in the synthetic data is not inherently better or worse: if our aim is to create a realistic replica of a synthetic dataset that has lots of duplicates (as in the AML data), reproducing the distribution might mean also reproducing many instances of the same row. So, removing duplicates should be considered depending on the specific aim of the model.
- __distilgpt2 works fine as a llm, but there might be better ones__: changing the large language model (llm) used might drastically change the results. A few of them were tested and distilgpt2 was found to be the best performing one. However, on the HuggingFace platform there are a lot of models of various type, some open source, and new get added everyday (at the time of writing, there are 477,241 models on HuggingFace, of which over 160k deal with natural language process). For this reason, consider experimenting with other models.


### Post-processing of the generated samples:

After the sampling of synthetic data from the model, an additional post-processing is advised and included in the notebook. The post-processing consists in two steps: manually assigning ```status``` depending on ```time``` and correcting anomalous values eventually present in the mutation features.

__Assigning ```status``` from ```time```__ comes from the assumption that __```status``` is completely determined by ```time```__: we assume that a patient whose ```time``` is equal to the maximum (the maximum value that can be found in the time column) will have ```status=0``` (no event), otherwise ```status=1``` (event). This is not true in general, and we can easily think of cases in which this is not true. For example, a patient could have ```status=0``` and ```time<time_max``` because the communications with that patient ended before the other patients. Or, a patient could have ```status=1``` and ```time=time_max``` if an event happens to be detected at ```time=time_max```. Moreover, this is based on the assumption that the last communications with each ```status=0``` patient happened at the same time, at ```time=time_max```. 
These assumptions limit the generalization of this post-processing, but we can think them as not so far from truth. In a usual data collection scenario, these assumptions hold most of the time; the data entries for which they do not hold are usually a stark minority and could be ignored, treated them as spurious data, without hindering the analysis. On top of that, these assumptions are exactly met in the Tazi et al. AML dataset, so we are safe to say that this post-processing is justified.


__Correcting anomalous values in the mutation features__ involves addressing all the synthetic values in the mutation features that are not 0 nor 1. The GReaT model sometimes will mix up similar values from different columns, or generating new values altogether; this results in mutation variables sometimes assuming values slightly different from the admitted ```0``` or ```1``` (e.g.: ```1.05``` or ```0.89```). In the post-processing, they will be set them to ```1``` in any case (instead of ```0```) based on two arguments: first, by design, the model is trained on just the present mutations, so it can be safe to assume that when it produces an output regarding a mutation it is signalling the presence of a mutation, not the absence of it; secondly, in the tests made before writing this, the anomalous values encountered were always numbers close to 1 and never close to 0.


## synth_analysis.ipynb

This notebook contains some functions to perform some integrity and quality analysis on synthetic data.
__Note__: it is written to do specific analysis on the data I have, so it is not always written in a general or modular way. It is also bound to this specific data, since for simplicity I hard coded some specifi of this dataset (for example, names of features to drop, number of rows etc..). Anyway it should still work on any synthetic AML dataset. Everything could be written in a better way; most of all, the data structures could also be improved a lot. It could also be useful in taking the functions scattered throughout the notebook and organize them, maybe as methods of a couple of well planned classes. This notebook was written little by little and starting with no real plan so there was no planning whatsoever. Good luck. (well, they DO work, at least, so it's not completely useless)