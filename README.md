# synthetic-aml-great
Generation and evaluation of synthetic AML patients data with a modified version of [@kathrinse/be-great](https://github.com/kathrinse/be_great/tree/main)

# Installation

In order to use the edited version of be-great present in this repo, install [be_great=0.0.4 as explained by the original authors](https://github.com/kathrinse/be_great/tree/main), then replace the installed .py files with the corresponding ones in this repo.

# Structure of the repo

## be_great 

These are the .py files that should replace the ones of the same name in be_great=0.0.4.

## synthesis

These are the scripts/notebook to train a modified great model and sample from it.

### Advices for the training:

- __not many epochs__: the model learns very quickly and it is prone to overfitting. As a rule of thumb, training for more than ~250 epochs is usually too much
- __conditioning feature__: the model is trained and samples starting from a specific feature, of which it reproduces the distribution exactly. In general, it works best if this feature has a balanced distribution (for example, a bad choice would be conditioning on a feature that is 90% of the time a value and 10% of the time the other(s) value(s)). For the AML data, I found that conditioning on "time" works best
- __if a features is uninformative, remove it__: the model is able to hold and make sense of the information of a large number of features linked in complex ways, so expecting it to learn everything would still produce somewhat meaningful results. However, if a feature is uninformative (for example, the patient id, or "status" if we assume it is determined by time), removing it allows the model to focus more on what it is important, producing sligthly better results
- __using words instead of abstract values might improve learning__: great is based on pre-trained large language models that learn the data after it has been translated in meaningful english sentences. Since these models a prior understanding of the meaning of the words, translating the abstract values into meaningful words might help the training. For examples, on tests on the "Titanic" dataset, replacing "0/1" with "not survived/survived" appeared to enhance learning. 
- __consider removing duplicates in the training__: if there are duplicate rows in the training data removing them or not before training might produce a significant difference in the production. First of all, if a row is repeated multiple times in the training data it will have more impact in the learning process. Secondly, the model seems to behave differently if there are duplicates in the real data or not, even if it was just one duplicate row versus no duplicate rows. Also, having less duplicate rows in the synthetic data is not inherently better or worse: if our aim is to create a realistic replica of a synthetic dataset that has lots of duplicates (as in the AML data), reproducing the distribution might mean also reproducing many instances of the same row. So, removing duplicates should be considered depending on the specific aim of the model.

## analysis






