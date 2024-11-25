# %% [markdown]
# # AML GReaT Synth

# %% [markdown]
# This notebook uses an edited version of be_great=0.0.4.
# 
# Refer to the readme file for any info on this notebook.

# %%
from be_great import GReaT
import pandas as pd

TRAIN_NO_DUPL = False # if True, remove duplicate rows from the training data
TRAIN_NO_STATUS = True # if True, remove "status" feature from the training data
CORRECT_STATUS = True # if True, assign "status" in post depending on "time"
CORRECT_ANOMALOUS_VALUES = True #if True, corrects anomalous values in mutation features in post

# %% [markdown]
# # Training

# %% [markdown]
# ### Load data

# %%
filename = 'data.csv'
data = pd.read_csv(filename)

# drop unnamed column (that is generated when reading)
data = data.drop(data.columns[data.columns.str.contains('unnamed', case=False)], axis=1)

# define which features not to use in training
features_to_ignore_in_training = ["patient_id", "tazi", "lda", "bayes", "kmeans", "umap", "kumap"]
if TRAIN_NO_STATUS == True:
    features_to_ignore_in_training.append("status")
train_data = data.drop(features_to_ignore_in_training, axis=1)

# %% [markdown]
# ### Initialize model

# %%
# parameters for the model
llm = "distilgpt2" # name of the large language model used (see HuggingFace for more options)
n_epochs = 25 # number of epochs to train
save_steps=2000 # Save model weights every x steps
experiment_dir="trainer" # name of the directory where all intermediate steps are saved
batch_size=32  # batch size

# initialize the model
great = GReaT(llm,                  
              epochs=n_epochs,                  
              save_steps=save_steps,               
             logging_steps=50,             # Log the loss and learning rate every x steps
              experiment_dir=experiment_dir, 
              batch_size=batch_size
             )

# %% [markdown]
# ### Training from scratch

# %%
trainer = great.fit(train_data, resume_from_checkpoint=False)

# %% [markdown]
# ### Training from a checkpoint

# %% [markdown]
# example of training from a checkpoint
# 
# Note that a checkpoint will work only if the parameters of the models and train_data are the same on what the model in the checkpoint saw (except for number of epochs)
# 

# %%
# checkpoint_path = "trainer/checkpoint-2000"
# trainer = great.fit(train_data, resume_from_checkpoint=checkpoint_path)

# %% [markdown]
# ### Save model

# %%
# In this example I generate the model name depending on the paramaters I used for the training

if TRAIN_NO_DUPL == True:
    nodupl_flag_name = "nodupl_" 
else:
    nodupl_flag_name = ""
    
if TRAIN_NO_STATUS == False:
    status_flag_name = "w-status_"
else:
    status_flag_name = ''
    
# save the model for future use
model_name = f"{n_epochs}_epochs_{status_flag_name}noclust_{nodupl_flag_name}{llm}"
great.save(model_name)

# %% [markdown]
# # Sampling

# %% [markdown]
# ### Load model

# %%
# load a previously trained model
# model_name = "250_epochs_noclust_distilgpt2"
# great = GReaT.load_from_dir(model_name)

# %% [markdown]
# ## Generate samples

# %%
# the model is ready to generate new samples in this way:
n_samples = 5
samples = great.sample(n_samples)

# %% [markdown]
# ## Sampling 10 synthetic AML datasets

# %%
def post_processing_AML_samples(samples, correct_status=False, correct_anomalous_values=False):
    
    '''
    This functions performs some post processing on synthetic AML datasets produced by GReaT.
    Converts the mutation variables into int.
    If correct_status is True, "status" feature is assigned depending on "time": if a row has time equal to the maximum value of time,
        sets "status" to 0, otherwise to 1.
    If correct_anomalous_values is True, each value of the mutations that is not equal to 0 nor 1 is set to 1. This is because the
        GReaT model will usually mix up similar values from different columns, or generating new values altogether; this results 
        in mutation variables sometimes assuming values slightly different from the admitted 0 or 1 (e.g.: 1.05 or 0.89). 
        This will set them always to 1 (instead of 0) based on two arguments: first, by design, the model is trained on just
        the present mutations, so it can be safe to assume that when it produces an output regarding a mutation it is signalling
        the presence of a mutation, not the absence of it; secondly, in the test made before writing this, the anomalous values
        encountered were always numbers close to 1 and never close to 0.

    inputs: 
        - samples (pandas.DataFrame) : the dataframe to be processed. It is expected to be a dataframe of synthetic AML data.
        - correct_status (bool, default=False) : if True, assign "status" variable from "time"
        - correct_anomalous_values (bool, default=False) : if True, correct anomalous values in the mutation features
    output:
        - samples (pandas.DataFrame) : the processed dataframe
    '''
 
    # all mutations names in the AML DATA
    mutations = ['ASXL1', 'ASXL2', 'ASXL3', 'ATRX', 'BAGE3', 'BCOR', 'BRAF', 'CBFB', 'CBL', 'CDKN2A', 'CEBPA_bi', 'CEBPA_mono', 'CNTN5', 'CREBBP', 'CSF1R',
                 'CSF3R', 'CTCF', 'CUL2', 'CUX1', 'DNMT3A', 'EED', 'ETV6', 'EZH2', 'FBXW7', 'ITD', 'FLT3_TKD', 'FLT3_other', 'GATA1', 'GATA2', 'GNAS',
                 'GNB1', 'IDH1', 'IDH2_p.R140', 'IDH2_p.R172', 'JAK2', 'JAK3', 'KANSL1', 'KDM6A', 'KIT', 'KMT2C', 'KMT2D', 'KMT2E', 'KRAS', 'LUC7L2',
                 'MED12', 'MLL', 'MPL', 'MYC', 'NF1', 'NFE2', 'NOTCH1', 'NPM1', 'NRAS_other', 'NRAS_p.G12_13', 'NRAS_p.Q61_62', 'PDS5B', 'PHF6', 'PPFIA2',
                 'PRPF8', 'PTEN', 'PTPN11', 'PTPRF', 'PTPRT', 'RAD21', 'RIT1', 'RUNX1', 'S100B', 'SETBP1', 'SF1', 'SF3B1', 'SMC1A', 'SMC3', 'SMG1', 'SPP1',
                 'SRSF2', 'STAG2', 'STAT5B', 'SUZ12', 'TET2', 'TP53', 'U2AF1', 'WT1', 'ZRSR2', 'add_8', 'add_11', 'add_13', 'add_21', 'add_22', 'del_20',
                 'del_3', 'del_5', 'del_7', 'del_9', 'del_12', 'del_13', 'del_16', 'del_17', 'del_18', 'minusy', 't_v_11', 't_10_21', 't_12_13', 't_12_17',
                 't_12_22', 't_13_19', 't_15_16', 't_15_17', 't_16_17', 't_16_21', 't_17_19', 't_17_21', 't_1_12', 't_1_14', 't_1_16', 't_1_17', 't_1_19',
                 't_1_3', 't_1_4', 't_1_5', 't_1_6', 't_2_17', 't_2_3', 't_2_5', 't_2_7', 't_2_9', 't_3_16', 't_3_21', 't_3_5', 't_3_7', 't_3_9', 't_4_12',
                 't_4_21', 't_4_9', 't_5_12', 't_5_17', 't_5_9', 't_6_9', 't_7_16', 't_7_17', 't_7_8', 't_8_10', 't_8_13', 't_8_16', 't_8_17', 't_8_21',
                 't_9_11', 't_9_13', 't_9_17', 't_9_22', 'complex', 'others_transloc', 'inv_3', 'inv_16']

     if correct_status == True:
        # manually correct "status" variable, which should be completely determined by "time"
        # specify the error threshold of time being equal to time_max
        error_threshold = 10e-3
        condition = abs(samples['time'] - samples['time'].max()) < error_threshold
        # change values in column 'A' based on the condition
        samples.loc[condition, 'status'] = 0
        samples.loc[~condition, 'status'] = 1
        
    if correct_anomalous_values == True:
        # correct anomlaous values
        for index in samples.index:
            for mutation in mutations:
                if samples.loc[index, mutation] not in [1.0, 0.0, 1, 0, "1", "0"]: 
                    samples.loc[index, mutation] = 1
    
    # make sure the mutations variables are of type int
    samples[mutations] = samples[mutations].astype('int')

        
    return samples.copy()

# %%
n_samples = 2017
n_runs = 10

for run in range(n_runs):
    # generate a synthetic dataframe
    samples = great.sample(n_samples)
    # post-processing 
    samples = post_processing_AML_samples(samples, 
                                correct_status=CORRECT_STATUS, 
                                correct_anomalous_values=CORRECT_ANOMALOUS_VALUES)
    # save the synthetic data to csv
    samples.to_csv(f"samples_{model_name}_run{run}.csv")


