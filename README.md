# Synergy — CSC RAS
*СИНЕРГИЯ — ФИЦ "ИУ" РАН*

## Description

This repo is used for human emotion recognition benchmarks on several datasets.

The [MultiBench](https://github.com/pliang279/MultiBench) repo is used for model building.

### Recognition models

All the unimodal models is targeted to recognise emotions based on audio features.
The tri-modal models is using all three modalities' features. 

The run files is listed below can be used to perform training tasks: 

#### Classification task

- Transformer fusion (tri-modal): `FusionClassification.py`
- Transformer unimodal: `TransformerClassification.py`

#### Regression task

- Transformer fusion (tri-modal): `FusionRegression.py`
- Transformer unimodal: `TransformerRegression.py`
- GRU unimodal: `GruRegression.py`
- LSTM unimodal: `LstmRegression.py`

### Datasets

The following datasets support is included in this repo:

#### From MultiBench repo

- CMU-MOSEI
- CMU-MOSI
- UR-FUNNY
- MUStARD

#### Other datasets

This dataset support is included in `custom_data.py`:

- RAMAS (emo2)

Other datasets that can be used in this repo, but is needed to be implemented as custom data:

- IEMOCAP
- MELD
- M3ED
- SEWA
- emoFBVP
- EU Emotion Stimulus
- MAHNOB-HCI
- NNIME
- RECOLA

### Usage

1. MultiBench repo sources root must be in `$PYTHONPATH`
2. Datasets root path can be configured in `DATA_PATH` constant of `commons.py` file.
You can add a custom dataset by expanding `custom_data.py` file for a proper data loading.
3. Any script that listed in [Recognition models](#recognition-models) can be used to run a task.
Dataset name must be specified in first argument.
4. `Test.ipynb` can be used to analyse training results.
