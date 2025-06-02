# Pretrained Trajectory Embedding Project

## 🚀 `run.py` Script Documentation

This script trains and evaluates embedding models and downstream tasks on user trajectory datasets.


| Argument            | Type   | Default     | Description                                                                        |
| ------------------- | ------ | ----------- | ---------------------------------------------------------------------------------- |
| `--device`          | `int`  | `0`         | Index of the acceleration device to use (e.g., GPU ID).                            |
| `--init_param`      | `flag` | `False`     | If set, initializes model parameters from scratch.                                 |
| `--embed_name`      | `str`  | `'ctle'`    | Name of the trajectory embedding model (e.g., `ctle`, `rnn`, `transformer`).       |
| `--embed_size`      | `int`  | `128`       | Dimension of the embedding vectors.                                                |
| `--hidden_size`     | `int`  | `None`      | Hidden layer size for embedding or task model. If `None`, it may be auto-inferred. |
| `--embed_epoch`     | `int`  | `5`         | Number of training epochs for the embedding model.                                 |
| `--pre_model_name`  | `str`  | `'mc'`      | Name of the pretrained model to load (if applicable).                              |
| `--task_name`       | `str`  | `'loc_pre'` | Name of the downstream task (e.g., location prediction `loc_pre`).                 |
| `--task_epoch`      | `int`  | `2`         | Number of training epochs for the downstream task.                                 |
| `--task_batch_size` | `int`  | `64`        | Batch size for training the downstream task.                                       |
| `--pre_len`         | `int`  | `3`         | Length of input sequences for prediction tasks.                                    |
| `--dataset`         | `str`  | `'pek'`     | Name of the dataset to use (e.g., `'pek'` for Beijing dataset).                    |


Using Example : 

```python run.py --device 0 --embed_name ctle --task_name loc_pre --dataset pek --embed_epoch 10```

The code will do the embedding pretrain first , then use it for downstream model training.

## loading mechanism

For pre-trained results, we write detection to automatically load pre-trained embedding parameters if they exist the same one.

## Datasets

The Dataset class provides a structured interface for preprocessing and generating user trajectory sequences from mobile signaling data. It supports user and location indexing, coordinate mapping, temporal segmentation, and optional motion feature extraction.

## Dataset Classes initial: 

Only for the traditional trajectory datasets, like pek.h5 , taxi.h5

### Parameters:

1. raw_df: DataFrame containing raw signaling data. Must include: user_id: Unique identifier for users. poi_id: Location identifier (used as latlng). datetime: Timestamps of visits.

2. coor_df: DataFrame with geospatial information. Indexed by poi_id, containing: lat: Latitude , lng: Longitude.

3. split_days: Dictionary or list mapping day indices (e.g., 'train', 'val', 'test') to actual day values.

### Key Attributes:

> df: Processed DataFrame including user/location indices and coordinates.
> num_user: Number of unique users.
> num_loc: Number of unique locations.

## Data Generation

Generate trajectory sequences grouped by user and day.

### Parameters:

1. min_len: Minimum sequence length to keep a user-day trajectory.

2. select_days: Key/index used to filter specific days from split_days. Use None for all. Aiming for distinguish different datasets

3. include_delta: If True, appends time and distance deltas as well as raw coordinates.

## Downstream

This module contains a diverse set of neural architectures tailored for next-location prediction and trajectory modeling. These models are designed to work on top of pretrained embeddings (e.g., from CTLE, Hier, etc.) and learn to predict future mobility patterns based on historical data.


| Class                                    | Description                                                                                           |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `RnnLocPredictor`                        | Abstract base class for RNN-based predictors.                                                         |
| `StrnnLocPredictor`                      | Spatial-Temporal RNN location predictor, captures both spatial and temporal intervals.                |
| `STRNN` / `STRNNCell`                    | Core modules implementing the STRNN structure.                                                        |
| `StlstmLocPredictor`                     | Spatial-Temporal LSTM for fine-grained trajectory modeling.                                           |
| `ErppLocPredictor`                       | Based on **E**vent-driven **R**ecurrent **P**oint **P**rocess, suitable for continuous-time modeling. |
| `DecoderPredictor` / `DecoderPredictor2` | Sequence-to-sequence decoders, used in Transformer-based forecasting.                                 |
| `TransformerPredictor`                   | Utilizes self-attention to model global dependencies in trajectories.                                 |
| `Seq2SeqLocPredictor`                    | Classic encoder-decoder RNN model for sequential prediction.                                          |
| `MCLocPredictor`                         | Lightweight memory-augmented predictor for fast and robust inference.                                 |

| Function                      | Purpose                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------ |
| `fourier_locpred(...)`        | Predict future locations using Fourier-based encoding and learned spatial representations. |
| `loc_prediction(...)`         | General interface for training and evaluating location prediction models.                  |
| `mc_next_loc_prediction(...)` | Fast inference using the memory-compressed `MCLocPredictor`.                               |


## Embed

To save the space, we omit the detailed introduction of each classes initializaion.

## all available embedding

1. CTLE (Contrastive Temporal Location Embedding)

Module: embed.ctle

Core Classes:

CTLE, CTLEEmbedding: Main encoder capturing spatiotemporal dynamics.

PositionalEncoding, TemporalEncoding: Time and position-aware encodings.

MaskedLM, MaskedHour: Pretraining heads using masked prediction objectives.

Training: train_ctle

2. Hier (Hierarchical Embedding)

Module: embed.hier

Core Classes:

HierEmbedding: Embeds data using hierarchical context (e.g., user-day-hour).

Hier: Main model architecture.

Training: train_hier

3. Teaser (Temporal Embedding and Sequence Alignment Representation)

Module: embed.teaser

Core Classes:

TeaserData: Sequence construction tailored for TEASER.

Teaser: Embedding model emphasizing temporal order.

Training: train_teaser

4. Fourier Encoding

Module: embed.fourier

Core Classes:

FourierEncoding_IM: Uses fourier form for latitude and longititude encoding.

Masked_GC / Masked_LM :

Training: train_fourier

