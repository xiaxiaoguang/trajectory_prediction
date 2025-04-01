# Task description

Target : Given a trajectory, whose shapes is $(lon,lat,time)$, predict the next location or the time costs to move to the next location.

The current experiment are : Given $(lon,lat)$ of starting points and destination points, predict the time costs, and which is actually for finding a better embedding.

# Code works

Data Preprocessing is too slow, I write a faster version with torch functions that can accelerated by C++.

For training code, because we need reproducing the results. I've not change it too much.

I can build a more comfortable training system, but changing it blindly without further ideas are in vain.

# Embedding Methods

1. Vanilla Fourier Embedding is an effective methods 

(At least for the prediction results, fourier embedding is great)

However the prediction results are not reflect the best embedding righteously. 

The biggest worries are the embedding may not suited for large language models, since the embedding size are not compatible.

In this phases we cannot trully evalute the qualities of embedding

2. Benchmark

Other methods in providing code are too simple. They don't considering the size of the datasets and the difficulty of the 

If there are related papers off the shelf , providing practible methods for travelling time prediction and benchmark, that will be better.

2. Combining with GNN

I think it would be better if the embedding are extracted from Graph directly.

Since the latitude and longitude reflects the location information, That's not contrast to the graph nodes embedding.

# Trajectory Prediction Papers (Travel Time / Distance)

"Pre-training Context and Time Aware Location Embeddings from Spatial-Temporal Trajectories for User Next Location Prediction" is a similar works. Though its predict next location, the method of their 

The relative visited time differences may attribute the position special functonalities. Instead of assigning one latent vector as each location’s final representation, a target location’s embedding vector is calculated by a parameterized mapping function of its contextual locations’ encoding vectors.

In their implementation, they do the location embedding first to form a time series (L, D), then use transformer solve the time series. In their descriptions , the transformer will learn to understand correlation between target locations and their corresponding contexts and improve the quality of the embedding.

They use Masked language modeling as their loss, that's a classical designation.

And to incoorporate the temporal information , they use masked hour loss which is predicted the time cost via the trajectory. That's actually similar as our task targets.


# Training with small datasets (First 5000 trajectory Only 100000 datapoints)

All the results below under 5 times experiments.

model|MAE|MSE|RMSE|
|-|-|-|-|
PositionalEncoding|1.2026|0.9007|0.9007
PositionalEncodingnew|1.7328|1.1032|1.1032
FourierEncoding|1.0597|0.8051|0.8051
SinEncoding|1.8899|1.1328|1.3601
FourierMLPEncoding|1.4096|0.9677|1.1787
LSTMBasedEncoder|1.4683|0.9492|0.9492
CNNBasedEncoder|1.2553|0.8658|0.8658
TransformerEncoder|1.4853|1.0050|1.0050
SinEncoding_dm32|1.4164|0.9632|1.1811
FourierEncoding_IM_dm32|0.6590|0.6068|0.7990
TransformerEncoder_dm32|1.4426|0.9758|1.1923

# Training with Large datasets (Full trajectory First 5000000 datapoints)

Training data 0.6

validation data 0.3

Testing data 0.05 

Single Experiment Result:

model|MAE|MSE|RMSE|
|-|-|-|-|
FourierEncoding_IM_dm64|0.4051|0.3592|0.6363
FourierEncoding_IM_dm128|0.3811|0.3417|0.6171


# Training with Final datasets (Full trajectory Full 13190038 datapoints)

Single Experiment Result:

model|MAE|MSE|RMSE|
|-|-|-|-|
FourierEncoding_IM_dm128|0.4085|0.3417|0.6389
FourierEncoding_IM_dm256|0.2682|0.3142|0.5168
FourierEncoding_IM_dm2048|0.0169|0.0548|0.1219

Here we suspect the dataset contain too many similar points pair , so they appear in both training dataset and test dataset. The data preprocess is still improve. But also there's no doubt that models had already learned how to generalize in this kind of data.

# Data Processing

After unique with distance , we get 11 millions datas, and observe it we find:

```json
torch.Size([11543987, 5])

[ 116.1000,   39.6824,  116.1079,   39.6947, 1394.5730],
[ 116.1000,   39.6824,  116.1079,   39.6947, 1794.8747],
[ 116.1000,   39.6824,  116.1079,   39.6947, 2564.3767],
[ 116.1000,   39.6824,  116.1079,   39.6947, 2964.6784],
[ 116.1000,   39.6824,  116.1079,   39.6947, 3379.3587],
[ 116.1000,   39.6824,  116.1079,   39.6947, 3384.7625],
[ 116.1000,   39.6824,  116.1079,   39.6947, 3734.1803],
[ 116.1000,   39.6824,  116.1079,   39.6947, 4154.2644],
[ 116.1000,   39.6824,  116.1079,   39.6947, 4549.1623],
[ 116.1000,   39.6824,  116.1079,   39.6947, 4969.2464],
```

There are huge numbers of same location pairs, while the costing times are greatly different.

Since if we only unique with first 4 dimension ,we will get 2.2 millions datas, that's a great gap.

# Training with New datasets (Full trajectory,Full 11543987 datapoints,W/O completely same datas)

Single Experiment Result:

model|MAE|MSE|RMSE|
|-|-|-|-|
FourierEncoding_IM_dm256|0.1949|0.2610|0.4402
FourierEncoding_IM_dm2048|0.2561|0.3317|0.5055

In 2048 dmodel, results is actually overfitting. If we use that for the llm, it will influence the performance.
