
# **Introduction**  

In recent years, the development of wireless technology and positioning sensor devices has made the analysis of trajectory data a prominent research focus. Trajectory data plays a crucial role in various domains, including travel recommendation, urban planning, and traffic control, offering significant commercial and practical value. These data consist of temporally ordered location points, typically represented by latitude and longitude coordinates, and are widely used in spatial–temporal data mining. Among these applications, location prediction is a key task.  

The evolution of location prediction models has progressed from traditional Markov models to deep learning-based approaches. Within the deep learning framework, location prediction can be formulated as either a **classification task**, where the goal is to predict the most probable location from predefined regions, or a **regression problem**, where the aim is to estimate precise geographical coordinates. Regardless of the formulation, the primary challenge lies in capturing the underlying patterns of human mobility, which varies across both time and space. To address this challenge, an effective **location embedding** is essential—it must encode both the functional significance of location indices and the deep interrelations between different geographical coordinates.  

With the increasing volume of data, **pretraining methods** have gained significant importance. The most successful application of pretraining is in **large language models**, which leverage vast amounts of internet data for training, demonstrating the power of pretrained embeddings. Previous research, such as [Recurrent Marked Temporal Point Processes](https://www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf), explored the use of recurrent neural networks to model temporal events based on empirical mathematical processes. Another study, [Pre-training Context and Time-Aware Location Embeddings from Spatial-Temporal Trajectories](https://ieeexplore.ieee.org/abstract/document/9351627), introduced a transformer-based pretrained location embedding method that captures richer contextual information. However, this approach primarily focuses on location indices and lacks deeper exploration of geographical coordinates.  

Our project aims to develop a **novel embedding pretraining method** that integrates advanced time series analysis techniques for handling geographical coordinates. Additionally, we incorporate more generalized pretraining strategies inspired by **natural language processing (NLP)** and other fields. We have systematically built the codebase and benchmarked our approach on existing methods. A preliminary experiment using **Fourier embeddings**—which leverage geographical coordinates—has demonstrated promising results, highlighting the effectiveness of geographical coordinate-based embeddings. The following sections provide a detailed analysis of our experiments.  

Currently, our research focuses on:  

1. Proposing a new **pretrained trajectory embedding method** that better utilizes temporal information from time series techniques while capturing spatial relationships across different locations.  
2. Exploring **various downstream models** for trajectory prediction, aiming to improve upon existing approaches.  

---

# **Experiment Results**  

Here are our current experimental results. The training parameters and embedding size are kept consistent across all models to ensure a fair comparison. Our **pretrained Fourier embeddings** demonstrate superior effectiveness compared to models without pretrained embeddings. However, there is still a noticeable performance gap when compared to the **CTLE (Context and Time-Aware Location Embedding)** approach.  

## Next location prediction of Latitude and Longtitude

All training results on Taxi datasets

model_embed_epoch|MAE|MSE|RMAE|MAPE|
-|-|-|-|-|
erpp_hier_20|0.063179|0.009140|0.095489|44.014057
erpp_ctle_20|0.052859|0.007457|0.086176|54.512438
erpp_fourier_20|0.030500|0.003467|0.058885|31.916752
erpp_teaser_20|0.071256|0.009668|0.098263|69.409706
stlstm_hier_20|0.058938|0.009258|0.093707|54.386267
stlstm_ctle_20|0.062762|0.009420|0.096520|66.675620
stlstm_fourier_20|0.055630|0.005138|0.071681|46.482868
stlstm_teaser_20|0.062277|0.008861|0.094091|41.977295
rnn_hier_20|0.064366|0.008770|0.093603|57.133127
rnn_ctle_20|0.075242|0.010861|0.104059|69.067685
rnn_fourier_20|0.053598|0.006060|0.077844|56.229359
rnn_teaser_20|0.071827|0.010633|0.103117|67.465528
gru_hier_20|0.216644|0.074388|0.272681|154.564946
gru_ctle_20|0.220683|0.076758|0.276796|153.025940
gru_fourier_20|0.150191|0.036191|0.190240|102.530502
gru_teaser_20|0.228513|0.080099|0.282883|156.975866
decoder_hier_20|0.085219|0.012836|0.113290|74.814934
decoder_ctle_20|0.079447|0.011556|0.107480|70.217153
decoder_fourier_20|0.025671|0.002520|0.050198|25.381582
decoder_teaser_20|0.086983|0.013105|0.114461|72.637299

All training results on MobilePek datasets

model_embed_epoch|MAE|MSE|RMAE|MAPE|
-|-|-|-|-|
erpp_hier_20|0.036912|0.003281|0.057283|10.257857
erpp_ctle_20|0.038718|0.003493|0.059097|10.071074
erpp_fourier_20|0.030224|0.002447|0.049466|7.835022
erpp_teaser_20|0.059467|0.007135|0.084466|16.332932
stlstm_hier_20|0.036948|0.003317|0.057593|10.285524
stlstm_ctle_20|0.038272|0.003442|0.058669|10.223886
stlstm_fourier_20|0.030201|0.002375|0.048731|7.835775
stlstm_teaser_20|0.058918|0.007015|0.083752|16.391380
rnn_hier_20|0.039617|0.003656|0.060466|10.899536
rnn_ctle_20|0.039810|0.003500|0.059157|10.444212
rnn_fourier_20|0.031031|0.002606|0.051051|8.180257
rnn_teaser_20|0.057177|0.006693|0.081809|15.822698
gru_hier_20|0.040875|0.003627|0.060222|11.416270
gru_ctle_20|0.038376|0.003436|0.058613|10.106173
gru_fourier_20|0.030500|0.002414|0.049135|8.020671
gru_teaser_20|0.065575|0.007958|0.089210|17.986450
decoder_hier_20|0.039498|0.003526|0.059380|10.490977
decoder_ctle_20|0.039604|0.003546|0.059548|10.370089
decoder_fourier_20|0.035800|0.003194|0.056516|9.280954
decoder_teaser_20|0.043221|0.003958|0.062916|11.728681
