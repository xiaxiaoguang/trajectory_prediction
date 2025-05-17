
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

model_with_pretrained_embedding|Accuracy|Recall|F1_micro|F1_macro
-|-|-|-|-
erpp_20_Fourier|0.050968|0.025593|0.050968|0.012206
stlstm_20_Fourier|0.051809|0.026141|0.051809|0.012814
rnn_20_Fourier|0.039588|0.018694|0.039588|0.007316
gru_20_Fourier|0.044877|0.026248|0.044877|0.014529 
decoder_20_Fourier|0.063189|0.059480|0.063189|0.042875
transformer_20_Fourier|0.061586|0.056520|0.061586|0.039440

model_WO_pretrained_embedding|Accuracy|Recall|F1_micro|F1_macro
-|-|-|-|-
erpp_20|0.013463|0.003256|0.013463|0.000734
stlstm_20|0.015106|0.004390|0.015106|0.001319
rnn_20|0.017190|0.006710|0.017190|0.002766
gru_20|0.014224|0.006858|0.014224|0.003724
decoder_20|0.064110|0.062251|0.064110|0.047898
transformer_20|0.066314|0.060498|0.066314|0.045106

model_with_pretrained_embedding|Accuracy|Recall|F1_micro|F1_macro
-|-|-|-|-
erpp_20_ctle|0.051689|0.024430|0.051689|0.013265
stlstm_20_ctle|0.050968|0.023019|0.050968|0.011874
rnn_20_ctle|0.067476|0.050125|0.067476|0.038343
gru_20_ctle|0.060624|0.043509|0.060624|0.034083
decoder_20_ctle|0.065753|0.056738|0.065753|0.042888
transformer_20_ctle|0.061746|0.056165|0.061746|0.041975

model_with_pretrained_embedding|Accuracy|Recall|F1_micro|F1_macro
-|-|-|-|-
erpp_20_hier|0.020355|0.007134|0.020355|0.002282
stlstm_20_hier|0.021156|0.008166|0.021156|0.002628
rnn_20_hier|0.011660|0.003508|0.011660|0.001260
gru_20_hier|0.016989|0.009695|0.016989|0.006289

model_with_pretrained_embedding|Accuracy|Recall|F1_micro|F1_macro
-|-|-|-|-
rnn_20_skipgram|0.003686|0.000392|0.003686|0.000038
erpp_20_skipgram|0.003446|0.000309|0.003446|0.000014
gru_20_skipgram|0.002845|0.000241|0.002845|0.000024
stlstm_20_skipgram|0.005089|0.000731|0.005089|0.000069

model_with_pretrained_embedding|Accuracy|Recall|F1_micro|F1_macro
-|-|-|-|-
rnn_20_tale|0.017951|0.007358|0.017951|0.003296
erpp_20_tale|0.019834|0.006090|0.019834|0.001978
gru_20_tale|0.019834|0.010947|0.019834|0.007018
stlstm_20_tale|0.019874|0.006423|0.019874|0.002381

model_with_pretrained_embedding|Accuracy|Recall|F1_micro|F1_macro
-|-|-|-|-
rnn_20_teaser|0.003726|0.000352|0.003726|0.000034
erpp_20_teaser|0.003606|0.000356|0.003606|0.000014
gru_20_teaser|0.003366|0.000397|0.003366|0.000090
stlstm_20_teaser|0.004408|0.000765|0.004408|0.000143


model_with_pretrained_embedding|Accuracy|Recall|F1_micro|F1_macro
-|-|-|-|-
rnn_20_poi2vec|0.014024|0.006366|0.014024|0.002123
erpp_20_poi2vec|0.018872|0.006544|0.018872|0.002151
gru_20_poi2vec|0.014986|0.007752|0.014986|0.004074
stlstm_20_poi2vec|0.021757|0.007317|0.021757|0.002375
