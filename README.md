# Outlier Analysis (2017) - Charu C. Aggarwal
- Info
  - 주로 Unsupervised learning의 상황에서 이상치를 탐지하는 알고리즘들에 대한 설명으로 이루어져있다.
  - 특정 task를 해결하기 위해 필요할 부분을 찾아보는 reference book으로 적절하다.
  - 논문들을 많이 알려주고 있기 때문에 참고하면 좋을 것 같다.
- Contents
  - Chapter02 Probabilistic and Statistical Models for Outlier Detection
  - Chapter03 Linear Models for Outlier Detection
    - Linear Regression, PCA, OCSVM
  - Chapter04 Proximity-Based Outlier Detection
    - Distance-Based
    - Density-Based (LOF, LOCI, Histogram, Kernel Density)
  - Chapter05 High-Dimensional Outlier Detection
    - Axis-Parallel subsapce
    - Generalized subspace
  - Chapter06 Outlier Ensembles
    - Variance reduction
    - Bias reduction
  - Chapter07 Supervised Outlier Detection
    - Cost-Sentitive (MetaCost, Weighting Method)
    - Adaptive Re-sampling (SMOTE)
    - Boosting
    - Semi-Supervision
    - Supervised Models for Unsupervised Outlier Detection
  - Chapter08 Outlier Detection in Categorical, Text, and Mixed Attributed Data
  - Chapter09 Time Series and Streaming Outlier Detection
    - Prediction-based Anomaly Detection
      - Univariate aase (ARIMA) : [`My Code`](./Outlier%20Analysis)
      - Multiple Time Series : [`My Code`](./Outlier%20Analysis)
        - selection method
        - PCA method

# Paper
### Categoriztion of Deep Anomaly Detection
- Deep learning for feature extraction
- Learning feature representations of normality
  - Generic normality feature learning
    - AutoEncoder, GAN, Predictability Modeling, Self-Supervised classification
  - Anomaly measure-dependent feature learning
    - Distance-based classification, One-class classification measure, Clustering-based measure
- End-to-end anomaly score learning
  - Ranking model, Prior-driven model, Softmax likelihood model, End-to-end one-class classification

### Survey
- Deep Learning for Anomaly Detection A Review (2020)
  - [`Paper Link`](https://arxiv.org/pdf/2007.02500.pdf) | [`My Summary`](./My%20summary) | `My Code`
  - `Key Word` : deep anomaly detection
- Autoencoders (2020)
  - [`Paper Link`](https://arxiv.org/pdf/2003.05991.pdf) | `My Summary` | `My Code`
  - `Key Word` : autoencoder

### Learning feature representations of normality
- Outlier Detection with AutoEncoder Ensemble (2017)
  - [`Paper Link`](https://saketsathe.net/downloads/autoencoder.pdf) | `My Summary` | `My Code`
  - `Key Word` : autoencoder, ensemble
- Auto-Encoding Variational Bayes (2014)
  - [`Paper Link`](https://arxiv.org/abs/1312.6114) | [`My Summary`](./My%20summary) | [`My Code`]((./My%20code))
  - `Key Word` : variational autoencoder
- Generatice Adversarial Nets (NIPS 2014)
  - [`Paper Link`](https://papers.nips.cc/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html) | [`My Summary`](https://minsoo9506.github.io/blog/GAN/) | [`My Code`](./My%20code) 
  - `Key Word` : gan
- Least Squares Generative Adversarial Networks (2016)
  - [`Paper Link`](https://arxiv.org/abs/1611.04076) | [`My Summary`](https://minsoo9506.github.io/blog/LSGAN/) | [`My Code`](./My%20code) 
  - `Key Word` : gan
- Adversarial Autoencoders (2016)
  - [`Paper Link`](https://arxiv.org/abs/1511.05644) | [`My Summary`](./My%20summary) | `My Code`
  - `Key Word` : autoencoder, adversarial, semi-supervised, dimension reduction
- Generative Probabilistic Novelty Detection with Adversarial Autoencoders (NIPS 2018)
  - [`Paper Link`](https://papers.nips.cc/paper/2018/file/5421e013565f7f1afa0cfe8ad87a99ab-Paper.pdf) | `My Summary`| `My Code`
- DEEP AUTOENCODING GAUSSIAN MIXTURE MODEL FOR UNSUPERVISED ANOMALY DETECTION (ICLR 2018)
  - [`Paper Link`](https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf) | [`My Summary`](./My%20summary) | `My Code`
  - `Key Word` : Dim reduction using AE, GMM using NN, unsupervised

### Time Series and Streaming Anomaly Detection
- Anomaly Detection In Univariate Time-Series : A Survey on the state-of-the-art
  - [`Paper Link`](https://arxiv.org/abs/2004.00433) | `My Summary` | `My Code`
  - `Key Word` : anomaly detection, time series
- USAD : UnSupervised Anomaly Detection on multivariate time series (KDD2020)
  - [`Paper Link`](https://dl.acm.org/doi/10.1145/3394486.3403392) | [`My Summary`](./My%20summary) | `My Code`
  - `Key Word` : Using both GAN & AE simultaneously, multivariate time series

# Project
- Kaggle Credit Card Fraud Detection (진행중)
  - tabular, binary classification, imbalance
  - 이론 실습 내용
    - `SMOTE`
    - `Unsupervised PCA based Anomaly Detection`
- 네트워크임베딩 대학원수업 기말 프로젝트 (진행중)
  - Using Graph Embedding Ensemble in Unsupervised Anomaly Detection
    - tabular, unsupervised
    - `Node2Vec`
- 다변량분석 대학원수업 기말 프로젝트 (진행중)
  - tabular, unsupervised
  - `PCA`, `Kernel PCA`