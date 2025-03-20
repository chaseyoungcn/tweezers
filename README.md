# Tweezers: A Framework for Security Event Detection via Event Attribution-centric Tweet Embedding

## Overview

Tweezers is a framework for security event detection via event attribution-centric tweet embedding. It processes security-related tweets and detects security events through advanced embedding techniques and clustering.

## Installation

### Setup
1. Clone the repository:
```bash
git clone https://github.com/jiancui-research/tweezers.git
cd tweezers
```

2. Create and activate a conda environment:

```bash
conda env create -f environment.yml
conda activate tweezers
```

## Project Structure

```bash
├── data/
│ ├── keywords_crawl.txt # Keywords for Twitter API filtering
│ ├── 202401_tweets_eval.json # Evaluation dataset for January 2024
│ └── 202402_tweets_eval.json # Evaluation dataset for February 2024
│
├── model/
│ ├── tweetembedder.py # Tweet embedding model implementation
│ ├── tweetclassifier.py # Tweet classification model
│ └── gnn.py # Graph Neural Network baseline implementations
│
├── tweet_features/ # Directory for storing embeddings of tweets with different embedding methods
│
├── utils/
│ ├── classification_utils.py # Utilities for tweet classification
│ ├── clustering_utils.py # Utilities for tweet clustering 
    ｜-get_embeddings 由图和原始embedding（由bertweet生成）使用tweetbedder生成embedding    
    ｜-find_best_eps  寻找最佳邻域eps，其中使用dbscan对embedding聚类，使用eids作为真实标签和预测标签计算指标nmi
│ ├── early_stopping.py # Early stopping implementation
│ ├── loss.py # Loss functions for training tweet embedding model
│ └── preprocess_text.py # Text preprocessing regex rules
│
├── trained_models/ # Directory for storing trained models
│ └── [model_name].pt # Trained model checkpoints
├── environment.yml # Conda environment specification
└── eval_tweetembedder.py # Evaluation script
  ｜-1. 加载tweet_features中已生成的原始embedding
  ｜-2. 调用get_embeddings获取tweetembedding（new）
  | -3. 调用find_best_eps 输出最佳指标
```

## Usage

### 1. Tweet Embedding

Run the following command to generate the tweet embedding performance reported in the paper (for testing set 202401 and 202402):
```bash
python eval_tweetembedder.py
```

### 2. End2End Security Event Detection


Given a tweet dataset, you should run the processing pipeline as needed (you can refer to the `utils/preprocess_text.py` for some necessary regex rules), then you can leverage our tweet embedding model to generate the tweet embedding and leverage DBSCAN to detect the security events.


## Model Training

The framework uses pre-trained models that can be found in the `trained_models/` directory. 
The training code will be released soon.


## Data Availability
Due to Twitter's API terms of service and privacy policies, we cannot directly share tweet contents. Instead, our released datasets (`202401_tweets_eval.json` and `202402_tweets_eval.json`) only contain tweet IDs. Researchers can use these IDs to retrieve the full tweet data using Twitter's API in accordance with their terms of service.


## Citation

If you use this framework in your research, please cite:
```bibtex
@article{cui2024tweezers,
  title={Tweezers: A Framework for Security Event Detection via Event Attribution-centric Tweet Embedding},
  author={Cui, Jian and Kim, Hanna and Jang, Eugene and Yim, Dayeon and Kim, Kicheol and Lee, Yongjae and Chung, Jin-Woo and Shin, Seungwon and Liao, Xiaojing},
  journal={arXiv preprint arXiv:2409.08221},
  year={2024}
}
```

## Contact
For any questions or feedback, please contact:
Jian Cui (cuijian@iu.edu)

