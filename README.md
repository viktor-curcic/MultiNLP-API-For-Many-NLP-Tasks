# MultiNLP-API-For-Many-NLP-Tasks

## Introduction

MultiNLP is an API capable of doing text summarization, question answering and sentiment analysis (more to come). It is realized using a fine-tuned T5 model and exported through Gradio. 

## Datasets

Datasets include: imdb - a large movie review dataset written for sentiment analysis; cnn_dailymail - a dataset comprising 300k articles and their summaries, used for summarization; and squad - Stanford Question Answering Dataset, a large training set for question answering. Five thousand data points were each taken from the datasets to be used for testing and training.

### Problems

The imdb dataset is informal in the first half, and formal in the second half. My results all gave negative as the sentiment before realizing the error. The dataset is now randomly shuffled.

## Metrics

They include: Accuracy (approx. 90% for MultiNLP) - for sentiment analysis; BLEU (approx. 10%) - poor results indicate problem with text generation, should fix!; ROUGE (approx. 50%) - good summarization capabilities

## Requirements

Install requirements

pip install -r requirements.txt

## Run file

python trainer.py

## Run API

python app.py

