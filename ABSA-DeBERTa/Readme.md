# Aspect-based Sentiment Analysis using BERT with Disentangled Attention

Aspect-Based Sentiment Analysis (ABSA) tasks aim to identify consumer's opinions about different aspects of products or services. BERT-based language models have been used successfully in applications that require a deep understanding of the language, such as sentiment analysis. This paper investigates the use of disentangled learning to improve BERT-based textual representations in ABSA tasks. Motivated by the success of disentangled representation learning in the field of computer vision, which aims to obtain explanatory factors of the data representations, we explored the recent DeBERTa model (Decoding-enhanced BERT with Disentangled Attention) to disentangle the syntactic and semantics features from a BERT architecture. Experimental results show that incorporating disentangled attention and a simple fine-tuning strategy for downstream tasks outperforms state-of-the-art models in ABSA's benchmark datasets.

## Reproducing results from paper

To further reproduce the results of ABSA-DeBERTa the [notebook](DeBERTa_Experiment.ipynb) should be followed. It is recommended to execute the notebook on Google Colaboratory for easy use.

