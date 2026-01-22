# FinRegNLP – Comparative Analysis of Fintech Regulations  

## Overview  
FinRegNLP is a Natural Language Processing (NLP) project that analyzes and compares fintech-related regulations across three major jurisdictions:  
- **India** – The Payment and Settlement Systems Act, 2007  
- **UK** – Directive (EU) 2015/2366 (Payment Services Directive 2 - PSD2)  
- **USA** – Dodd-Frank Wall Street Reform and Consumer Protection Act  

The project applies NLP techniques to extract, preprocess, and analyze key similarities, differences, sentiment, and themes in payment system regulations, consumer protection laws, and financial stability measures.  

---

## Features  
- **Text Preprocessing** – Cleaning, tokenization, stopword removal, lemmatization.  
- **Similarity Analysis** –  
  - TF-IDF + cosine similarity → India–UK: **0.36**, India–US: **0.39**, UK–US: **0.21**  
  - BERT embeddings + cosine similarity → US–UK: **0.75**, US–India: **0.72**, India–UK: **0.71**  
- **Sentiment Analysis** – VADER sentiment scoring:  
  - Neutral tone dominates (India: **66.87%**, US: **52.86%**, UK: **40.23%**)  
  - UK regulations show highest positive sentiment (**45.06%**)  
- **Topic Modeling** – LDA topic modeling to extract thematic clusters.  
- **Visualization** – Graphs for similarity scores, sentiment distribution, and topic frequency.  

---

## Methodology  
1. **Data Extraction** – Collected regulation texts in PDF format and extracted raw text.  
2. **Preprocessing** – Standard NLP pipeline:  
   - Lowercasing  
   - Removing punctuation/numbers  
   - Tokenization  
   - Stopword removal  
   - Lemmatization  
3. **Similarity Analysis** –  
   - TF-IDF + cosine similarity for lexical similarity.  
   - BERT embeddings + cosine similarity for semantic similarity.  
4. **Sentiment Analysis** – Applied VADER to measure tone across jurisdictions.  
5. **Topic Modeling** – Used Latent Dirichlet Allocation (LDA) to identify major themes.  
6. **Visualization** – Plotted results for similarity, sentiment trends, and topic distributions.  

---

## Results (Key Insights)  
- **Lexical similarity** is low across jurisdictions (0.21–0.39), but **semantic similarity** is much higher (0.71–0.75), highlighting differences in wording but alignment in meaning.  
- **Sentiment analysis** confirms a predominantly neutral regulatory tone, with the **UK being the most positive** at 45.06%.  
- **Topic modeling** reveals recurring themes in consumer protection, payment systems, and compliance requirements.  

---

## How to Run  
1. Clone the repository:  
```bash
git clone https://github.com/harrishsrinivasan/NLP-Driven-Comparative-Analysis-of-Fintech-Regulations.git
