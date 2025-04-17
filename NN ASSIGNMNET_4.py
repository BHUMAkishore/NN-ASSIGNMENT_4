#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK resources (only the first time)
nltk.download('punkt')
nltk.download('stopwords')

def nlp_preprocess(sentence):
    # 1. Tokenize
    tokens = word_tokenize(sentence)
    print("Original Tokens:", tokens)
    
    # 2. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens_no_stopwords = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    print("Tokens Without Stopwords:", tokens_no_stopwords)
    
    # 3. Apply stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in tokens_no_stopwords]
    print("Stemmed Words:", stemmed_words)

# Example usage
sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."
nlp_preprocess(sentence)


# In[5]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spay download en_core_web_sm')


# In[8]:


import re

# Input sentence
sentence = "Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."

# Define simple regular expressions to identify named entities
patterns = {
    "PERSON": r"\b[A-Z][a-z]*\s[A-Z][a-z]*\b",  # Match capitalized first and last names
    "DATE": r"\b\d{4}\b",  # Match 4-digit numbers (e.g., 2009)
    "ORDINAL": r"\b\d{1,2}(th|st|nd|rd)\b",  # Match ordinals like 44th, 1st
    "GPE": r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b",  # Match capitalized places (e.g., United States)
    "WORK_OF_ART": r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b",  # Match capitalized phrases (e.g., Nobel Peace Prize)
}

# Extract named entities using the patterns
entities = []
for label, pattern in patterns.items():
    matches = re.findall(pattern, sentence)
    for match in matches:
        entities.append((match, label))

# Output the results
for ent in entities:
    print(f"Text: {ent[0]}, Label: {ent[1]}")


# In[7]:


import numpy as np

def scaled_dot_product_attention(Q, K, V):
    # Step 1: Compute the dot product of Q and Kᵀ
    matmul_qk = np.dot(Q, K.T)
    
    # Step 2: Scale the result by dividing it by √d (where d is the key dimension)
    d_k = Q.shape[-1]  # or K.shape[-1], the key dimension
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)
    
    # Step 3: Apply softmax to get attention weights
    attention_weights = np.exp(scaled_attention_logits) / np.sum(np.exp(scaled_attention_logits), axis=-1, keepdims=True)
    
    # Step 4: Multiply the attention weights by V to get the output
    output = np.dot(attention_weights, V)
    
    return attention_weights, output

# Test inputs
Q = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
K = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
V = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# Apply the scaled dot-product attention function
attention_weights, output = scaled_dot_product_attention(Q, K, V)

# Display results
attention_weights, output


# In[10]:


get_ipython().system('pip install transformers')


# In[18]:


get_ipython().system('pip install --upgrade typing_extensions==4.9.0')


# In[22]:


get_ipython().system('pip install textblob')
#python -m textblob.download_corpora


# In[23]:


from textblob import TextBlob

# Input sentence
sentence = "Despite the high price, the performance of the new MacBook is outstanding."

# Analyze sentiment
blob = TextBlob(sentence)
polarity = blob.sentiment.polarity

# Determine sentiment label
if polarity > 0:
    label = "POSITIVE"
elif polarity < 0:
    label = "NEGATIVE"
else:
    label = "NEUTRAL"

print(f"Sentiment: {label}")
print(f"Confidence Score (approx): {abs(polarity):.4f}")


# In[ ]:




