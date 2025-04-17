# NN-ASSIGNMENT_4
NAME:B.KISHORE BABU
ID:700752976

1.What is the difference between stemming and lemmatization? Provide examples with the word “running.”
Stemming cuts off word endings without understanding context, so "running" becomes "run" by chopping.
Lemmatization uses grammar rules and vocabulary to return the base form, so "running" also becomes "run", but in a more accurate, context-aware way.

2.Why might removing stop words be useful in some NLP tasks, and when might it actually be harmful?
Removing stop words is useful in tasks like text classification or topic modeling because it reduces noise and focuses on meaningful words.
However, it can be harmful in tasks like sentiment analysis or question answering, where stop words like "not" or "is" can carry important meaning that affects the result.

1.How does NER differ from POS tagging in NLP?
NER (Named Entity Recognition): Identifies specific entities (e.g., people, locations, dates) in text.
Example: "Barack Obama" → PERSON, "2009" → DATE.
POS Tagging (Part-of-Speech): Labels each word with its grammatical role (e.g., noun, verb, adjective).
Example: "Barack" → NNP (Proper noun), "was" → VB (Verb).

2.Describe two applications that use NER in the real world (e.g., financial news, search engines).
1. Financial News Analysis:
Application: NER helps extract important entities such as companies, stock symbols, and financial terms from financial news articles.
Example: Extracting entities like "Apple Inc." (organization), "TSLA" (stock symbol), and "2025" (year) from articles allows automated systems to track and analyze market trends.
2. Search Engines:
Application: NER enhances search engine results by identifying and categorizing entities in queries and web pages, improving the accuracy of search results.
Example: If a user searches for "Tesla stock price," NER helps identify "Tesla" as a company and "stock price" as a financial query, delivering the most relevant results.

1.Why do we divide the attention score by √d in the scaled dot-product attention formula?
We divide the attention score by √d (where d is the dimension of the key vectors) in the scaled dot-product attention to prevent the dot products from becoming too large.
Reason :
When d is large, the dot product values between Q and K can grow big.
This causes the softmax function to output very small gradients (due to saturation), making it hard for the model to learn.
Dividing by √d scales down the scores, keeping them in a range where softmax works effectively.
Example:
If Q and K have large values, their dot product could be, say, 100.
softmax([100, 1]) ≈ [1, 0] → Very sharp, almost binary.
After scaling:
softmax([100/√64, 1/√64]) ≈ [0.88, 0.12] → Smoother, more learnable.

2.How does self-attention help the model understand relationships between words in a sentence?
Self-attention allows a model to focus on different words in a sentence when processing each word. It helps capture contextual relationships like:
Example Sentence:
"The cat sat on the mat because it was tired."
When processing "it", self-attention helps the model focus on "cat" to understand what "it" refers to.
Key Benefits:
Captures long-range dependencies (e.g., link "it" to "cat" even if they're far apart).
Provides context-aware meaning (e.g., "bank" in "river bank" vs. "money bank").
Handles multiple relationships at once using attention weights.


