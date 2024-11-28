FAKE REVIEW CLASSIFICATION AND TOPIC MODELING


INTRODUCTION:
	
Fake Review Classification and Topic Modeling is an advanced area in natural language processing (NLP) aimed at identifying deceptive reviews and understanding key themes in textual data. Fake reviews, often used to manipulate public opinion, undermine trust in online platforms. Classification techniques leverage machine learning algorithms, deep learning algorithm and pre-trained model RoBERTa to detect these fraudulent reviews. Topic modeling, a complementary approach, extracts underlying themes, providing insights into genuine customer concerns and discussions. By integrating both methods, organizations can enhance review authenticity, improve decision-making, and promote transparent digital ecosystems, ensuring users access reliable and valuable information for informed choices.

DATA COLLECTION:
•	The dataset contains product reviews with text and metadata such as ratings , votes etc..
•	It has 40432 entries and 4 attributes.

DATA PRE-PROCESSING:

The pre-processing steps that are used in this data are as follows:
•  Tokenization: Splitting the text into individual words.
•  Removal of non-alphabetic tokens: Filtering out non-alphabetic characters.
•  Lowercasing: Converting all words to lowercase for uniformity.
•  Stopword removal: Removing common English stopwords (e.g., "the," "and") using NLTK's stopword list.
•  Lemmatization: Reducing words to their base or root form using WordNetLemmatizer.
•  Stemming: Further reducing words to their stems using PorterStemmer.
•  TF-IDF Vectorization: Transforming the cleaned text into numerical vectors with a vocabulary of 5000 words, excluding stopwords.

METHODOLOGY:
ELBOW METHOD TO FIND THE OPTIMAL CLUSTERS:

The Elbow Method is a technique used to determine the optimal number of clusters in k-means clustering. It involves plotting the Within-Cluster-Sum of Squared Errors (WCSS) against the number of clusters and identifying the "elbow point" where WCSS reduction slows. It ensures effective clustering and prevents overfitting or underfitting.

 
CLUSTERING:
For clustering, we used K-Means clustering and DBSCAN clustering.

K-Means Clustering: 

•	K-means clustering is an unsupervised machine learning algorithm used to partition data into K clusters based on feature similarity. It iteratively assigns data points to the nearest cluster center (centroid) and updates the centroids to minimize the Within-Cluster-Sum of Squares (WCSS). It's widely used for segmentation and pattern recognition tasks.
•	The optimal k value chosen here is 4.
•	To find the effectiveness of clustering, we used silhouette score to evaluate. The silhouette score of k-means clustering is 0.00963 which is not a good value.
•	Therefore, we used DBSCAN clustering.

DBSCAN Clustering:

•	DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that groups data points based on density. It identifies clusters as dense regions separated by sparser regions, handling noise and outliers effectively. DBSCAN doesn’t require predefined cluster counts, making it suitable for non-linear and irregularly shaped data. 
•	To find the effectiveness of clustering, we used silhouette score to evaluate the model. The silhouette score of DBSCAN clustering is 0.373 which is better than K-Means Clustering.

TOPIC MODELING:
Topic modeling is an unsupervised machine learning technique used to identify hidden themes or topics in large text datasets. It helps organize and summarize textual data by clustering words into topics based on co-occurrence patterns. LDA is used here for topic modeling.

LDA:

LDA (Latent Dirichlet Allocation) is a popular topic modeling algorithm. It assumes documents are mixtures of topics, where each topic is a distribution of words. LDA is used because it provides interpretable results, effectively capturing topic structures in text, aiding in applications like text summarization, sentiment analysis, and information retrieval.

FAKE REVIEW CLASSIFICATION:
	After clustering and topic modeling, we move onto fake review classification. For this, we used two machine learning algorithms, LSTM, RoBERTa for comparison.

Machine learning algorithms:
	The two machine learning algorithms used here are Logistic Regression and Support Vector Machine.

Deep learning algorithm:
	The deep learning algorithm used here is LSTM.

RoBERTa:
	RoBERTa (Robustly Optimized BERT Approach) is a transformer-based NLP model built on BERT. It improves performance by optimizing training strategies, such as dynamic masking, larger datasets, and longer training. RoBERTa is used for tasks like text classification, sentiment analysis, and question answering, offering superior accuracy and generalization in language understanding.

MODEL EVALUATION:
•	For evaluating the machine learning models, we used measures such as accuracy, F1 – score, precision, recall.
•	In case of deep learning and RoBERTa, we used accuracy and loss for evaluating the model.

CONCLUSION:
	
Based on the bar chart comparing model accuracies:

o	LSTM achieves the highest accuracy (90%), making it the best-performing model for the given task.
o	SVM follows with an accuracy of 87%, performing better than LR and RoBERTa.
o	Logistic Regression (LR) scores 85%, slightly outperforming RoBERTa, which has the lowest accuracy at 83%.

LSTM is the most suitable model for this task, demonstrating superior accuracy, while RoBERTa's lower accuracy suggests it may need further tuning or might be less effective for this dataset.

