# Data-Mining

1. [Spark Fundamentals](#spark-fundamentals-link)
2. [Frequent Itemsets](#frequent-itemsets-link)
3. [Similar Items and Recommendation Systems](#similar-items-and-recommendation-systems-link)
4. [Community Detection](#community-detection-link)
5. [Streaming](#streaming-link)
6. [Clustering](#clustering-link)

### Spark Fundamentals
This assignment involved three parts which include the following, working on the yelp challenge dataset:

Task 1: Data Exploration ([link](https://github.com/rohitpenumarti/Data-Mining/blob/master/hw1/task1.py))
 - total number of reviews
 - number of reviews in 2018
 - number of distinct users who wrote reviews
 - top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote
 - number of distinct businesses that have been reviewed
 - top 10 businesses that had the largest number of reviews and the number of reviews they had

Task 2: Partition ([link](https://github.com/rohitpenumarti/Data-Mining/blob/master/hw1/task2.py))
 - show number of partitions for the RDD used in task 1 question 6 and number of items per partition
 - use customized partition function to improve the performance of map and reduce tasks - include time duration comparison between the default partition and the customized partition

Task 3: Exploration on Multiple Datasets ([link](https://github.com/rohitpenumarti/Data-Mining/blob/master/hw1/task3.py))
 - What are the average stars for each city

### Frequent Itemsets
This assignment involved two parts which include the following:

Task 1: Frequent Itemsets ([link](https://github.com/rohitpenumarti/Data-Mining/blob/master/hw2/task1.py))
 - compute frequent businesses (singletons, pairs, triples, etc.) qualified as frequent given support threshold
 - compute frequent users (singletons, pairs, triples, etc.) qualified as frequent given support threshold

Task 2: SON Algorithm ([link](https://github.com/rohitpenumarti/Data-Mining/blob/master/hw2/task2.py))
 - preprocess Ta Feng dataset (kaggle) data to consider purchases by same consumer on separate days as different transactions
 - implement the SON algorithm to find frequent itemsets on qualified users (filter threshold k) based on support threshold

### Similar Items and Recommendation Systems
This assignment involved three parts including the following, using the yelp challenge dataset:

Task 1: Jaccard Based LSH ([link](https://github.com/rohitpenumarti/Data-Mining/blob/master/hw3/task1.py))
 - implement Locality Sensitive Hashing algorithm using Jaccard Similarity and return candidate pairs

Task 2.1: Item-Based Recommendation System ([link](https://github.com/rohitpenumarti/Data-Mining/blob/master/hw3/task2_1.py))
 - implement Item-based collaborative filtering recommendation system with pearson similarity

Task 2.2: Model-Based Recommendation System ([link](https://github.com/rohitpenumarti/Data-Mining/blob/master/hw3/task2_2.py))
 - implement model-based recommendation system using XGBoost

Task 2.3: Hybrid Recommendation System ([link](https://github.com/rohitpenumarti/Data-Mining/blob/master/hw3/task2_3.py))
 - implement hybrid recommendation system by combining both Item-based and Model-based recommendation systems

### Community Detection ([link](https://github.com/rohitpenumarti/Advanced-Mathematical-Finance/blob/master/Homeworks/Homework%204/Penumarti.Rohit.HW4.ipynb))
This assignment involved two parts including the following, using the yelp challenge dataset:

Task 1: Community Detection Using GraphFrames ([link](https://github.com/rohitpenumarti/Data-Mining/blob/master/hw4/task1.py))
 - create graph from user business pairs: two users have edges between them if they have reviewed at least k (filter threshold) common business between the two
 - implement community detection algorithm using Spark GraphFrames using the Label Propagation Algorithm

Task 2: Community Detection using Girvan-Newman Algorithm ([link](https://github.com/rohitpenumarti/Data-Mining/blob/master/hw4/task2.py))
 - compute betweenness between each edge in original graph constructed in previous part
 - group users based modularity of communities

### Streaming
This assignment involved three parts including the following:

Task 1: Bloom Filtering ([link](https://github.com/rohitpenumarti/Data-Mining/blob/master/hw5/task1.py))
 - implement Bloom Filtering algorithm to estimate whether a specific user_id has appeared in the data stream before

Task 2: Flajolet-Martin ([link](https://github.com/rohitpenumarti/Data-Mining/blob/master/hw5/task2.py))
 - implement Flajolet-Martin algorithm to estimate the number of unique users within a window in the data stream

Task 3: Fixed Size Sampling ([link](https://github.com/rohitpenumarti/Data-Mining/blob/master/hw5/task3.py))
 - implement reservoir sampling algorithm

### Clustering
This assignment involved one part including the following:

Task: Bradley-Fayyad-Reina (BFR) ([link](https://github.com/rohitpenumarti/Data-Mining/blob/master/hw6/task.py))
 - implement the BFR algorithm on a given dataset

### Final Project
This is the final project which is an extension of assignment 3 ([link](https://github.com/rohitpenumarti/Data-Mining/tree/master/FinalProject)). For this project, we had to lower the RMSE value to a suitable value. To achieve this, I created many new features using the user and business data provided as a part of the Yelp dataset challenge, and then I created the model and used Bayesian optimization to tune the hyperparameters (hyperopt). I then used the hybrid recommendation system to give the results.
