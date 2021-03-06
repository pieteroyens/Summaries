# Summary IR1 UvA, 2019-2020
- [Summary IR1 UvA, 2019-2020](#summary-ir1-uva-2019-2020)
  - [Week 1: Lecture 1 & 2](#week-1-lecture-1--2)
    - [Introduction](#introduction)
    - [Crawling](#crawling)
    - [Text Analysis](#text-analysis)
    - [Link Analysis](#link-analysis)
      - [Link analysis](#link-analysis-1)
      - [PageRank](#pagerank)
    - [Indexing](#indexing)
      - [Data Structures](#data-structures)
      - [Inverted Index](#inverted-index)
      - [Constructing an Index](#constructing-an-index)
      - [Updating an Index](#updating-an-index)
  - [Week 2: Lecture 3, Offline Evaluation](#week-2-lecture-3-offline-evaluation)
    - [Evaluation Measures](#evaluation-measures)
      - [Relevance as a Binary Classifier](#relevance-as-a-binary-classifier)
      - [Relevance as a non-Binary Classifier](#relevance-as-a-non-binary-classifier)
        - [(N)DCG:](#ndcg)
        - [Model-based measure: browsing/positions-based models, RPB](#model-based-measure-browsingpositions-based-models-rpb)
        - [Model-based measure: document utility, Cascade Model, Expected Effort](#model-based-measure-document-utility-cascade-model-expected-effort)
        - [Novelty-based measures](#novelty-based-measures)
        - [Issues with previous measures, Session measures](#issues-with-previous-measures-session-measures)
  - [Week 2: Lecture 4, Meta-Evaluation](#week-2-lecture-4-meta-evaluation)
    - [Comparing Metrics](#comparing-metrics)
      - [Maximum Entropy Method](#maximum-entropy-method)
    - [Collection Construction](#collection-construction)
  - [Week 2, Lecture 4: Query Processing](#week-2-lecture-4-query-processing)
    - [Query Analysis](#query-analysis)
    - [Query Processing/ Ranking](#query-processing-ranking)
      - [First-Phase Query Processing /  Simple Ranking](#first-phase-query-processing--simple-ranking)
      - [Simple Ranking: Term based Vector Space Model](#simple-ranking-term-based-vector-space-model)
      - [Simple Ranking: Term based Language Modeling](#simple-ranking-term-based-language-modeling)
      - [Simple Ranking: Term based BM25](#simple-ranking-term-based-bm25)
  - [Week 3: Lecture 5, Semantic Based Retrieval](#week-3-lecture-5-semantic-based-retrieval)
    - [Simple Ranking: Topic based Modeling](#simple-ranking-topic-based-modeling)
    - [Simple Ranking: Latent Semantic Indexing/ Analysis, Vector models](#simple-ranking-latent-semantic-indexing-analysis-vector-models)
    - [Intermezzo: which technique works better](#intermezzo-which-technique-works-better)
    - [Simple Ranking: Nerual Models](#simple-ranking-nerual-models)
      - [Word Embeddings](#word-embeddings)
      - [Document Embeddings](#document-embeddings)
  - [Week 3: Lecture 6, Offline Learning to Rank](#week-3-lecture-6-offline-learning-to-rank)
    - [LTR Preliminaries](#ltr-preliminaries)
    - [LTR Goals](#ltr-goals)
      - [Pointwise Approach](#pointwise-approach)
      - [Pairwise Approach](#pairwise-approach)
      - [Pairwise Approach: RankNet](#pairwise-approach-ranknet)
      - [Listwise Approach: LambdaRank](#listwise-approach-lambdarank)
      - [Other Approaches: ListNet, ListMLE](#other-approaches-listnet-listmle)
    - [LTR Recap](#ltr-recap)
  - [Week 4, Lecture 7: IR-User Interaction](#week-4-lecture-7-ir-user-interaction)
    - [Online evaluation](#online-evaluation)
      - [A/B Testing](#ab-testing)
      - [Interleaving](#interleaving)
    - [Interaction Models](#interaction-models)
      - [Basic Click Models](#basic-click-models)
        - [Position-based model](#position-based-model)
        - [Cascade model](#cascade-model)
      - [Parameter Estimation for Click Models](#parameter-estimation-for-click-models)
      - [Applications of Click Models](#applications-of-click-models)
  - [Week 4, Lecture 8: Counterfactual Learning to Rank](#week-4-lecture-8-counterfactual-learning-to-rank)
    - [Learning from User Interactions](#learning-from-user-interactions)
    - [Unbiased Learning to Rank](#unbiased-learning-to-rank)
    - [Counterfactual Learning to Rank](#counterfactual-learning-to-rank)
      - [Evaluation](#evaluation)
        - [Naive Evaluation](#naive-evaluation)
        - [Counterfactual Evaluation: Inverse Propensity Scoring (IPS)](#counterfactual-evaluation-inverse-propensity-scoring-ips)
      - [Propensity-weighted Learning to Rank](#propensity-weighted-learning-to-rank)
        - [Possible Different Metrics](#possible-different-metrics)
      - [Summary](#summary)
    - [Estimating Position Bias](#estimating-position-bias)
      - [RandTop-n Algorithm](#randtop-n-algorithm)
      - [RandPair](#randpair)
      - [Intervention Harvesting](#intervention-harvesting)
      - [Jointly Learning and Estimating](#jointly-learning-and-estimating)
  - [Week 5, Lecture 9: Online LTR](#week-5-lecture-9-online-ltr)
    - [Online Evaluation, Evaluation between two systems](#online-evaluation-evaluation-between-two-systems)
      - [Online Evaluation: Team-Draft Interleaving](#online-evaluation-team-draft-interleaving)
      - [Online Evaluation: Probablistic Interleaving](#online-evaluation-probablistic-interleaving)
      - [Online Evaluation: Optimized Interleaving](#online-evaluation-optimized-interleaving)
    - [Online Learning to Rank: how to optimize Rankers](#online-learning-to-rank-how-to-optimize-rankers)
      - [Dueling Bandit Gradient Descent (DBGD)](#dueling-bandit-gradient-descent-dbgd)
      - [Reusing Historical Interactions](#reusing-historical-interactions)
      - [Multileave Gradient Descent](#multileave-gradient-descent)
      - [Problem of Dueling Bandit Gradient Descent](#problem-of-dueling-bandit-gradient-descent)
      - [Pairwise Differentiable Gradient Descent](#pairwise-differentiable-gradient-descent)
      - [Online Models Compared](#online-models-compared)
      - [Future for Online LTR](#future-for-online-ltr)
      - [Conclustion Online LTR](#conclustion-online-ltr)
  - [Week 5, Lecture 10: Entity Search](#week-5-lecture-10-entity-search)
    - [Entity Representation](#entity-representation)
      - [1-Field Entity (as Probability Distr over words/categories)](#1-field-entity-as-probability-distr-over-wordscategories)
      - [2-Field Entity (as Probability Distr over words/categories)](#2-field-entity-as-probability-distr-over-wordscategories)
      - [3-Field Entity (as Probability Distr over words/categories)](#3-field-entity-as-probability-distr-over-wordscategories)
      - [Field Entity, how to use](#field-entity-how-to-use)
      - [Knowledge Graph as Tensor (as Vector embedding)](#knowledge-graph-as-tensor-as-vector-embedding)
    - [Learning to Rank Entities](#learning-to-rank-entities)
  - [Week 6, Lecture 11: Conversational Search](#week-6-lecture-11-conversational-search)
  - [Week 6, Lecture 12](#week-6-lecture-12)

___

## Week 1: Lecture 1 & 2

### Introduction

Information Retrieval is technology that *connects* **_people_** to **information**. We have __two__ phases:Offline and Online. In this course, the following will be discussed: 

- Offline
  - Term-based Scoring
  - Semantic Scoring
  - Learning to Rank
  - Offline Evaluation
  - Hypothesis Testing
- Online
  - Counterfactual LTR
  - Interactions
  - Counterfactual Evaluation
  - Online Evaluation

Different **scenarios** include:

- Search and Retrieval
- Conversation Search
- Entity Search
- Recommender Systems

**Subject** of this lecture:
![lectur overview](images/intro_overview.png)

### Crawling
Crawling has two main topic and and some additional topics:

1. *Crawling*
   1. Politeness: How to be polite?
      1. Identify: fill in `user-agent` field in the HTTP request and include the word "crawler", "robot", etc.
      2. Obey Exclusion Protocols: certain pages have exlusio rules
      3. Keep a low bandwidth usage in a given site: crawls delays, emperical thresholds.
   2. Extending the Web Repository:
      1. Strategies to find pages: Random ordering, Bread-first
      2. Prioritization Strategies: Conectivity of pages, Importance of pages
   3. Refreshing the Web Repository:
      1. Age: time passed since last download of a page
      2. Longevity: estimated update frequency of page
      3. **Actual Implementation**: maintaining three queues:
         1. Queue for new sites.
         2. Queue for popular/relevant sites
         3. Queue for the rest of the web
2. *Practical Considerations*
   1. Storage and data structures
   2. Distributed crawling
      1. Firewall mode: crawlers only check own pages
      2. Cross-over mode: Duplicating pages to other crawlers if it's around your border
      3. Exachange mode: crawlers share non-local links and send URLs in small batches
3. *Additional Topics*
   1. Duplicate detection.
      1. Problems: Mirroring, Different URLs &rarr; same content, URL modifiers
      2. Solutions: Hashing by computing complete hash value of document (and then comparing hash values of docs). Or Shinling (near duplicate detection).
   2. Spam
      1. Types: Cloacking (redirection spam), link, content, click. (May want to add further explenation/ solution)

### Text Analysis

1. *Statistical properties of written text*
   1. Zipf's law: how a term is distributed across documents where, on avarage, the first most common word occurs twice as often as the second most common, the second twice as often as the third, etc. ![zipfs law](images/zipfs_law.png)
   2. Heaps' law: an estimate of the vocabulary size as a function of the collection size: $vocab = const * words^\beta$. It is a simple relationship between collection size and vocab size.
2. *Text Analysis Pipeline*
   1. Remove white-space and punctuation
   2. Convert terms to lower-case
   3. Remove stop-words: frequency- or dictionary-based
   4. Convert terms to their stems
   5. Deal with phrases
   6. Apply language-specific processing rules
3. *Stemming*
   1. Algorithmic 
   2. dictionary-based
   3. hybrid
4. *Phrases*
   1. Noun-phrases: sequence of nouns/ adjective followed by nouns
   2. Detecing phrases at query processing time
   3. Use frequent n-grams
5. *Spell Checking*
   1. Simple typos: 
      1. Insertion Error
      2. Deletion Error
      3. Substitution Error
      4. Transpoosition Error
   2. Homophones: words with the same sound but different meaning
   3. Multiple corrections
   4. Considering context

### Link Analysis

#### Link analysis

Link analysis can be done on multiple levels:
  
- Macroscopic: structure of the Web at large
- Mesoscopic: properies of areas/regions of the Web
- Microscopic: statistical properties of links and nodes

Needs possible further explenations
#### PageRank

Checks the qulaity of a page, independent of queries. It's a stationary state of a random walk with teleportation through the pages: we go through a page and follow links, if we're at a dead end, we teleport to a random page, or at each step with a probability.

The more a page is visited, the better the page.The long-term visit rate = PageRank.

Something about **Ergodic Markov Chains**.

Important take-aways: query-independent and precomputed.

### Indexing

The following informaton is specific for WebSearch. How do we store links?

#### Data Structures

Four different structures:

1. Inverted index (see: [Inverted Index](#inverted-index))
2. Web graph: the links are stored seperately
3. Forward Index: index from documents to words
4. Page Attribute File: meta information of document (click-through rate, length, PageRank, etc.)

#### Inverted Index

![Inverted Index](images/inverted_index.png)

**Intuition**: it's like the index of a book: you have a word with the page number it's on. For a webpage: words with links to document ID's.

It's made up of two structures:

1. Dictionary, each entry contains:
   1. Number of pages containing the term
   2. Point to the start of the inverted lists
   3. Other meta-data about the term
2. Inverted Lists: this is like the index of the book, where we can store things like:
   - document number
   - per document how often the word occurs in it
   - document and the positions of the word
   - Weights (TF-IDF, etc.)

Summarised in a picture:
![Inverted Index Summary](images/inverted_index_summary.png)

What we shouldn't store in here: things that are independent of a single word.

- PageRank
- Document Length

#### Constructing an Index

There are simple inexers, they have problems though:

1. In-memory: if the indexes become to large it doesn't fit into memory. There exits two approaches to fix this:
   1. Two-pass index: first pass is collecting statistics, then fill in the inverted index.
   2. One-pass index: when you are out of memory, you off-load it to the disk, rinse and repeat. When you are done, you stick your off-loaded materials together.
2. Single-threaded: you can do MapReduce (distributed indexing)

#### Updating an Index

When we get new information for our webpages or other information, we have to update our indexes. There are several solutions.

1. No Merge
   1. Method: We keep on creating new delta indeces
   2. Pros: low index maintenance cost
   3. Cons: query all new indeces and somehow merge results &rarr; high processing costs.
2. Incremental update
   1. Method: keep on adding things to old index
   2. Pros: no read/write of entire index when updating
   3. Cons: evantually memory for index will run out and will become very slow.
3. Immediate merge (in-memory)
   1. Method: you have a delta index and old index, sometimes you merge the two.
   2. Pros: only one single index
   3. Cons: You use the same index for merging and querying (problems when users want to use your system)
4. Lazy merge
   1. Method: hybrid, merging in the background (new into old). 
5. Page deletions: what to do when a page is deleted and we have to remove it from the index?
   1. Method: keep a lists of deleted documents, then we go over the the entire list and delete the from the collection (garbage collection).

___

## Week 2: Lecture 3, Offline Evaluation

How do we measure/ quantify how good a retrieval system is? Old techniques relied on the **HiPPO-technique: Highest Paid Person's Opinion**. Improvements have been made since.

We can do user-studies (not coverted), off- and online evaluation. This lecture talks about offline evaluation.

Offline evaluation we proxy a user and ask if the algorithm brought up more relant results than another. The two element we have to test a system:

1. User Queries
2. Retrieved Documents

and then labels of relevance for each document.

Offline evaluation is made up of three parts: [evaluation measures](#evaluation-measures), [comparisons](#comparisons), [collection construction](#collection-construction).

### Evaluation Measures

There are about 200+ measures, therefor we have to carefully choose which measure to use. 

First, we have to define what a "*relevant*" document is. This can be done in multiple ways and we have to assume the following things about relevance:

1. **Topical**: if both a query and document are about the same topic.
2. **Binary**: relevant vs. non-relevant
3. **Independent**: relevant of documet A does not depend on document B
4. **Stable**: judgements change over time and are never stable
5. **Consistent**: the relevance labels are consitent across different judges
6. **Complete**: we have the labels for every document in a collection

#### Relevance as a Binary Classifier

The two metric then are *precision* and *recall*:
![precision and recall](images/evaluation_precision_recall.png)

When we have a ranked list, how do we turn it into a binary classification problem?

Answer: we look at the *top-k* documents, and see which of the retrieved documents are relevant. We do this for Precision and Recall: Precision/Recall @ k (**P/R@k**). 

A **high precision** means:

- user looks at top results
- repository is large, with easy to find relevant pages
- imperfect recall is OK

A **high recall** means:

- users are not satisfied with only top-k documents. Think about patents, legal or medical searches.  

However, these are *set*-metrics, so they don't suit our purposes. We need to use them to a metric that is usefull for a retrieval system.

**Average Precision (AP)** is such a metric and calculated in the following way:
![AP](images/evaluation_average_precision.png)

AP is the area-under-the-curve of the Precision/Recall Graph:
![AP Graph](images/evaluation_AP_curve.png)

It gives you a score for a ranking and not just for your binary classification. You always devide by your total number of relevant documents.

#### Relevance as a non-Binary Classifier

##### (N)DCG: 

We can have more relevance labels other than just two (Bing for example has 7). Precision and Recall cannot handle this. A metric that *can* handle this is **Discounted Cumulative Gain (DCG)**.

Intuition behind DCG: when you find a document, how much *utility* does it give and how much effort does it take to find this relevant document? 

The forumla:
![DCG](images/evaluation_DCG.png)

Non-linear gain: the utility/usefullnes function. This function is exponential because when your document is relevant and at the top, it has alot more effect on the score.

Dicount: the effort to find the document

We need to normalize (range of 0~1) this function because we can then average over queries that have different numbers of relevant documents. We get **NDCG**. You normalize with: optimal DCG, aka the DCG of the documents were ranked optimally by their non-binary rank.

##### Model-based measure: browsing/positions-based models, RPB

Described how users interact with results, aka it's a *model of the user*.

It depends on the position of the document if the users goes to the next one. So the higher a document is, the more chance it has to be viewed. We get a metric called **Rank Biased Precision (RPB)** and is the expected utility at stopping at rank k.

The model:
![browsing mode](images/evaluation_browsing_model.png)

The metric:
![browsing mode](images/evaluation_browsing_model_metric.png)

Where $\theta=P(continuing)$. It expresses the probability of *reaching* rank k and *stopping* at rank k.

**How** do we choose $\theta$ or how is this metric better than NDCG? We can use the logs of our search engine and we can look at the clickthrough rates for every position. We can than choose a $\theta$ such that RBP matches the curve of the clickthrough rate best.

**Why** do we want this, why do we want a user based model? Because we *care* about users and want to align with them.

**What doesn't work** in RBP: it does not capture the relevance of a document on the position. When two documents equally relevant and are in 1st and 2nd place, we still would see a skewed probability to reach the document in the second position. This is not captured by this model.

##### Model-based measure: document utility, Cascade Model, Expected Effort

The cascade model simulates the probability of going to the next document based on the utility of the current document.

The model:
![Cascade Model](images/evaluation_cascade_model.png)
Where $\theta_2,\theta_1,\theta_0$ are the probabilities of moving to the next document when a document is highly, somewhat or not relevant.

The **main difference with RBP**: no moving on only depends on what you have seen so far.

$\theta$ is chosen differently this time and is pre-computed as such:

$\theta_i:=\frac{2^{(1-rel_i)}-1}{2^{\text{max rel}}}$

So the more relevant a document is, the more probability of stopping.

This gives rise to a new metric, **Expected Effort/ Expected Reciprocal Rank**:

$ERR=\sum_{r=1}^{n}(\underbrace{\prod_{i=1}^{r-1}(1-\theta_i)\theta_i}_{\text{prob of reaching rank r}})\frac{1}{r}$

$\theta$ is in the original paper pre-defined and unlike the **RBP** where we can compute them based on the CTR.

##### Novelty-based measures

There are also click and time model-based measures, which will not go into now.

##### Issues with previous measures, Session measures

All the previously mentioned measures have something called the **redundancy problem**: when we optimize for these measures, we get that the first relevant document contains usefull information but all documents after with the same information are worth less to the user. This is called **redundant information**.

We also have no **diversity** in the search results. A system does not know beforhand the user's intent. When a search result is **diverse** we cater to a space of possible user intents.

A **solution**: we gather different user intents, and then let annotators rank documents for each different intent. Every intent is treated as equal.

we get a new metric **$\alpha$-nDCG**: generalization of nDCG, that accounts for *novelty* and *diveristy*. Where $\alpha$

![alpha nDGC](images/evauation_alpha_ndcg.png)

There is **not a set way** to find intents.

All previous evaluation metric are defined on a **single query**, not a session or search history. However, we do want to optimize for a session. There exists **session based metrics**. In here we don't have query/document pairs but instead topic/collection pairs. You then need a metric that takes everything into account. It gives rise the the **Session DCG**. You have a score for each search in a session, and your final score is the sum of scores.

___

## Week 2: Lecture 4, Meta-Evaluation

### Comparing Metrics

Meta-Evaluations not discussed:

- based on query logs: 
- side-by-side: predicting choice when results are side-by-side
- discriminitive power: the powerfullness of a test (statistical significance)

How do evaluate objective functions? We focus on the **informativeness** of a metric: the ability to train a system when we use an objective function, aka how **informative** a objective function is to the system.

**Intuition**: tells us how good a metric for a specific problem is.

**Example**: when we have a system that only cares for P@1, and the model improves by putting a document that needs to be at position 1, from position 10 to 5 (good move), the metric P@1 is not informative. Since, both position 10 and 5, for the metric P@1 are equally bad.

#### Maximum Entropy Method

Don't understand quite well, may have to revisit.

**Framework/How** do we find a metric that is more informative? When we have the *value* of a metric, we should be able to tell with more certainty the probability of each document that it is relevant for a given query result. We do this by measuring randomness/**Entropy**, and want to maximaize Entropy under the constraint of our metric value.

### Collection Construction

When we have a problem, sometimes there is no collection (training, validation, test) of queries and relevant results. How do we build these collections?

We need three components:

1. Documents: the type of things we want to retrieve (film, image, webpage, etc.)
2. Queries: collecting queries from users and we can generate topics to put queries under if we have a search problem. If you have a filtering problem, you would have a different type of query.
3. Labels: if you have an extremely large collection, you cannot label each document. You want to have the labeled collection be reusable (= useable when we want to apply other objective functions). To have this guarantee, you need to label everything. There are no theoretical guarantees but only practical guarantees for solutions for this problem: competitions are organized. In these systems are built and create rankings for of documents. These rankings are cutt-off at depth K, and human judges judge each document as (non-)relevant. This is call** depth-K pooling**: ![Depth K Pooling](images/evaluation_depthk.png). For neural models you also have techniques called **Uniform Random Sampling** and **Stratief Random Sampling**. (Need to rewatch lecture for last two technqiues).

Take-away from **offline evaluation**:
![Offline evaluation take-away](images/evaluation_offline_takeaway.png)

## Week 2, Lecture 4: Query Processing

When we have gotten and cleaned our data, how do we retrieve it? When we get a query, what do we do? We do **query analysis** and **query processing**

### Query Analysis

The pipeline for query analysis/preperation is and should be **the same as the pipeline as you would process you documents**:

- Normalization: lowercasing/tokenization
- Spelling correction
- Segmentation
- Stemming
- Term expansion
- etc.

### Query Processing/ Ranking

We have two phases for big collections (10.000+): First- and Second-phase ranking.

**First-phase** ranking is simple ranking and semantic ranking and is done on the entire collection. We are left with ~10.000 documents.

**Second-phase** ranking is Learning to Rank (Complex Ranking) and this is done on the remaning ~10.000 documents.

If we only have a small collection you do not need two phases.

Overview: ![Ranking Overview](images/ranking_query_processing.png)

#### First-Phase Query Processing /  Simple Ranking

Steps:

1. **Matching**: Filtering the documents that contain the words of our query. This can be done by AND/OR operations: we want documents that contain ALL of the words in a query or document or document that at least contain ONE words of the query. **AND operation = Precision**, **OR operation = Recall**.
2. **Simple Ranking**: will cover this in (sub-)chapters.
3. Heap: Not menitoned during the lectures.

#### Simple Ranking: Term based Vector Space Model

***Represent* documents as vectors**. To *match* a document with a query we can measure the euclidian distance between the document vector and the query vector and get the: **cosine similarity**.

We say that each element in a vector corresponds to one term. We get the following possible vector representations:

- **One-hot-encoding**: 0's and 1's, simple.
- **Term frequency**: frequency of each word &rarr; more information than One-hot.![tf](images/tf.png)
- **Inverse document frequency**: how many document contain the specific term/word. We think: the more documents contain the specific word, the worse it represents a document. This leads us to take the inverse of this frequency. ![idf](images/idf.png)![idf example](images/idf_example.png)
- **TF-IDF**: a mix of term frequency and inverse document frequency: ![tf-idf](images/tf-idf.png)

#### Simple Ranking: Term based Language Modeling

***Represent* documents as probability distributions**. (Usually) using a unigram language model: a joint probability of the sequence of words in a document, which are independent from each-other. Example: ![Language model example](images/lm_example.png)

**In other words**: documents are represented as probability distributions, specifically as a multinomial distribution over words.

**Matching**: we can match a document with a query, both represented as distributions by

1. calculating the *KL-Divergence*
2. Query likelihood matching: ![Quey Likelihood Matching](images/query_llh_matching.png)

**Problem**: with this is when a word that occurs in a query occurs zero times in a document and we multiply it with another term, we get a probability of zero for the entire document.

**Solution**: (Laplace) Smoothing, Jelinek-Mercer Smoothing, Dirichlet Smoothing,

#### Simple Ranking: Term based BM25

We represent a document as a abstract score. The formula is abitrary to know. There is no intuition to be gathered here except that it stand for **Best Match 25**, it happend to be the 25th version and this worked the best.
___

## Week 3: Lecture 5, Semantic Based Retrieval

In the previous week, all the models are purely term based. The result is that we fail to capture *semantics*. In semantic based models, we still use the vector and distribution models, though we try to capture semantics.

### Simple Ranking: Topic based Modeling

In topic modeling we assume the following:

- Documents are a multinomial distribution over topics over words.
- Topics are a multinomial distribution over words.
- Topics themselves are a multinomial distribution.

So how is a document built up? A document has a length (number of words), imagine that these words are not yet filled in &rarr; they are placeholders. For each placeholders, we sample from the topics, then from the sampled topic we sample a word.

**The problem**: we don't know anything about the topics. How many are there? What are they about? It's a latent variable.

**Solution** Latent Dirichlet Allocation: since words and topics are both multionomial distributions, we use their priors: Dirichlet Distributions. The multionomial distributions are sampled from the Dirichlet Distributions. The number of topics is usually chosen either by hand or by (hierarchical topic) models.

**Choosing Parameters**: We choose random parameters, compute the likelihood and by using *Expectation Maximization* we eventually get the local optimal parameters. We have two steps:

1. *E-step*: define the expected value of the llh function, with resepct to the current estimates of the parameters.
2. *M-step*: find the parameters that maximize this quantity.

**Matching Documents to Queries**: it's the same as a language model where we transform our document and query to distributions and we calculate the *KL Divergence* or use *Query Likelihood Matching*.

**Intuition**: we try to incorporate the semantics/ dependencies between words. If a document and query share the same topics and also the same words, they must be very similar in meaning.

### Simple Ranking: Latent Semantic Indexing/ Analysis, Vector models

Again, documents are represented as vectors, and every word is a cell in that vector. In this method we use *Singluar Value Decomposition*.

Steps:

1. Perform SVD and low-rank approximation on our collection of documents. With this, we will throw away a lot of singluar values so we end up with a relatively small matrix (~1000 long and wide). This is now our **sementic space**
2. Given a document and query, we will represent them as vector in the obtained semantic space.
3. Perform **cosine similarity** on these vectors.


### Intermezzo: which technique works better

There is no real answer. Basically you have to implement a technique, test it and get the NDCG score. Then based on this you can decide which works best for the application.

### Simple Ranking: Nerual Models

#### Word Embeddings

Representing words as vector (Word2Vec). But we work with document in IR, not words. So how do we translate word embeddings to our IR space?

We can **avarage** the word embeddings.

**Problem**: the avarage of longer documents lose their meaning.

**Solution**: Document Embeddings

#### Document Embeddings

We still take the **avarage** of the word embeddings but we add a paragraph embedding to the avarage. This paragraph vector is computed by gradient decent. We get the following sheme:
![paragraph embedding](images/neural_models.png)

**Matching**: how do we match a document embedding to a query? The following way:
![document embedding matching](images/neural_model_matching.png)

___

## Week 3: Lecture 6, Offline Learning to Rank

Documents have more signals than just words to match it with queries. We can have things like popularity, language, clicks, etc. We can use machine learning to use these features and create rankings of documents.

### LTR Preliminaries

**Representation**: Documents and queries are represented as numerical vectors

**Prediction**: A ranking model takes in a vector and maps it from a vector to a real-valued score. Based on these scores, we sort the documents.

**Features**:

- Document-only: document length etc.
- Document Query Combination: BM25 etc.
- Query-only: Query length etc.

**Training methods**:

- Offline / Supervised LTR: learned from annotated data but it's hard to get the labels
- Online / Counterfactual: learn from user interactions, easy to get but hard to interpret.

### LTR Goals

![Offline LTR Goals](images/offline_LTR_goals.png)

First, we need a appropriate *loss function*. The main two loss functions in ML are:

1. Regresion
2. Classification

When we have scores we can use the **softmax function** in classification models or the **cross entropy** as our loss function of two probability distributions over a discrete set of events. We then use the cross entropy loss to optimize the model.

Now, how do we use the losses? We have multiple approaches

#### Pointwise Approach

We choose between two different loss functions:

1. **Regression loss** gives us a mean squared error loss
2. **Classification loss** gives use the softmax and cross entropy loss

**Problems**: this would not work on a large set of documents because of 

- **class imbalance**, many irrelevant vs. few relevant
- **query level feature normalization**, distribution of features differs greatly per query

These can be overcome (normalization, class re-weighing), but there is a much bigger issue: we try to score each document independently and then try to rank them, so **we are not optimizing the ranking quality**.

**Ranking is not a regression or classification problem**.

#### Pairwise Approach

Instead of looking just at the document-level, **we consider pairs of documents**. 

**Naive Implementation**: compare each pair of documents with each-other.

**Problems**: Quadratic Complexity during ranking time and we would need to pair-references have to be aggregated and can lead to paradoxical situations.

**Solution**: We don't change the model, we *change the loss function*. We get ![Pairwise Loss](images/pairwise_loss.png)

**Intuition**: Sum over all pairs where one document is more relevant than another. We take the difference in scores and put that into a loss function. This still has quadratic complexity but this is during training time, not ranking time.

**Possible Pairwise loss functions**: ![Pairwise Loss Functions](images/pairwise_loss_functions.png)

#### Pairwise Approach: RankNet

It's a pairwise approach. It interprets the score differences as probabilities. We the difference in scores is run through a sortof softmax function. When have two documents we compute this score/ probability for both orders of documents, and we then compute the cross-entropy &larr; Loss function. We get the following: ![RankNet Loss Function](images/RankNet_loss_function.png)

In the derivative of the loss, we have lambda's which act like forces pushing pairs of documents apart or together: ![RankNet Lambdas](images/RankNet_lambdas.png)

**Problems**: the probabilty function is not real and it treats every document pair as equally important. However, we care more about some documents than others and would prefer to see those correctly ordered then less important ones.

**Solution**: Listwise Approach

#### Listwise Approach: LambdaRank

The goal is to opitimze the ranking quality. We have metrics such as *NDCG*, *ERR*, *Precision*, etc. but how do we optimize for these metrics? They are non-differentiable metrics.

LambdaRank makes the following obervations:

- We do not need the costs of the loss functions, only the gradients: we do not care about the loss, only its gradient.
- Gradients should be bigger for pairs of documents that produces a bigger impact on the loss function (NDCG) by swaping positions: when we swap documents and get a better NDCG we know that swap is more important.

In LambdaRank we then do the following: ![LambdaRank Loss](images/LambdaRank_loss.png)

I.e. we multiply the lambda from RankNet with the difference of NDCG scores of swapping two documents. You can do this for all other metrics.

#### Other Approaches: ListNet, ListMLE

Using probabistic models for ranking, which are differentiable. (Should look at the lecture better for these methods.)


### LTR Recap
![LTR Recap](images/Offline_LTR_Recap.png)

___

## Week 4, Lecture 7: IR-User Interaction

How do users interact with systems?
  
- Item selection
- We give it queries
- Give attention (mouse movements, scrolling, clicks)
- Like/Favorite
- Timing between user actions
- Closing of browser
- Etc.

We mainly focus on clicks. They show the interest of the user. However, clicks are becoming less important due to position bias and sometimes it already shows the relevant information without the need to click.

There is no clear understanding if many/few clicks(interactions) are good or bad. Clicks and other interactions are **ambiguous**. Then, why do we need them? We can use them for:

- Evaluate IR system
- Improve IR system

### Online evaluation

#### A/B Testing

We have two different version of a new system. We assign a small percentage (~0.5%/0.5%) of users to each of the version and we measure the interactions we are interested in for a certain amount of time. If we want more clicks, the system that gets more clicks wins.

**Practical considerations**:

- Choosing metrics
- Control extraneous factors
- Estimate adequate sample size
- Choose time period T (~2+ weeks)
- Novelty Impact

**Pros**:

- Feedback from Real-users
- We don't need labeled data
- Can evaluate any change to the system

**Cons**:

- Variance between users
- Not very sensitive
- Needs many observations to get statistical significance

#### Interleaving

We have two different versions of a new system, but we mix the results of each system together for a given query. The user then sees the interleaved ranking (unknowing from which system it comes) and the user does it's thing.

If the user then clicked on more links of system A, then system A won and vice versa.

How results are mixed is called **mixing policy**.

How a system is judged better is called **scoring rule**.

**Pros**:

- Highly sensitive
- Users are not split up in different groups, so we need less observations

**Cons**:

- Evaluation ranking only
- only doucment-level metrics

### Interaction Models

Different types of models to model search interactions:

- Click models
- Models of mouse hovering
- Models of time between user actions

#### Basic Click Models

##### Position-based model

The probability of a click depends on the position.

There are multiple probabilities:

- Examination: if we read a result snippet, this **depends on the rank**.
- Attractiveness: if we like the snippet after reading, this **doesn't depend on the rank**.
- Click: if we have examined a snippet and are attracted by it.

We get: ![Position Based Click Model](images/click_model_position_based.png)

**Pros**:

- Examination
- Attractiveness

**Cons**:

- It only considers the position of a result
- Doesn't model that we may have already found good information

##### Cascade model

The probability of a click depends on what happens before

We do not skip results, so it assumes a user goes over every document in the list. If a click happens, we stop. If a click doesn't happen, we continue examining

We get the following probabilities:![Click Model Cascade](images/click_model_cascade.png)

Now the examination of the next document depends on the previous.

**Pros**:

- Examination at document position r depends on the examinations and clicks of all docouments above position r

**Cons**:

- Only one click is allowed in this model

#### Parameter Estimation for Click Models

A model is good when it reflects the data. We can compute the **maximum likelhiood estimation** or we can de **expectation-maximization** (E-step and M-step).

#### Applications of Click Models

Possible applications:

- simulating user
- using parpameters as features in ranking
- We can get evaluation metrics

Based on the probability of a click, we can get **Expected Reciprocal Rank (ERR)**:

![ERR](images/ERR.png)

where $1/r$ is that we are satisfied at rank $r$. From the click model, we can get $P(S_r = 1)$, which is the probability that we are satisfied at rank $r$.

___

## Week 4, Lecture 8: Counterfactual Learning to Rank

There are limitations with annotated datasets to train ranking systems:

- Datasets are static
- expensive and time consuming to get
- we are not sure if judges agree with the users
- can be unethical if the data is privacy sensitive (email for example)
- impossible for small scale problem (personalization for example)

They are used and useful, but not everybody has access to them and are often misaligned with true user preferences. So we need an **alternative of LTR**.

### Learning from User Interactions

We only look at *clicks*.

Pros:

- When you have users, it's free
- User behavior is indicative of their preferences

**Cons**:

- interaction only give implicit feedback

There are **difficulties**

- **Noise**
  - Users click for unexpected reasons
  - Click occur not because of relevancy
  - Click don't occur eventhough something is relevant
- **Bias**
  - Position Bias: high ranked documents get more attention
  - Item selection bias: interactions ar limited to presented documents
  - Presentation bias: think of netflix where it has a trailer has a very large display while other movies get a small thumbnail.
  - Many more

We will focus on **position bias** and present a method called **unbiased learning to rank**.

### Unbiased Learning to Rank

**Goal**: 

- optimize a ranker w.r.t. what the users think is relevant.
- Avoid being biased by other factors the influence interactions (truly unbiased doesn't exist, we usually say which biases we remove)

There are two approaches: 

- **Online LTR**
  - Learn by directly interacting with users
  - Handle biases through randomization of displayed results
- **Counterfactual LTR**
  - Learning from historical interactions
  - Handle biases through using a model of user bevaiour

### Counterfactual Learning to Rank

#### Evaluation

How can we evaluate a ranker before deploying it and without annotated data? With counterfactual evaluation we have **historical interaction data**(clicks) and use this to evaluate a new ranking function.

We do not have full information, aka the true relevance labels for each document. **We only have clicks**. There are two problems with this:

1. A click is a biased and noisy indicator
2. A missing click does not necessarily mean that a document is not relevant

We get something like this:

![Counterfactual list of clicks](images/counterfactual_list.png)

We can remove noise but hardly remove bias. To do this, we must first decompose a click. A click exists out of **examination** and **relevance**. We get: 
![Click Decomposed](images/click_decomposed.png)

In other words: if they have observed something and then thinking if it is relevant or not.

##### Naive Evaluation

Assume that clicks are a unbiased relevance signal. But what happens if we use this metric?

This estimator will weigh document according to their examination probabilities.

**Intuition**: the more likely a document is to be observed, the more weight it has.

**Problem/Effect** on evaluation process: we get a self-fulfilling profecy. When we have put a document very high during logging (when the click data was collected) then it will gather, because of **position bias**, a lot of clicks. This model will then again put the same document very high and leads to documents that aren't relevant to be perceived as relevant. And so the estimate will be off

**Solution**: inverse propensity scoring

##### Counterfactual Evaluation: Inverse Propensity Scoring (IPS)

Why is it called counterfactual? We pretend a click happened in a world where every document is always observed.

We get the following function which gives an **unbiased estimate**:
![Unbiased Estimate](images/unbiased%20estimate.png)

**Intuition**: for every click we are inversely weighing it to the examination probability.

#### Propensity-weighted Learning to Rank

We have the estimates (read: objective functions) from the evaluation, but we **cannot optimize the directly**. They are non-differentiable, so we cannot do SGD.

**Solution**: we can create a **differentiable upper bound**

1. A rank function gives a rank given a document
2. We describe this as a sum over all other documents in the ranking that have a higher or equal score (example: if you are doc. no. 2 you get a score of 2 because there is one document above you and yourself, aka you get your rank).
3. We then use a **hinge-loss** to upperbound your rank (a function that always returns something greater than your rank).
4. We get a function that **is differentiable and great than your rank**.

##### Possible Different Metrics

- Sigmoid-like bound
- Average Relevance Position (unbiased estimator and upper bound)
- DCG (unbiased estimator and lower bound)

#### Summary

![cfltr](images/cfltr_summary.png)

### Estimating Position Bias

How do we actually measure a document being observed? How do we actually measure the position bias?

We make the assumption that the **observation probability** only depends on the rank of a document. The probability of obersving a document is called the **propensity score**.

Different methods of estimating this:

#### RandTop-n Algorithm

Every document is equally likely to occur at each rank, then the only thing that matters for a user to click on the document is that rank. We get the following:
![Rand N](images/randn.png)

**Problem**: it gives a very horrible user experience.

#### RandPair

Choose a pivot at rank k, and swap with a random other document at that rank.

#### Intervention Harvesting

Based on A/B testing. 
![Intervention](images/interventiontesting.png)

#### Jointly Learning and Estimating

(Have to go back to the lectures, method didn't seem very popular)

____

## Week 5, Lecture 9: Online LTR

A older way to deal with position bias. Originally A/B testing is a good way to counter position bias.

**Pros**:

- Straightforward
- Able to test many aspects of user behaviour

**Cons**:

- Inefficient (requires a lot of user data)
- You have to test for a long time
- You need to be able to recognize individual users

**Online Evaluation** uses the same principle of randomization as A/B testing, but it's more efficient.

### Online Evaluation, Evaluation between two systems

The reason why these following method are called *online* is because we have **control over what to display** to the users. These algorithms decide both what to **display** while also **learning from clicks**. They are more efficient because of the control over what data is gathered.

Some basic requirements:

![online requirements](images/online_requirements.png)

#### Online Evaluation: Team-Draft Interleaving

![online interleaving](images/online_interleaving.png)

The ranker that receives the most clicks is seen as the better ranker (A clicks are equally weighted). To filter out noise you have to repeat the proces of interleaving a lot of times.

**Problem**: some times the systems create a tie and we cannot tell which system is better

#### Online Evaluation: Probablistic Interleaving

It uses **fidelity conditions**: if we are sure one ranker is better than another, we want the evaluation method to agree.

We treat **rankers as probability distributions** over a set of documents. In other other words: we get a ranking of documents and use a "trick" to create a distribution for a ranker, where each document is assigned a certain probability. We get something like:

![Probablistic Interleaving](images/interleaving_prob.png)

Then, during the interleaving phase, when a ranker is chosen, a document is sampled from that distribution (and not the next best document).

We can prove that this method has **fidelity**

The final method:

![Probablistic Interleacing Method](images/prob_interleaving_method.png)

**Pros, Cons, Properties**:

![Prob Interleaving Properties](images/prob_interleaving_properties.png)

(the problems were not very clear to me)

#### Online Evaluation: Optimized Interleaving

It sees **interleaving as an optimization problem**, aka we can optimize it. They saw a problem with probablistic interleaving in that it allowed *every* possible document to be at the higher ranks. This method allows only top documents at the higher rank.

**So which interleavings can we do?**: They only interleavings allows are documents which should be on top of certain other documents, should always be on top when interleaved.

**Which scoring function do we have? / How do we weight each click?**: 

1. Linear Rank Difference
2. Inverse Rank Difference

**What do we do against Position Bias?**: We assume that we only have position bias. You multiply the probability of a click with a score based the weight of a click. Example:
![Optimized Interleaving Example](images/optimized_interleaving_example.png)

This becomes a *linear optimization problem*, in which **each interleaving get a probablity**.

**Intuition**: we use randomization to make sure that with random clicks we expect ther would be no winner between two systems. So that means when clicks are not random, we can say with certainty that this is not because of chance.

**Properties**:
![Optimized Interleaving](images/optimizer_interleaving_properties.png)


### Online Learning to Rank: how to optimize Rankers

#### Dueling Bandit Gradient Descent (DBGD)

**Intuition**: if online evaluation can tell us if a ranker is better than another, then we can use it to find an improvement of our system.

**How**: sampling model variants and comparing them with interleaving, then the gradient of a model can be estimated w.r.t. user satisfaction.

**Example**: ![Bandit](images/bandit_example.png)

We have to move smoothly through the space, aka we take small steps through the space and eventually we get a optimal set of parameters for our model.

#### Reusing Historical Interactions

**Intuition**: like bandit gradient descent we explore for the optimal parameters but it this is based on what we have seen before (click history).

**How**: sample a large number of rankers and create a candidate set, and use historical interactins to select the most promising candidate for DBGD. So in the previous example image, the model (the arrows in the space) will be sampled from a large collection and looks at previous clicks, and decides which model is best.

**Problems**: for all Online learning, early stopping is impossible (since it's online) and this method degrades in performance over time. It degrades because it does not allow for a ranker to thoroughly explore it's search space and we are not sure it was a good decision to exclude a ranker or not.  

#### Multileave Gradient Descent

**Intuition**: same as DBGD except we now use *multileaving*, which allows for multiple rankers to being compared simultaneously.

**Example**:
![Multi Leaving](images/multileaving.png) 

This method is doing it efficiently but it is not an improvement per se. It is at the cost of computational cost.

#### Problem of Dueling Bandit Gradient Descent

- Performance at convergence is worse than offline approaches, even without noise and position bias.
- The assumption that there is a single optimal model is in practice untrue
- The assumption that the utility space is smooth w.r.t. to the model weights is in practice untrue
- Neural Models have no advantage over linear models

#### Pairwise Differentiable Gradient Descent

Different approach than DBGD.

**Intuition**: by taking a *pairwise approach*, maybe you can make it unbiased while still being *differentiable*. Without relying on online evaluation methods (interleaving) and sampling of models.

**How**: by optimizing a ranking model that models documents as a distribution (*Plackett Luce ranking model*). So a scoring model where every document is put through a softmax function with each score. This gives us a *confidence* for each document.

Then, the *pairwise document preference* comes from the fact that when they see a click, they assume a user has seen all documents above the one clicked, and the one below (this comes assumption comes from eye tracking studies). 

However, **this approach is biased**. To correct this they use *randomization*, by swapping two documents. If then, two documents are equally likely, the user will click on both documents at a certain rank. **In other words** when we expect documents that are equally likely and swapped, you would see roughly the same number of clicks at all swapped positions (only looking at the position of the swap).

Example: 
![Equally Likely](images/equally_likely.png)

**Method Summary**
![Pairwise GD](images/pairwise_gd.png)

**Visualization**
![Pairwise Vis](images/Pairwise_vis.png)

#### Online Models Compared

**Emperical Conclusion**:
![Online Compared](images/online_compared.png)


#### Future for Online LTR

DBGD appears to be lacking for optimizing ranking systems

Novel approaches have more potential like Pairwise Diff. GD.

**Emperical**:
![counterfact vs online](images/online_vs_counterfactual.png)

**Theoretical**:
![Counterfact vs online theory](images/counterfact_vs_online_theory.png)

#### Conclustion Online LTR

![Online LTR conclusion](images/onlineLTR_conclusion.png)

___

## Week 5, Lecture 10: Entity Search

Techniques in the previous lectures are applied in the next lectures.

**What is an Entity**: uniquely identifiable thing or object (usually people, companies, places). 

**What is special about them**: 

- They are connected/ have relationship to other entities
- They represent something real
- They have attributes
- They have a type

**What is the difference between an Entity and a document?** Document are unstructured while entities are very structured.

Entities can be put into *knowledge graphs*, which is just a repository of entities and their relations and attributes represented as a graph.

Most searches (~41%) are entity searches. Usually in websearch, you run two pipelines because of this. One is the regular search as previously discussed, the second one will be discussed here.

**Challanges for Entity Retrieval**: it is hard to represent entities. We cannoy the same techniques as for document. Also entities are usually short data structures.

**Which techniques from "regular" IR can we keep and which do we have to change**:

- Evaluation: Keep &rarr; same evaluation methods hold
- Learning to Rank: Keep &rarr; only slight change
- IR - User interaction: Keep &rarr; only change may be the interface
- Document Representation & Matching: **Change**

### Entity Representation

Trying to build a textual representation (document).

#### 1-Field Entity (as Probability Distr over words/categories)

You put all the information in attributes in a single field.

#### 2-Field Entity (as Probability Distr over words/categories)

You have a title field (name, title, label) and the content field (other attributes).

#### 3-Field Entity (as Probability Distr over words/categories)

Same a two field but now the connections go into a third "outgoing links" field.

#### Field Entity, how to use

If you want to retrieve an entity structured this way we can, for example, take the BM25, for 3 values: query, title and content/attributes. And then we do what we normally do to retrieve documents.

How can we use all the information of an entity to the fullest? Because we have *term based similarity but also topic based similarity between entities and queries*.

![Entity Example](images/entity_example.png)

We can combine them by summing or multiplying:

![Entity Retrieval](images/entity_retreival.png)

**Intuition**: we can users standard IR techniques to retrieve entities but we need an extra step to convert them to documents.

#### Knowledge Graph as Tensor (as Vector embedding)

We can represent entities as a tensors. It is built-up as such:

- n = all entities
- m = relations between entities. These relations can be anything (born in, married to, works at, studied at, etc.)

![Entity Tensor](images/entity_as_vector.png)

**What do we do with this?**: we decompose the tensor (SVD or with a neural net) and get a *representation* of one entity. Once we have one matrix representation of one entity, we can use the *cosine similarity*.

### Learning to Rank Entities

Learning to Rank is essentially the same if we have the features. However, some features in entities are different:

- connections: % of outgoing links vs ingoing links for example
- page rank
- attributes (if entities share the same attributes)
- number of categories
- popularity and importance from a wiki page (using external information)
- etc.

___

## Week 6, Lecture 11: Conversational Search


___

## Week 6, Lecture 12
