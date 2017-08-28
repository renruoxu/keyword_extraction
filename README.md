### TagRank for product data
Inspired from [TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf), the TagRank method a novel method which applies graph-based method for keyword extraction. 

### Files
```Bash
-- preprocessing/
	|-- __init__.py
	|-- preprocess.py #preprocessing functions
-- TagRank.ipynb # TagRank model for keyword extraction
-- TagRank.py # TagRank model in Python format
-- data.csv # data downloaded as a CSV file
-- requirements.txt # pre-requisitions of the model
```

#### Reasons for using Graph-based model
The target of this task is to find high-level words that are descriptive of the product.
Graph-based model can:

1) Identify general words, as such words has more connections in the graph.

2) Identify general words that are descriptive of the product by looking at neighbour nodes in the graph.

It is able to identify the semantic relationship between words rather than just choosing words from the product names.

#### To run the scripts
```
pip install -r requirements.txt
jupyter notebook
```
Run the notebook named TagRank.ipynb

### Model Descriptions
The following sections describe the TagRank model for keyword extraction.

#### Data Format

|product_name|category|query|event|date|
|------------|--------|-----|-----|----|
|--- X 10 --- 七色 多層次搭配 圓下擺 LAYERED 素面 無袖背心 打底|Male Fashion|無袖|Impression|31/7/17|

#### Preprocess
##### 1.Clean up data
- Download "Product Keyword Extraction - Data" as "data.csv"
- Remove duplicate records
- Remove "event" entries from the data
- Remove wrong product names (#NAME? and #ERROR!)

##### 2.Text preprocessing (preprocessing/preprocess.py)
- preprocess the data to remove punctuations.
- Tokenize the text using jieba, and keep only ['ns', 'n', 'vn', 'v','eng'] tags.

#### TagRank
##### 1. Construct a graph based on the tokens
Each token in text is considered as a node in the graph. The connectivity of the graph consists the following cases:
- Two tokens are connected if they appear in the same product name (edge weight = 1.), and
- tokens in the product name are connected to tokens in the query (If event == "Impression", edge weight = 2. If event == "Click", edge weight = 3.) Note: the token in product name point to tokens in the query.

##### 2. Run page rank over the graph
The page rank algorithm will give the scores (the importance) of each node.

##### 3. Find important word for a given text
The TagRank algorithm will run over the neighbours of every node to find the representative tokens for the given text.

Suppose the $t_i$ stands for the $i^{th}$ token of a given text. The importance of neighbours of $t_i$ are:

$$n_i^j = (1-d) + d* \frac{w(i,j)} {\sum_{k}w(i,k)},$$
where $n_i^j$ is the $j^{th}$ neighbour of $t_i$, and $d$ is the damping factor.

The TagRank algorithm sum all the tag scores and ranks nodes in the graph according to the score.
