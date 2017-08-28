
# coding: utf-8

# ### TagRank for product data
# 
# #### Reasons for using Graph-based model
# The target of this task is to find high-level words that are descriptive of the product.
# Graph-based model can:
# 
# 1) Identify general words, as such words has more connections in the graph.
# 
# 2) Identify general words that are descriptive of the product by looking at neighbour nodes in the graph.
# 
# It is able to identify the semantic relationship between words rather than just choosing words from the product names.
# 
# #### Data Format
# 
# |product_name|category|query|event|date|
# |------------|--------|-----|-----|----|
# |--- X 10 --- 七色 多層次搭配 圓下擺 LAYERED 素面 無袖背心 打底|Male Fashion|無袖|Impression|31/7/17|
# 
# #### Preprocess
# ##### 1.Clean up data
# - Download "Product Keyword Extraction - Data" as "data.csv"
# - Remove duplicate records
# - Remove "event" entries from the data
# - Remove wrong product names (#NAME? and #ERROR!)
# 
# ##### 2.Text preprocessing (preprocessing/preprocess.py)
# - preprocess the data to remove punctuations.
# - Tokenize the text using jieba, and keep only ['ns', 'n', 'vn', 'v','eng'] tags.
# 
# #### TagRank
# ##### 1. Construct a graph based on the tokens
# Each token in text is considered as a node in the graph. The connectivity of the graph consists the following cases:
# - Two tokens are connected if they appear in the same product name (edge weight = 1.), and
# - tokens in the product name are connected to tokens in the query (If event == "Impression", edge weight = 2. If event == "Click", edge weight = 3.) Note: the token in product name point to tokens in the query.
# 
# ##### 2. Run page rank over the graph
# The page rank algorithm will give the scores (the importance) of each node.
# 
# ##### 3. Find important word for a given text
# The TagRank algorithm will run over the neighbours of every node to find the representative tokens for the given text.
# 
# Suppose the $t_i$ stands for the $i^{th}$ token of a given text. The importance of neighbours of $t_i$ are:
# 
# $$n_i^j = (1-d) + d* \frac{w(i,j)} {\sum_{k}w(i,k)},$$
# where $n_i^j$ is the $j^{th}$ neighbour of $t_i$, and $d$ is the damping factor.
# 
# The TagRank algorithm sum all the tag scores and ranks nodes in the graph according to the score.



import pandas as pd
import numpy as np
from preprocessing.preprocess import preprocess, tokenize
import networkx as nx
import operator
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.cross_validation import train_test_split

# load data and preprocess
product_log = pd.read_csv("data.csv",sep="\t",names=["product_name", "category", "query", "event", "date"])

print product_log.shape, product_log.drop_duplicates().shape
product_log = product_log.drop_duplicates()
product_log = product_log[product_log.event!= "Event"]
product_log = product_log[(product_log.product_name!="#NAME?") & (product_log.product_name!="#ERROR!")]
product_log.head(5)




# preprocess the data
product_log["query"] = product_log["query"].map(preprocess)
product_log.product_name = product_log.product_name.map(preprocess)

# tokenize
get_ipython().magic(u'time titles = product_log.product_name.map(lambda x: list(tokenize(x)))')
get_ipython().magic(u"time keywords = product_log['query'].map(lambda x: list(tokenize(x)))")
product_log["query_tokens"] = keywords
product_log["title_tokens"] = titles




product_log[["event","query_tokens","title_tokens"]]




product_log_train, product_log_test = train_test_split(product_log,test_size=0.2)
product_log_train.shape, product_log_test.shape




class TagRank(object):
    """
    Construct the TagRank graph for given data
    """
    def __init__(self,damping_factor=0.85):
        self.graph = None
        self.damping_factor = damping_factor
        
    def _vectorize(self,tokens):
        """
        Vectorize a list of tokens
        
        Inputs:
            @tokens: a list of tokens to vectorizer
        
        Output:
            A scipy sparse matrix of shape (vocabulary_size,1)
        """
        data = []
        row = []
        col = []
        for w in tokens:
            data.append(1.)
            row.append(self.vocab[w])
            col.append(0)
        return csr_matrix((data,(row,col)),shape=(len(self.vocab),1))
    
    def _query_title_vec(self,query,title,event):
        """
        Construct the cooccurrence matrix for product_name to query. 
        The edge weight from product_name to query is either 1.5 or 2.
        The edge weight from query to product_name is 1.
        
        Inputs:
            @query: the tokens of a user query
            @title: the tokens of a product name
            @event: event type. The value can be "Impression" or "Click"
            
        Output:
            The co-occurrence matrix in scipy sparse format
        """
        data = []
        row = []
        col = []
        if event == "Impression": val = 1.2
        else: val = 1.5
            
        for wt in title:
            for wq in query:
                data.append(val)
                row.append(self.vocab[wt])
                col.append(self.vocab[wq])
                
                data.append(1.)
                row.append(self.vocab[wq])
                col.append(self.vocab[wt])
        return csr_matrix((data,(row,col)),shape=(len(self.vocab),len(self.vocab)))
        
        
        
    def _construct_sparse_matrix(self,events,queries,titles):
        """
        Construct the cooccurrence matrix for a given record. 
        The co-occurrence matrix is added up by two elements: 
        1)The co-occurrence matrix of a given product name, and 
        2)The co-occurrence matrix from product_name to query.
        
        Inputs:
            @query: the tokens of a user query
            @title: the tokens of a product name
            @event: event type. The value can be "Impression" or "Click"
            
        Output:
            The co-occurrence matrix in scipy sparse format 
        """
        # construct vocabulary
        vocab = set([w for sentence in (queries + titles) for w in sentence])
        vocab = {w:i for (w,i) in zip(vocab,range(len(vocab)))}
        self.vocab = vocab
        self.i2w = {i:w for w,i in vocab.iteritems()}
        self.coo_matrix = csr_matrix(([],([],[])),shape=(len(self.vocab), len(self.vocab)))
        
        # construct sparse matrix
        row = []
        col = []
        data = []
        for e,q,t in zip(events,queries,titles):
            vec = self._vectorize(t)
            coo_matrix = (vec * vec.T) / len(self.vocab)
            self.coo_matrix += coo_matrix
            
            self.coo_matrix += self._query_title_vec(q,t,e)
        self.coo_matrix /= len(titles)
        
        
    def build_graph(self,events,queries,titles):
        """
        Build the TagRank graph.
        
        Inputs:
            @events: list of events
            @queries: list of tokenized queries
            @titles: list of tokenized product names
        """
        self._construct_sparse_matrix(events,queries,titles)
        self.graph = nx.from_scipy_sparse_matrix(self.coo_matrix)
        self.scores = nx.pagerank(self.graph)
        
    def get_ranking(self,tokens,topK=5):
        """
        Get top ranked tokens from the TagRank graph for a given list of tokens.
        
        Inputs:
            @tokens: list of tokens for a product name
            @topK: top K nodes to retrieve from the graph. Default=5
            
        Output:
            Top K nodes and related score from the graph
            [(node,aggregated_score)]
        """
        rankings = {i:0. for i in range(len(self.vocab))}
        for w in tokens:
            try:
                i = self.vocab[w]
                rank_sum = sum(self.scores[n] for n in self.graph.neighbors(i))
                for n in self.graph.neighbors(i):
                    rankings[n] += (1-self.damping_factor) + self.damping_factor * (self.scores[n] / rank_sum)
            except:
                pass
        
        rankings = sorted(rankings.items(), key=operator.itemgetter(1),reverse=True)
        results = [(self.i2w[i],score) for i,score in rankings[:topK]]
        return results




# build TagRank model for training data
tagrank = TagRank()
get_ipython().magic(u'time scores = tagrank.build_graph(product_log_train.event, product_log_train.query_tokens, product_log_train.title_tokens)')


# ### Evaluation of the model 



get_ipython().magic(u'time predicted_tags = [np.array(tagrank.get_ranking(title))[:,0] for title in product_log_test.title_tokens]')
def calculate_nhits(pred_tags,true_tags):
    """
    Find number of hits of the predicted results
    """
    return len(set(pred_tags).intersection(true_tags))
n_hits = map(lambda (p,t): calculate_nhits(p,t), zip(predicted_tags, product_log_test.query_tokens.values))




import matplotlib.pylab as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

plt.boxplot(n_hits)
plt.title("Number of hits at top 5 tags")




print "average number of hits at Top 5: ", np.average(n_hits)


# ##### Print some results for test data



def print_ranking(i):
    print 
    print test_data.product_name.iloc[i]
    print "-------------------------------"
    rankings = tagrank.get_ranking(test_data.title_tokens.iloc[i])

    for w,score in rankings:
        print w,score
    print "----------END------------------"




tagrank = TagRank()
get_ipython().magic(u'time scores = tagrank.build_graph(product_log.event, product_log.query_tokens, product_log.title_tokens)')




# load test data
test_data = pd.read_csv("test.csv",sep="\t")
test_data["product_name"] = test_data["Product Name"].map(preprocess)

# tokenize
get_ipython().magic(u'time titles = test_data.product_name.map(lambda x: list(tokenize(x)))')
test_data["title_tokens"] = titles
test_data.head(5)




# Sample results
for i in np.random.choice(range(test_data.shape[0]),replace=False,size=10):
    print_ranking(i)




keywords = []
for title in test_data.title_tokens:
    ws = tagrank.get_ranking(title,topK=5)
    keywords.append(" ".join([w[0] for w in ws]))
test_data["Keywords"] = keywords




test_data.drop(["product_name", "title_tokens"],axis=1).to_csv("results.csv",sep="\t",index=False,encoding="UTF-8")


# ### Future Work
# - Use TFIDF to lower the importance of repeated words.
# - Include window size for computing co-occurrence matrix.
# - Include customized words for jieba tokenizer ("褲" etc.).






