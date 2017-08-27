
# coding: utf-8

# ### TagRank for product data
# 
# #### Preprocess
# - Download "Product Keyword Extraction - Data" as "data.csv"



import pandas as pd
from preprocessing.preprocess import preprocess, tokenize
import networkx as nx
import operator

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




from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix




class TagRank(object):
    def __init__(self,window=3):
        self.window = window
        self.graph = None
        
    def _vectorize(self,tokens):
        """
        Vectorize a list of tokens
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
        Find the cooccurrence matrix of query and title
        """
        data = []
        row = []
        col = []
        if event == "Impression": val = 2.
        else: val = 3.
            
        for wt in title:
            for wq in query:
                data.append(val)
                row.append(self.vocab[wt])
                col.append(self.vocab[wq])
                
                data.append(val)
                row.append(self.vocab[wq])
                col.append(self.vocab[wt])
        return csr_matrix((data,(row,col)),shape=(len(self.vocab),len(self.vocab)))
        
        
        
    def _construct_sparse_matrix(self,events,queries,titles):
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
        self._construct_sparse_matrix(events,queries,titles)
        self.graph = nx.from_scipy_sparse_matrix(self.coo_matrix)
        self.scores = nx.pagerank(self.graph)
        
    def get_ranking(self,tokens,topK=5):
        rankings = {i:0. for i in range(len(self.vocab))}
        for w in tokens:
            i = self.vocab[w]
            rankings[i] += self.scores[i]
#             for n in self.graph.neighbors(i):
#                 rankings[n] += self.scores[n]
        rankings = sorted(rankings.items(), key=operator.itemgetter(1),reverse=True)
        results = [(self.i2w[i],score) for i,score in rankings[:topK]]
        return results




tagrank = TagRank()




get_ipython().magic(u'time scores = tagrank.build_graph(product_log.event, product_log.query_tokens, product_log.title_tokens)')




def print_ranking(i):
    print 
    print product_log.product_name.iloc[i]
    print "-------------------------------"
    rankings = tagrank.get_ranking(product_log.title_tokens.iloc[i])
    for w,score in rankings:
        print w,score




for i in np.random.choice(range(product_log.shape[0]),replace=False,size=10):
    print_ranking(i)


# ### TO DO
# - Use TFIDF to lower the importance of repeated words, such as "現貨"
# - Apply direction into the graph (from product_name to query)






