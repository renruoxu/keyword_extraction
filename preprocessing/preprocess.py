import re
import jieba.posseg as pseg
# preprocessing
def preprocess(string):
    """
    preprocess the string to remove punctuations
    
    Inputs:
        @string: a text string
        
    Outputs:
        A cleaned string
    """
    string = re.sub(r"[\^&%$#@><+\-*.,!?;:/~\(\)]+","",string.lower())
    return string

def tokenize(string,allow_pos_tags = ['ns', 'n', 'vn', 'v','eng']):
    """
    Tokenize a given string with a set of allowed POS tags
    
    Inputs:
        @string: a text string
        @allow_pos_tags: a list of allowed POS tags. Default is ['ns', 'n', 'vn', 'v','eng']
        
    Return:
        @A generator of tokens
    """
    words = pseg.cut(string)
    for r in words:
        if r.flag in allow_pos_tags:
            yield(r.word)