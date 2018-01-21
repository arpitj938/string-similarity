from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import jellyfish
import nltk
from nltk.util import ngrams
from ngram import NGram
from sklearn.metrics import jaccard_similarity_score

def word_grams(words, min=1, max=4):
    s = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s

tfidf = TfidfVectorizer(stop_words = 'english')
string =[ """There are many algorithms for checking string simlarity""","""string simlarity can be measured by many algoriths"""
,"""Machine learning include many algorithms for string simalrity""","""cosine simlarity is one of the algorithm used for string simlarity"""]
y = tfidf.fit_transform(string)
# tfidf_matrix = [x for x in y]
y_array = y.toarray()
string_mat = [x  for x in y_array]
# print string_mat
sim_arry1 = cosine_similarity(y_array)
print 'cosine similarity',sim_arry1[0]
from nltk.tokenize import word_tokenize
string_token = [word_tokenize(s) for s in string]
# print jellyfish.levenshtein_distance('Siri','Siri')
sim_arry2 = [1.0-jellyfish.levenshtein_distance(unicode(string[0]),unicode(s))/((len(string[0])+len(s))/2.0) for s in string]
print 'levenshtein', sim_arry2
sim_arry3 = [1.0-jellyfish.hamming_distance(unicode(string[0]),unicode(s))/((len(string[0])+len(s))/2.0) for s in string]
print 'hamming', sim_arry3
sim_arry4 = [1.0-jellyfish.damerau_levenshtein_distance(unicode(string[0]),unicode(s))/((len(string[0])+len(s))/2.0) for s in string]
print 'dameru', sim_arry4
sim_arry5 = [jellyfish.jaro_distance(unicode(string[0]),unicode(s)) for s in string]
print 'jaro' ,sim_arry5
sim_arry6 = [jellyfish.jaro_winkler(unicode(string[0]),unicode(s)) for s in string]
print 'jaro winkler',sim_arry6
sim_arry7 = [jellyfish.match_rating_comparison(unicode(string[0]),unicode(s)) for s in string]
print 'match rating comparison',sim_arry7
# tokens = word_tokenize([string])
# print(string_token)
# print tfidf_matrix

# print(y.toarray()
ngram_array = [word_grams(s.split(' ')) for s in string]
# print ngram_array
n = NGram()
# print list(n.split(string[0]))
ngram_array = [ list(n.split(s)) for s in string]
# print ngram_array
sim_arry8 = [NGram.compare(string[0].lower(), s.lower(), N=4) for s in string]
print 'ngram',sim_arry8

def jaccard_distance(a,b):
	# print a, b
	inter_len = float(len(list(a.intersection(b))))
	union_len = float(len(list(a.union(b))))
	return inter_len/union_len
# print list(ngram_array[0].intersection(ngram_array[1]))
sim_arry9 = [jaccard_distance(NGram(ngram_array[0]),NGram(s)) for s in ngram_array]
print 'jaccard',sim_arry9
def average_of_all(sim_arry1,sim_arry2,sim_arry3,sim_arry4,sim_arry5,sim_arry6,sim_arry7,sim_arry8,sim_arry9):
	ans = []
	for i in range(len(string)):
		sum_of_all = sim_arry1[0][i]+sim_arry2[i]+sim_arry3[i]+sim_arry4[i]+sim_arry5[i]+sim_arry6[i]+sim_arry7[i]+sim_arry8[i]+sim_arry9[i]
		ans.append(sum_of_all/9.0)
	return ans

print average_of_all(sim_arry1,sim_arry2,sim_arry3,sim_arry4,sim_arry5,sim_arry6,sim_arry7,sim_arry8,sim_arry9)