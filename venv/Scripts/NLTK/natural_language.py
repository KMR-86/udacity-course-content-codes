import nltk

#stopwords code

from nltk.corpus import stopwords
sw=stopwords.words("english")
print("number of stopwords",len(sw))
print(sw)
sw.sort()
print(sw)
#stemmer code

from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer('english')
print(stemmer.stem("responsible"))
print(stemmer.stem("intelligent"))
print(stemmer.stem("bcdef"))  #meaningless words stay the same