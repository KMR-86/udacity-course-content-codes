from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer()
str1="hi i am mushi and i love learning machine learning"
str2="hi i am preata and i love travelling to useless places"
str3="i do not love doing anything"
email_list=[str1,str2,str3]
bag_of_words=vectorizer.fit_transform(email_list)
bag_of_words=vectorizer.transform(email_list)
print(bag_of_words)
print(vectorizer.vocabulary_)
print(vectorizer.vocabulary_.get("love"))

#output analysing
'''
(0, 5)	1 
this means that 0th string has the word that is indexed as the 5th word only 1 time
the word indexing is random 
hi is the 5th word.

6th word is learning and it is present twice in the 0th string
so thats why we get 
(0 , 7)   2 

'''