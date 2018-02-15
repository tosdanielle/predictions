import pandas

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC

from textblob import TextBlob

def load_dataset():
    messages  = pandas.read_csv("spam.csv", encoding='latin-1')
    messages = messages.rename(columns={"v1":"label", "v2":"content"})
   
    contents_train, contents_test, labels_train, labels_test = train_test_split(messages['content'],
                                                                                messages['label'],
                                                                                test_size=0.2)
        
    print "Total messages:", messages.shape[0]
    print "Training set contains", contents_train.shape[0], "messages"
    print "Testing set contains", contents_test.shape[0], "messages"
    
    return contents_train, contents_test, labels_train, labels_test
   
def split_into_lemmas(content):
    words = TextBlob(content.lower()).words
    return [word.lemma for word in words]
 
def feature_extraction(contents_train, contents_test):
    count_vector = CountVectorizer(analyzer=split_into_lemmas, stop_words='english')
    
    #Learn the vocabulary dictionary and return term-document matrix.
    train_messages = count_vector.fit_transform(contents_train)
    test_messages = count_vector.transform(contents_test)
    
    return train_messages, test_messages
    
def metrics(classifier, predictions, labels_test):

    metric_matrix = confusion_matrix(labels_test, predictions)
    print classifier + (' classifier accuracy: '),format(accuracy_score(labels_test, predictions))
    #False positive
    print format(metric_matrix[0][1]),('ham messages were wrongly classified as spam while'),format(metric_matrix[0][0]),('were classified correclty')
    #False negative
    print format(metric_matrix[1][0]), ('spam messages wrongly classified as ham while'), format(metric_matrix[1][1]),('were classified correclty')
    #print classification_report(labels_test, predictions)
    
def naive_bayes_classifier(train_messages, labels_train, test_messages):
    naive_bayes = MultinomialNB() 
    #train the classifier based on the training messages
    naive_bayes.fit(train_messages, labels_train) 
    
    #predic label of test messages using the trained model
    nb_predictions = naive_bayes.predict(test_messages) 
    return nb_predictions

def linear_svc_classifier(train_messages, labels_train, test_messages):
    linear_svc = LinearSVC()
    #train the classifier based on the training messages
    linear_svc.fit(train_messages, labels_train)
    
    #predic label of test messages using the trained model
    linear_svc_predictions = linear_svc.predict(test_messages)
    return linear_svc_predictions


contents_train, contents_test, labels_train, labels_test = load_dataset()
train_messages, test_messages = feature_extraction(contents_train, contents_test)

tfidf_train_messages = TfidfTransformer().fit_transform(train_messages)
tfidf_test_messages = TfidfTransformer().fit_transform(test_messages)

#Naive Bayes classifier
nb_predictions = naive_bayes_classifier(train_messages, labels_train, test_messages)
metrics('Naive Bayes', nb_predictions, labels_test)
print '\n------------------------------------------------\n'

#Naive Bayes classifier with TFIDF
nb_tfidf_predictions = naive_bayes_classifier(tfidf_train_messages, labels_train, tfidf_test_messages)
metrics('Naive Bayes with TFIDF',nb_tfidf_predictions, labels_test)
print '\n------------------------------------------------\n'

#Linear SVC classifier
linear_svc_predictions = linear_svc_classifier(train_messages, labels_train, test_messages)
metrics('Linear SVC', linear_svc_predictions, labels_test)
print '\n------------------------------------------------\n'

#Linear SVC classifier with TFIDF
linear_svc_tfidf_predicitions = linear_svc_classifier(tfidf_train_messages, labels_train, tfidf_test_messages)
metrics('Linear SVC with TDIDF', linear_svc_tfidf_predicitions, labels_test)
print '\n------------------------------------------------\n'
