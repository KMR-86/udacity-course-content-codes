def classify(features_train, labels_train):
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier

    ### your code goes here!

    from sklearn import svm
    clf = svm.SVC(C=100.0, gamma=1.0, kernel='rbf')
    clf.fit(features_train, labels_train)
    return clf