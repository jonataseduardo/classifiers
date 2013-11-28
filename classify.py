import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import LeaveOneOut
from multiprocessing import Queue, Process
from Queue import Empty
from scipy.stats import f_oneway, ttest_ind


def set_classifiers(list_knn=[3, 5, 7], list_svm=[1]):
    """
    Parameters
    __________
    list_knn: list of number of neigbours
    list_svm: list of C parameters

    Returns
    _______
    classifiers: diciontary of classifiers with
    keys = classifiers name
    values = classifers function

    """
    if type(list_knn) is not dict:
        classifiers = dict()
        for neigh in list_knn:
            classifiers["KNN" + str(neigh)] = KNeighborsClassifier(neigh)
        for C in list_svm:
            classifiers["SVM_C" + str(C)] = SVC(kernel="linear", C=C)
        return classifiers
    else:
        classifiers = dict()
        for neigh in list_knn:
            classifiers["KNN" + str(neigh)] = (KNeighborsClassifier(neigh),
                                               list_knn[neigh])
        for C in list_svm:
            classifiers["SVM_C" + str(C)] = (SVC(kernel="linear", C=C),
                                             list_svm[C])
        return classifiers


def loo_predict(X, y, classifiers, loo=None):
    """
    Parameters
    ____________
    X: array of shape [n_samples, n_features]
    y: array of shap [n_features]
    classifiers: diciontary of classifiers with
    keys = classifiers name
    values = classifers function

    Return
    ______

    result = dictionary with
    key = calssifires name
    values = classificaire matthews correlation coeficient score

    """
    if loo is None:
        loo = LeaveOneOut(len(y))

    result = dict()
    for clf in classifiers:
        prediction = []
        truelabel = []
        for t in loo:
            X_train, y_train = X[t[0]], y[t[0]]
            X_valid, y_valid = X[t[1]], y[t[1]][0]
            classifiers[clf].fit(X_train, y_train)
            predict = classifiers[clf].predict(X_valid)[0]
            truelabel.append(y_valid)
            prediction.append(predict)
        result[clf] = MCC(truelabel, prediction)
    return result


def loo_feature_selection(X, y, classifiers, list_features, nthreads=1):
    """
    Parameters:
        X[n_samples, n_features]
        y[n_features]
        classifiers
        list_features:
        ntreads


    Returns:
        (list_features, list_MCC)
    """
    qin = Queue()
    qout = Queue()
    for dim in list_features:
        qin.put(dim)

    loo = LeaveOneOut(len(y))

    def doit():
        while True:
            try:
                dim = qin.get(block=False)
                res = loo_predict(X[:, :dim], y, classifiers, loo)
                res["nfeatures"] = dim
                qout.put(res)
            except Empty:
                break

    processes = [Process(target=doit) for i in range(nthreads)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    y = dict()
    for i in range(qout.qsize()):
        x = qout.get()
        #print x
        if i == 0:
            for k in x:
                y[k] = [x[k]]
        else:
            for k in x:
                y[k].append(x[k])

    return y


def predict(X_train, y_train, X_valid, y_valid, classifiers):
    """
    Parameters
    ____________
    X: array of shape [n_samples, n_features]
    y: array of shap [n_features]
    classifiers: diciontary of classifiers with
    keys = classifiers name
    values = classifers function

    Return
    ______

    result = dictionary with
    key = calssifires name
    values = classificaire matthews correlation coeficient score

    """
    result = dict()
    if type(classifiers.values()[0]) is not tuple:
        print type(classifiers.values()[0])
        for clf in classifiers:
            classifiers[clf].fit(X_train, y_train)
            y_predict = classifiers[clf].predict(X_valid)
            result[clf] = MCC(y_valid, y_predict)
        return result
    else:
        for clf in classifiers:
            clff = classifiers[clf][0]
            dim = classifiers[clf][1]
            clff.fit(X_train.T[:dim].T, y_train)
            y_predict = clff.predict(X_valid.T[:dim].T)
            result[(clf, dim)] = MCC(y_valid, y_predict)
        return result


def MCC(y_true, y_predict):
    """
    Parameters
    ----------
    y_true: np.array([n])
    y_predict: np.array([n])

    Returns
    -------
    matthews correlation coeficiente: float

    References
    __________
    [1] Gorodkin, J. (2004). Comparing two< i> K</i>-category assignments by
    a< i> K</i>-category correlation coefficient.
    Computational Biology and Chemistry, 28(5), 367-374.

    [2] Jurman, G., Riccadonna, S., & Furlanello, C. (2012).
    A comparison of MCC and CEN error measures in multi-class prediction.
    PloS one, 7(8), e41882.

    """
    N = max(len(set(y_predict)),
            len(set(y_true)))
    S = len(y_true)

    X = np.zeros((S, N))
    Y = np.zeros((S, N))

    for l, i in zip(X, y_predict):
        l[i] = 1

    for l, i in zip(Y, y_true):
        l[i] = 1

    return COV(X, Y) / np.sqrt(COV(X, X) * COV(Y, Y))


def COV(X, Y):
    """
    Parameters
    ----------
    X: np.array([n_samples,n_classes])
    Y: np.array([n_samples,n_classes])

    Returns
    _______
    cov: float

    """
    S, N = X.shape
    x = X.T - X.mean(axis=0).reshape(N, 1)
    y = Y.T - Y.mean(axis=0).reshape(N, 1)
    return np.sum(x * y) / N


def sep_multclass(X, y):

    """
    Parameters
    ----------
    X: np.array([n_samples,n_classes])
    Y: np.array([n_samples,n_classes])

    Returns
    _______
    cly: List of samples of each class

    """
    cly = []
    for cl in set(y):
        cly.append(X[y == cl])
    return cly


def multianova(X, y):
    def sep_multclass(X, y):

        """
        Parameters
        ----------
        X: np.array([n_samples,n_classes])
        Y: np.array([n_samples,n_classes])

        Returns
        _______
        cly: List of samples of each class

        """
        cly = []
        for cl in set(y):
            cly.append(X[y == cl])
        return cly

    def ff_oneway(X):
        return f_oneway(*X)

    G = np.apply_along_axis(sep_multclass, 0, X, y).T

    return np.apply_along_axis(ff_oneway, 1, G)


def anova_sort(X, y):
        """
        Parameters
        ----------
        X: np.array([n_samples, n_features])
        y: np.array([n_samples])

        Returns
        _______
        X_sorted, [[F_value,p-value]]

        """

        C = multianova(X, y)
        ac = np.argsort(C.T[1])
        return X.T[ac].T, C[ac]


def ttest_sort(X, y):
    """
    Parameters
    ----------
    X: np.array([n_samples, n_features])
    y: np.array([n_samples])

    Returns
    _______
    X_sorted, [[t_value,p-value]]

    """
    labels = list(set(y))
    if len(labels) != 2:
        print "number of labels has to be equal 2"
        return None
    else:
        t, p = ttest_ind(X[y == labels[0]], X[y == labels[1]])
        ac = np.argsort(t)
        return X.T[ac].T, np.column_stack((t[ac], p[ac]))
