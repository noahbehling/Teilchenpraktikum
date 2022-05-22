import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sklean libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import ensemble
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, precision_score, accuracy_score, jaccard_score

df_signal = pd.read_csv("data/signal.csv", delimiter=";")
df_background = pd.read_csv("data/background.csv", delimiter=";")


# drop columns with key words that are associated with MC Simulation 

df_signal = df_signal.drop(df_signal.filter(regex="MC").columns, axis=1)
df_signal = df_signal.drop(df_signal.filter(regex="Weight").columns, axis=1)
df_signal = df_signal.drop(df_signal.filter(regex="Corsika").columns, axis=1)
df_signal = df_signal.drop(df_signal.filter(regex="I3EventHeader").columns, axis=1)
df_signal = df_signal.drop(df_signal.filter(regex="end").columns, axis=1)
df_signal = df_signal.drop(df_signal.filter(regex="start").columns, axis=1)
df_signal = df_signal.drop(df_signal.filter(regex="time").columns, axis=1)
df_signal = df_signal.drop(df_signal.filter(regex="NewID").columns, axis=1)

df_background.drop(df_background.filter(regex="MC").columns, axis=1, inplace=True)
df_background.drop(df_background.filter(regex="Weight").columns, axis=1, inplace=True)
df_background.drop(df_background.filter(regex="Corsika").columns, axis=1, inplace=True)
df_background.drop(df_background.filter(regex="I3EventHeader").columns, axis=1, inplace=True)
df_background.drop(df_background.filter(regex="end").columns, axis=1, inplace=True)
df_background.drop(df_background.filter(regex="start").columns, axis=1, inplace=True)
df_background.drop(df_background.filter(regex="time").columns, axis=1, inplace=True)
df_background.drop(df_background.filter(regex="NewID").columns, axis=1, inplace=True)


df_signal.replace({np.inf: np.nan, -np.inf: np.nan}, value=None, inplace=True)
df_background.replace({np.inf: np.nan, -np.inf: np.nan}, value=None, inplace=True)
# convert all infinities into NaN

df_signal.dropna(axis=1, how="any", inplace=True)
df_background.dropna(axis=1, how="any", inplace=True)
# delete NaN from dataframes

col_signal = df_signal
col_background = df_background
# save columns in sets

for feature in col_signal:
    if feature not in col_background:
        df_signal.drop(feature, axis=1, inplace=True)

for feature in col_background:
    if feature not in col_signal:
        df_background.drop(feature, axis=1, inplace=True)
# remove columns from eother Dataframe that are not in the other dataframe

data = [df_signal, df_background]
df = pd.concat(data)
# merge both dataframes into oner dataframe

df = df.loc[:, df.apply(pd.Series.nunique) != 1]
# remove all columns where every value in the column is equal

X = df.drop("label", axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=15, shuffle=True
)
# split the datafrme in a test- and a train-set whlie shuffeling the data

print(len(X.columns))
# print the number of features

# next the optimal number of attributes will be determined by looking for the number with the highest Jaccard index with the naive bayes clasifier

N_att = np.linspace(10, 100, 10, dtype=int).tolist()
# numbers of attributes to be tested

jac = []
# list that will be used to save the respective jaccard index

# loop over attribute numbers
for att in N_att:
    X_new = SelectKBest(score_func=f_classif, k=att)
    # generate instence to select best features

    d_fit = X_new.fit(X_train, y_train)
    # generate scores for features

    scores = d_fit.scores_
    # scores of features.

    args_max = np.argsort(scores)[::-1]
    # save scores largest to lowest score into an array

    attributes = []
    for i in range(att):
        attributes.append(X_train.columns.tolist()[args_max[i]])
    # saving impotant attributes into a list

    X_train_short = X_train.loc[:, attributes]
    X_test_short = X_test.loc[:, attributes]
    # remove unimportant attributes

    NB_clf = GaussianNB()
    # generate instance of Naive-Bayes classifier

    NB_clf.fit(X_train_short, y_train)
    # train on data

    NB_Jscore = jaccard_score(np.array(y_test), NB_clf.predict(X_test_short))
    jac.append(NB_Jscore)
    # calculate and save jaccard index

    print("Jaccard-Score: ", NB_Jscore, " with ", att, " features")
    # status update

jac_max = max(jac)
jac_maxpos = jac.index(jac_max)
att = N_att[jac_maxpos]
print(att, " attributes will be used")
# save best number of attributes


# now the data will be prepared by removing unimportant attributes

X_new = SelectKBest(score_func=f_classif, k=att)
d_fit = X_new.fit(X_train, y_train)
# generate scores to select features

scores = d_fit.scores_
# get the scores

args_max = np.argsort(scores)[::-1]
# save scores largest to lowest scores into array

features = []
for i in range(att):
    features.append(X_train.columns.tolist()[args_max[i]])
# save only important features

X_train = X_train.loc[:, features]
X_test = X_test.loc[:, features]
# remove unimprtant features


# the next step is the implementation of the random forest classifier

# find the optimal number of trees

N_trees = np.linspace(10, 150, 15, dtype=int).tolist()
# list with the numbers of trees to be tested

jac = []
# array to safe jaccard index

for tree in N_trees:
    RFClf = ensemble.RandomForestClassifier(n_estimators=tree)
    # generate classifier with tree trees

    RFClf.fit(X_train, y_train)
    # train classifier

    rfc_Jscore = jaccard_score(y_test, RFClf.predict(X_test))
    # calculate score

    jac.append(rfc_Jscore)
    # save jaccard index

    print("jaccard score, RFC: ", rfc_Jscore, " with ", tree, " trees")
    # status update


jac_max = max(jac)
jac_maxpos = jac.index(jac_max)
trees = N_trees[jac_maxpos]
# save everithing related toi the maximal value

print("Maximal index, RFC: ", jac_max, " at ", jac_maxpos)
print("With ", trees, "trees")
# print the result


plt.plot(N_trees, jac)
plt.yscale("log")
plt.xlabel("Number of trees")
plt.ylabel("Jaccard Score")
plt.title("Jaccard-Score, RFC")
# plt.show()
# make a beautiful plot to see if there is a trend

# <<<<<<< HEAD
# #plt.savefig('/plots/RF_test.pdf')
# ||||||| 340fc38
# plt.savefig('/plots/RF_test.pdf')
# =======
plt.savefig("plots/RF_test.pdf")
# >>>>>>> 333782f0034d33d89ecaac4534123250a69b7cf3
plt.clf()

# finally the classifier will be used with the optimal trees

RFClf = ensemble.RandomForestClassifier(n_estimators=trees)
# generate classifier with 100 trees

# RFClf.get_params()
# returns parameters of estimator in dictionary?

RFClf.fit(X_train, y_train)
# train classifier

y_pred = RFClf.predict_proba(X_test)
# predict values for the test set

y_pred = y_pred[:, 1]
# congratulations, its a vector

fpr1, tpr1, thr1 = roc_curve(y_test, y_pred)
# get estimates of false positive rate, true positive rate and thr


cv_score_rfc_eff = cross_val_score(RFClf, X, y, cv=5, scoring="recall")
print(
    "Efficiency: %0.4f (+/- %0.4f)"
    % (cv_score_rfc_eff.mean(), cv_score_rfc_eff.std() * 2)
)

cv_score_rfc_rein = cross_val_score(RFClf, X, y, cv=5, scoring="precision")
print(
    "Purity: %0.4f (+/- %0.4f)"
    % (cv_score_rfc_rein.mean(), cv_score_rfc_rein.std() * 2)
)

cv_score_rfc_J = cross_val_score(RFClf, X, y, cv=5, scoring="jaccard")
print(
    "Jaccard Index: %0.4f (+/- %0.4f)"
    % (cv_score_rfc_J.mean(), cv_score_rfc_J.std() * 2)
)
#  efficiency, precision and jaccard index with cross-validation

# now with kNN classifier

# estimate best number for neighbours

N_neighbours = np.linspace(10, 50, 10, dtype=int).tolist()
# list with the numbers of trees to be tested

jac = []
# list to save jaccard index

for neigh in N_neighbours:

    kNN_clf = ensemble.RandomForestClassifier(n_estimators=neigh)
    # generate classifier with neigh neighbours

    kNN_clf.fit(X_train, y_train)
    # train classifier

    kNN_Jscore = jaccard_score(y_test, kNN_clf.predict(X_test))
    # calculate score

    jac.append(kNN_Jscore)
    # berechne den jaccard score und speicher ihn ab

    print("jaccard score, kNN: ", kNN_Jscore, " with ", neigh, " neighbours")
    # status update

jac_max = max(jac)
jac_maxpos = jac.index(jac_max)
neighbours = N_neighbours[jac_maxpos]
# save everithing related the the maximal value

print("Maximal index, kNN: ", jac_max, " at ", jac_maxpos)
print("With ", neighbours, " neighbours")
# print the result


plt.plot(N_neighbours, jac)
plt.yscale("log")
plt.xlabel("Number of Neighbours")
plt.ylabel("Jaccard Score")
plt.title("Jaccard-Score, kNN")
# plt.show()
# make a beautiful plot to see if there is a trend

# <<<<<<< HEAD
# #plt.savefig('/plots/kNN_test.pdf')
# ||||||| 340fc38
# plt.savefig('/plots/kNN_test.pdf')
# =======
plt.savefig("plots/kNN_test.pdf")
# >>>>>>> 333782f0034d33d89ecaac4534123250a69b7cf3

plt.close()

# kNN learning

knn_clf = KNeighborsClassifier(n_neighbors=neighbours)
knn_clf.fit(X_train, y_train)
PRED_knn = knn_clf.predict_proba(X_test)
PRED_knn = PRED_knn[:, 1]
fpr2, tpr2, thr2 = roc_curve(y_test, PRED_knn)


cv_score_knn_eff = cross_val_score(knn_clf, X, y, cv=5, scoring="recall")
print(
    "Efficiency: %0.4f (+/- %0.4f)"
    % (cv_score_knn_eff.mean(), cv_score_knn_eff.std() * 2)
)

cv_score_knn_rein = cross_val_score(knn_clf, X, y, cv=5, scoring="precision")
print(
    "Purity: %0.4f (+/- %0.4f)"
    % (cv_score_knn_rein.mean(), cv_score_knn_rein.std() * 2)
)

cv_score_knn_J = cross_val_score(knn_clf, X, y, cv=5, scoring="jaccard")
print(
    "Jaccard Index: %0.4f (+/- %0.4f)"
    % (cv_score_knn_J.mean(), cv_score_knn_J.std() * 2)
)


# finally the Naive-Bayes classifier

NB_clf = GaussianNB()
NB_clf.fit(X_train, y_train)
NB_pred = NB_clf.predict_proba(X_test)
NB_pred = NB_pred[:, 1]
fpr3, tpr3, thr3 = roc_curve(y_test, NB_pred)


cv_score_nb_eff = cross_val_score(NB_clf, X, y, cv=5, scoring="recall")
print(
    "Efficiency: %0.4f (+/- %0.4f)" % (cv_score_nb_eff.mean(), cv_score_nb_eff.std() * 2)
)

cv_score_nb_rein = cross_val_score(NB_clf, X, y, cv=5, scoring="precision")
print(
    "Purity: %0.4f (+/- %0.4f)"
    % (cv_score_nb_rein.mean(), cv_score_nb_rein.std() * 2)
)

cv_score_nb_J = cross_val_score(NB_clf, X, y, cv=5, scoring="jaccard")
print(
    "Jaccard Index: %0.4f (+/- %0.4f)" % (cv_score_nb_J.mean(), cv_score_nb_J.std() * 2)
)


# all into one happy plot

plt.plot(fpr3, tpr3, label="Naive Bayes ROC")
plt.plot(fpr2, tpr2, label="kNN ROC, NN = {}".format(neighbours))
plt.plot(fpr1, tpr1, label="Random Forest ROC, {} trees".format(trees))
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
#plt.yscale("log")
plt.title("ROC curve")
plt.legend(loc="best")
# plt.show()

# <<<<<<< HEAD
# #plt.savefig('/plots/ROC.pdf')
# ||||||| 340fc38
# plt.savefig('/plots/ROC.pdf')
# =======
plt.savefig("plots/ROC.pdf")
# >>>>>>> 333782f0034d33d89ecaac4534123250a69b7cf3
# plt.close()
