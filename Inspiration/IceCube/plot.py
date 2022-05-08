import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as skl
# import pymrmr
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import ensemble
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, roc_auc_score, roc_curve, precision_score, accuracy_score, jaccard_score
from sklearn.naive_bayes import GaussianNB

print(" ")
print("Preparing Data...")
#Read Data
signal = pd.read_csv("data/data_signal.csv", sep = ";")
bkg = pd.read_csv("data/data_background.csv", sep = ";")
#Dropping unwanted columns from signal and bkg
signal = signal.drop(signal.filter(regex='MC').columns, axis=1)
signal = signal.drop(signal.filter(regex='Weight').columns, axis=1)
signal = signal.drop(signal.filter(regex='Corsika').columns, axis=1)
signal = signal.drop(signal.filter(regex='I3EventHeader').columns, axis=1)
signal = signal.drop(signal.filter(regex='end').columns, axis=1)
signal = signal.drop(signal.filter(regex='start').columns, axis=1)
signal = signal.drop(signal.filter(regex='time').columns, axis=1)
signal = signal.drop(signal.filter(regex='NewID').columns, axis=1)
signal = signal.drop('label', axis=1)

bkg = bkg.drop(bkg.filter(regex='MC').columns, axis=1)
bkg = bkg.drop(bkg.filter(regex='Weight').columns, axis=1)
bkg = bkg.drop(bkg.filter(regex='Corsika').columns, axis=1)
bkg = bkg.drop(bkg.filter(regex='I3EventHeader').columns, axis=1)
bkg = bkg.drop(bkg.filter(regex='end').columns, axis=1)
bkg = bkg.drop(bkg.filter(regex='start').columns, axis=1)
bkg = bkg.drop(bkg.filter(regex='time').columns, axis=1)
bkg = bkg.drop(bkg.filter(regex='NewID').columns, axis=1)
bkg = bkg.drop('label', axis=1)



#Dropping +-inf and NaN values for bkg and signal
signal.replace([np.inf, -np.inf], np.nan)
signal.dropna(axis = 'columns', inplace = True)
signal = signal.drop(signal.std()[(signal.std() == 0)].index, axis=1)



bkg.replace([np.inf, -np.inf], np.nan)
bkg.dropna(axis = 'columns', inplace = True)
bkg = bkg.drop(bkg.std()[(bkg.std() == 0)].index, axis=1)

#Dropping columns that are only in 1 of the datasets
bcol = bkg.columns
scol = signal.columns

for att in scol:
    if att not in bcol:
        signal.drop(att, axis=1, inplace = True)

for att in bcol:
    if att not in scol:
        bkg.drop(att, axis=1, inplace = True)

print("Data preparation Done") 

#Making labels
sig_label = np.ones(signal.shape[0])
bkg_label = np.zeros(bkg.shape[0])
#Combine Bkg and Signal
combined_df = pd.concat([signal, bkg], ignore_index=True)
combined_label = np.append(sig_label, bkg_label)

##########################################################################################################################################################

#adding labels for mRMR and shuffle data
combined_df.insert(0, 'label', combined_label) # check


shuffled = combined_df.sample(frac=1, replace=True, random_state=1)

# #Discretize Data for mRMR
# kbins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
# discrete_data = kbins.fit_transform(shuffled)
#pymrmr.mRMR(dataset, 'MIQ', 5) #THIS DOES ERROR FOR >1

#Alternate way since mrmr wont work properly
y = shuffled['label']
X = shuffled.drop('label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# feature selection

# Number of features
# N_feat = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# nn = 20
#
# jac = []
# Number=1
# for feat in N_feat:
#
#     X_new = SelectKBest(score_func=f_classif, k=feat)
#     d_fit = X_new.fit(X_train, y_train)
#     # Generate scores
#     scores = d_fit.scores_
#     #print(scores)
#     # Sort after value
#     sorted_scores = sorted(scores, reverse=True)
#     args_max = np.argsort(scores)[::-1]
#     # print(args_max)
#     features = []
#     for i in range(feat):
#         features.append(X.columns.tolist()[args_max[i]])
#     #print(features)
#     # Delete all features except inportant ones
#     X_train_sclief = X_train.loc[:, features]
#     X_test_sclief = X_test.loc[:, features]
#
#     knn_clf = KNeighborsClassifier(n_neighbors=nn)
#     knn_clf.fit(X_train_sclief, y_train)
#     PRED_knn = knn_clf.predict_proba(X_test_sclief)
#     PRED_knn = PRED_knn[:, 1]
#     fpr2, tpr2, thr2 = roc_curve(y_test, PRED_knn)
#
#     knn_precision = precision_score(np.array(y_test), knn_clf.predict(X_test_sclief))
#     knn_eff = accuracy_score(np.array(y_test), knn_clf.predict(X_test_sclief))
#     print('KNN accuracy score(sklearn) = ', knn_eff)
#     print('KNN precision score(sklearn) = ', knn_precision)
#     knn_Jscore = jaccard_score(np.array(y_test), knn_clf.predict(X_test_sclief))
#     jac.append(knn_Jscore)
#     print("Number ", Number, " done")
#     Number=Number+1
# jac_max = max(jac)
# jac_maxpos = jac.index(jac_max)
# print(jac)
# print(jac_max)
# print(jac_maxpos)

N_feat = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# feats = N_feat[jac_maxpos]
feats = 20
X_new = SelectKBest(score_func=f_classif, k=feats)
d_fit = X_new.fit(X_train, y_train)
# Generate scores that give quality of features
scores = d_fit.scores_
# Sort after value
sorted_scores = sorted(scores, reverse=True)
args_max = np.argsort(scores)[::-1]
# print(args_max)
# Read N most important features and delete rest.
features = []
for i in range(feats):
    features.append(X.columns.tolist()[args_max[i]])
# print(features)
# werfe aus den trainingsdaten und testdaten alle features bis auf die wichtigsten raus
X_train = X_train.loc[:, features]
X_test = X_test.loc[:, features]


# learning

# Random Forest Classifier
RFClf = ensemble.RandomForestClassifier(n_estimators=100)
RFClf.get_params()
RFClf.fit(X_train, y_train)
# predict labels
y_pred = RFClf.predict_proba(X_test)
y_pred = y_pred[:, 1]
fpr1, tpr1, thr1 = roc_curve(y_test, y_pred)

#print(roc_auc_score(y_test, y_pred))
#print(r2_score(y_test, y_pred))

RFC_precision = precision_score(y_test, RFClf.predict(X_test))
RFC_eff = accuracy_score(y_test, RFClf.predict(X_test))
print('RFC accuracy score(sklearn) = ', RFC_eff)
print('RFC precision score(sklearn) = ', RFC_precision)
rfc_Jscore = jaccard_score(y_test, RFClf.predict(X_test))
print('jaccard score, RFC: ', rfc_Jscore)

cv_score_rfc_eff = cross_val_score(RFClf, X, y, cv=5, scoring='recall')
print("Effizienz: %0.4f (+/- %0.4f)" % (cv_score_rfc_eff.mean(), cv_score_rfc_eff.std() * 2))
cv_score_rfc_rein = cross_val_score(RFClf, X, y, cv=5, scoring='precision')
print("Reinheit: %0.4f (+/- %0.4f)" % (cv_score_rfc_rein.mean(), cv_score_rfc_rein.std() * 2))
cv_score_rfc_J = cross_val_score(RFClf, X, y, cv=5, scoring='jaccard')
print("Jaccard Index: %0.4f (+/- %0.4f)" % (cv_score_rfc_J.mean(), cv_score_rfc_J.std() * 2))

#KNN Classifier
nn = 20
knn_clf = KNeighborsClassifier(n_neighbors=nn)
knn_clf.fit(X_train, y_train)
PRED_knn = knn_clf.predict_proba(X_test)
PRED_knn = PRED_knn[:, 1]
fpr2, tpr2, thr2 = roc_curve(y_test, PRED_knn)

knn_precision = precision_score(y_test, knn_clf.predict(X_test))
knn_eff = accuracy_score(y_test, knn_clf.predict(X_test))
print('KNN accuracy score(sklearn) = ', knn_eff)
print('KNN precision score(sklearn) = ', knn_precision)
knn_Jscore = jaccard_score(y_test, knn_clf.predict(X_test))
print('jaccard score, kNN: ', knn_Jscore)

cv_score_knn_eff = cross_val_score(knn_clf, X, y, cv=5, scoring='recall')
print("Effizienz: %0.4f (+/- %0.4f)" % (cv_score_knn_eff.mean(), cv_score_knn_eff.std() * 2))
cv_score_knn_rein = cross_val_score(knn_clf, X, y, cv=5, scoring='precision')
print("Reinheit: %0.4f (+/- %0.4f)" % (cv_score_knn_rein.mean(), cv_score_knn_rein.std() * 2))
cv_score_knn_J = cross_val_score(knn_clf, X, y, cv=5, scoring='jaccard')
print("Jaccard Index: %0.4f (+/- %0.4f)" % (cv_score_knn_J.mean(), cv_score_knn_J.std() * 2))

#Naive Bayes:
NBClf = GaussianNB()
NBClf.fit(X_train, y_train)
NB_pred = NBClf.predict_proba(X_test)
NB_pred = NB_pred[:, 1]
fpr3, tpr3, thr3 = roc_curve(y_test, NB_pred)

NB_precision = precision_score(y_test, NBClf.predict(X_test))
NB_eff = accuracy_score(y_test, NBClf.predict(X_test))
print('NB accuracy score(sklearn) = ', NB_eff)
print('NB precision score(sklearn) = ', NB_precision)
NB_Jscore = jaccard_score(y_test, NBClf.predict(X_test))
print('jaccard score, NB: ', NB_Jscore)

cv_score_nb_eff = cross_val_score(NBClf, X, y, cv=5, scoring='recall')
print("Effizienz: %0.4f (+/- %0.4f)" % (cv_score_nb_eff.mean(), cv_score_nb_eff.std() * 2))
cv_score_nb_rein = cross_val_score(NBClf, X, y, cv=5, scoring='precision')
print("Reinheit: %0.4f (+/- %0.4f)" % (cv_score_nb_rein.mean(), cv_score_nb_rein.std() * 2))
cv_score_nb_J = cross_val_score(NBClf, X, y, cv=5, scoring='jaccard')
print("Jaccard Index: %0.4f (+/- %0.4f)" % (cv_score_nb_J.mean(), cv_score_nb_J.std() * 2))

print("Plotting ROC curve...")

plt.figure(1)
plt.plot(fpr3, tpr3, label='Naive Bayes ROC')
plt.plot(fpr2, tpr2, label='kNN ROC, NN = {}'.format(nn))
plt.plot(fpr1, tpr1, label='Random Forest ROC')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig("plots/roc_curve.pdf")

print("ROC Curve done")

print(" ")
print("PROGRAM SUCCEDED")
