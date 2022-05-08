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

df_signal = pd.read_csv('data/signal.csv', delimiter=';')
df_background = pd.read_csv('data/background.csv', delimiter=';')

# the first step will be the preparation of the data

columns = ['CorsikaWeightMap.SpectrumType',
 'CorsikaWeightMap.TimeScale', 'CorsikaWeightMap.AreaSum',
 'CorsikaWeightMap.Atmosphere',
 'CorsikaWeightMap.CylinderLength',
 'CorsikaWeightMap.CylinderRadius',
 'CorsikaWeightMap.DiplopiaWeight',
 'CorsikaWeightMap.EnergyPrimaryMax',
 'CorsikaWeightMap.EnergyPrimaryMin',
 'CorsikaWeightMap.FluxSum',
 'CorsikaWeightMap.Multiplicity',
 'CorsikaWeightMap.SpectralIndexChange',
 'CorsikaWeightMap.Weight', 'I3MCWeightDict.ActiveLengthAfter',
 'I3MCWeightDict.ActiveLengthBefore',
 'I3MCWeightDict.AutoExtension',
 'I3MCWeightDict.EnergyLost',
 'I3MCWeightDict.GeneratorVolume',
 'I3MCWeightDict.InIceNeutrinoEnergy',
 'I3MCWeightDict.InjectionSurfaceR',
 'I3MCWeightDict.InteractionColumnDepth',
 'I3MCWeightDict.InteractionCrosssection',
 'I3MCWeightDict.InteractionType',
 'I3MCWeightDict.LengthInVolume',
 'I3MCWeightDict.MaxAzimuth', 
 'I3MCWeightDict.MaxEnergyLog',
 'I3MCWeightDict.MaxZenith',
 'I3MCWeightDict.MinAzimuth',
 'I3MCWeightDict.MinEnergyLog',
 'I3MCWeightDict.MinZenith',
 'I3MCWeightDict.NeutrinoImpactParameter',
 'I3MCWeightDict.OneWeight',
 'I3MCWeightDict.PowerLawIndex',
 'I3MCWeightDict.PrimaryNeutrinoEnergy',
 'I3MCWeightDict.RangeInMeter',
 'I3MCWeightDict.RangeInMeterWaterEquiv',
 'I3MCWeightDict.TotalColumnDepth',
 'I3MCWeightDict.TotalCrosssection',
 'I3MCWeightDict.TotalDetectionLength',
 'I3MCWeightDict.TotalInteractionProbability',
 'I3MCWeightDict.TotalInteractionProbabilityWeight',
 'I3MCWeightDict.TotalPropagationProbability',
 'I3MCWeightDict.TrueActiveLengthAfter',
 'I3MCWeightDict.TrueActiveLengthBefore',
 'MCECenter.value',
 'MCMostEnergeticInIce.x',
 'MCMostEnergeticInIce.y',
 'MCMostEnergeticInIce.z',
 'MCMostEnergeticInIce.time',
 'MCMostEnergeticInIce.zenith',
 'MCMostEnergeticInIce.azimuth',
 'MCMostEnergeticInIce.energy',
 'MCMostEnergeticInIce.length',
 'MCMostEnergeticInIce.type',
 'MCMostEnergeticInIce.fit_status',
 'MCMostEnergeticTrack.x',
 'MCMostEnergeticTrack.y',
 'MCMostEnergeticTrack.z',
 'MCMostEnergeticTrack.time',
 'MCMostEnergeticTrack.zenith',
 'MCMostEnergeticTrack.azimuth',
 'MCMostEnergeticTrack.energy',
 'MCMostEnergeticTrack.length',
 'MCMostEnergeticTrack.type',
 'MCMostEnergeticTrack.fit_status',
 'MCPrimary1.x',
 'MCPrimary1.y',
 'MCPrimary1.z',
 'MCPrimary1.time',
 'MCPrimary1.zenith',
 'MCPrimary1.azimuth',
 'MCPrimary1.energy',
 'MCPrimary1.length',
 'MCPrimary1.type',
 'MCPrimary1.fit_status',
 'Weight.HoSa',
 'Weight.Ho',
 'Weight.Sa',
 'Weight.Astro2', 
 'MPEFitHighNoiseFitParams.logl', 
 'MPEFitHighNoiseFitParams.rlogl']

df_signal.drop(columns, axis = 1, inplace = True)
# remove Monte-Carlo truths in the signal dataframe 

columns = ['CorsikaWeightMap.AreaSum',
 'CorsikaWeightMap.Atmosphere',
 'CorsikaWeightMap.CylinderLength',
 'CorsikaWeightMap.CylinderRadius',
 'CorsikaWeightMap.DiplopiaWeight',
 'CorsikaWeightMap.EnergyPrimaryMax',
 'CorsikaWeightMap.EnergyPrimaryMin',
 'CorsikaWeightMap.FluxSum',
 'CorsikaWeightMap.Multiplicity',
 'CorsikaWeightMap.ParticleType',
 'CorsikaWeightMap.Polygonato',
 'CorsikaWeightMap.PrimarySpectralIndex',
 'CorsikaWeightMap.TimeScale',
 'CorsikaWeightMap.Weight', 
 'MCECenter.value',
 'MCMostEnergeticInIce.x',
 'MCMostEnergeticInIce.y',
 'MCMostEnergeticInIce.z',
 'MCMostEnergeticInIce.time',
 'MCMostEnergeticInIce.zenith',
 'MCMostEnergeticInIce.azimuth',
 'MCMostEnergeticInIce.energy',
 'MCMostEnergeticInIce.length',
 'MCMostEnergeticInIce.type',
 'MCMostEnergeticInIce.fit_status',
 'MCPrimary1.x',
 'MCPrimary1.y',
 'MCPrimary1.z',
 'MCPrimary1.time',
 'MCPrimary1.zenith',
 'MCPrimary1.azimuth',
 'MCPrimary1.energy',
 'MCPrimary1.length',
 'MCPrimary1.type',
 'MCPrimary1.fit_status', 
 'Weight.Ho',
 'Weight.Sa',
 'Weight.Astro2',
 'Weight.HoSa', 
 'MPEFitHighNoiseFitParams.logl', 
 'MPEFitHighNoiseFitParams.rlogl']

df_background.drop(columns, axis = 1, inplace = True)
# remove Monte-Carlo truths from background dataframe 

# drop von anderen Protokoll ###############################################################################

df_signal = df_signal.drop(df_signal.filter(regex='MC').columns, axis=1)
df_signal = df_signal.drop(df_signal.filter(regex='Weight').columns, axis=1)
df_signal = df_signal.drop(df_signal.filter(regex='Corsika').columns, axis=1)
df_signal = df_signal.drop(df_signal.filter(regex='I3EventHeader').columns, axis=1)
df_signal = df_signal.drop(df_signal.filter(regex='end').columns, axis=1)
df_signal = df_signal.drop(df_signal.filter(regex='start').columns, axis=1)
df_signal = df_signal.drop(df_signal.filter(regex='time').columns, axis=1)
df_signal = df_signal.drop(df_signal.filter(regex='NewID').columns, axis=1)

df_background.drop(df_background.filter(regex='MC').columns, axis=1, inplace = True)
df_background.drop(df_background.filter(regex='Weight').columns, axis=1, inplace = True)
df_background.drop(df_background.filter(regex='Corsika').columns, axis=1, inplace = True)
df_background.drop(df_background.filter(regex='I3EventHeader').columns, axis=1, inplace = True)
df_background.drop(df_background.filter(regex='end').columns, axis=1, inplace = True)
df_background.drop(df_background.filter(regex='start').columns, axis=1, inplace = True)
df_background.drop(df_background.filter(regex='time').columns, axis=1, inplace = True)
df_background.drop(df_background.filter(regex='NewID').columns, axis=1, inplace = True)

############################################################################################################


df_signal.replace({np.inf : np.nan, -np.inf : np.nan}, value=None, inplace = True)
df_background.replace({np.inf : np.nan, -np.inf : np.nan}, value=None, inplace = True)
# convert all infinities into NaN

df_signal.dropna(axis=1, how='any', inplace = True)
df_background.dropna(axis=1, how='any', inplace = True)
# delete NaN from dataframes 

col_signal = df_signal
col_background = df_background
# save columns in sets

for feature in col_signal:
    if feature not in col_background:
        df_signal.drop(feature, axis=1, inplace = True)

for feature in col_background:
    if feature not in col_signal:
        df_background.drop(att, axis=1, inplace = True)
# remove columns from eother Dataframe that are not in the other dataframe

data = [df_signal, df_background]
df = pd.concat(data)
# merge both dataframes into oner dataframe 

df = df.loc[:,df.apply(pd.Series.nunique) != 1]
# remove all columns where every value in the column is equal 

X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, shuffle=True)
# split the datafrme in a test- and a train-set whlie shuffeling the data 


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

    print("Jaccard-Score: ", NB_Jscore, " with ", att, ' features')
    # status update

jac_max = max(jac)
jac_maxpos = jac.index(jac_max)
att = N_att[jac_maxpos]
print(att, ' attributes will be used')
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

N_trees = np.linspace(10, 310, 11, dtype=int).tolist()
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
    
    print('jaccard score, RFC: ', rfc_Jscore, ' with ', tree, ' trees')
    # status update


jac_max = max(jac)
jac_maxpos = jac.index(jac_max)
trees = N_trees[jac_maxpos]
# save everithing related toi the maximal value

print('Maximal index, RFC: ', jac_max, ' at ', jac_maxpos)
print('With ', trees, 'trees')
# print the result


plt.plot(N_trees, jac)
plt.yscale('log')
plt.xlabel('Number of trees')
plt.ylabel('Jaccard Score')
plt.title('Jaccard-Score, RFC')
#plt.show()
# make a beautiful plot to see if there is a trend

#plt.savefig('/plots/RF_test.pdf')
plt.close()

# finally the classifier will be used with the optimal trees 

RFClf = ensemble.RandomForestClassifier(n_estimators=trees)
# generate classifier with 100 trees

#RFClf.get_params()
# returns parameters of estimator in dictionary?

RFClf.fit(X_train, y_train)
# train classifier

y_pred = RFClf.predict_proba(X_test)
# predict values for the test set

y_pred = y_pred[:, 1]
# congratulations, its a vector

fpr1, tpr1, thr1 = roc_curve(y_test, y_pred)
# get estimates of false positive rate, true positive rate and thr

RFC_precision = precision_score(y_test, RFClf.predict(X_test))
print('RFC precision score(sklearn) = ', RFC_precision)

RFC_eff = accuracy_score(y_test, RFClf.predict(X_test))
print('RFC accuracy score(sklearn) = ', RFC_eff)

rfc_Jscore = jaccard_score(y_test, RFClf.predict(X_test))
print('jaccard score, RFC: ', rfc_Jscore)
# generate accuracy, precision and jaccard score

cv_score_rfc_eff = cross_val_score(RFClf, X, y, cv=5, scoring='recall')
print("Effizienz: %0.4f (+/- %0.4f)" % (cv_score_rfc_eff.mean(), cv_score_rfc_eff.std() * 2))

cv_score_rfc_rein = cross_val_score(RFClf, X, y, cv=5, scoring='precision')
print("Reinheit: %0.4f (+/- %0.4f)" % (cv_score_rfc_rein.mean(), cv_score_rfc_rein.std() * 2))

cv_score_rfc_J = cross_val_score(RFClf, X, y, cv=5, scoring='jaccard')
print("Jaccard Index: %0.4f (+/- %0.4f)" % (cv_score_rfc_J.mean(), cv_score_rfc_J.std() * 2))
#  efficiency, precision and jaccard index with cross-validation

# now with kNN classifier 

# estimate best number for neighbours 

N_neighbours = np.linspace(10, 310, 11, dtype=int).tolist()
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
    
    print('jaccard score, kNN: ', kNN_Jscore, ' with ', neigh, ' neighbours')
    # status update

jac_max = max(jac)
jac_maxpos = jac.index(jac_max)
neighbours = N_neighbours[jac_maxpos]
# save everithing related the the maximal value

print('Maximal index, kNN: ', jac_max, ' at ', jac_maxpos)
print('With ', neighbours, ' neighbours')
# print the result


plt.plot(N_neighbours, jac)
plt.yscale('log')
plt.xlabel('Number of Neighbours')
plt.ylabel('Jaccard Score')
plt.title('Jaccard-Score, kNN')
#plt.show()
# make a beautiful plot to see if there is a trend

#plt.savefig('/plots/kNN_test.pdf')

plt.close()

# kNN learning 

knn_clf = KNeighborsClassifier(n_neighbors=neighbours)
knn_clf.fit(X_train, y_train)
PRED_knn = knn_clf.predict_proba(X_test)
PRED_knn = PRED_knn[:, 1]
fpr2, tpr2, thr2 = roc_curve(y_test, PRED_knn)

knn_precision = precision_score(y_test, knn_clf.predict(X_test))
print('KNN precision score(sklearn) = ', knn_precision)

knn_eff = accuracy_score(y_test, knn_clf.predict(X_test))
print('KNN accuracy score(sklearn) = ', knn_eff)

knn_Jscore = jaccard_score(y_test, knn_clf.predict(X_test))
print('jaccard score, kNN: ', knn_Jscore)

cv_score_knn_eff = cross_val_score(knn_clf, X, y, cv=5, scoring='recall')
print("Effizienz: %0.4f (+/- %0.4f)" % (cv_score_knn_eff.mean(), cv_score_knn_eff.std() * 2))

cv_score_knn_rein = cross_val_score(knn_clf, X, y, cv=5, scoring='precision')
print("Reinheit: %0.4f (+/- %0.4f)" % (cv_score_knn_rein.mean(), cv_score_knn_rein.std() * 2))

cv_score_knn_J = cross_val_score(knn_clf, X, y, cv=5, scoring='jaccard')
print("Jaccard Index: %0.4f (+/- %0.4f)" % (cv_score_knn_J.mean(), cv_score_knn_J.std() * 2))


# finally the Naive-Bayes classifier

NB_clf = GaussianNB()
NB_clf.fit(X_train, y_train)
NB_pred = NB_clf.predict_proba(X_test)
NB_pred = NB_pred[:, 1]
fpr3, tpr3, thr3 = roc_curve(y_test, NB_pred)

NB_precision = precision_score(y_test, NB_clf.predict(X_test))
print('NB precision score(sklearn) = ', NB_precision)

NB_eff = accuracy_score(y_test, NB_clf.predict(X_test))
print('NB accuracy score(sklearn) = ', NB_eff)

NB_Jscore = jaccard_score(y_test, NB_clf.predict(X_test))
print('jaccard score, NB: ', NB_Jscore)

cv_score_nb_eff = cross_val_score(NB_clf, X, y, cv=5, scoring='recall')
print("Effizienz: %0.4f (+/- %0.4f)" % (cv_score_nb_eff.mean(), cv_score_nb_eff.std() * 2))

cv_score_nb_rein = cross_val_score(NB_clf, X, y, cv=5, scoring='precision')
print("Reinheit: %0.4f (+/- %0.4f)" % (cv_score_nb_rein.mean(), cv_score_nb_rein.std() * 2))

cv_score_nb_J = cross_val_score(NB_clf, X, y, cv=5, scoring='jaccard')
print("Jaccard Index: %0.4f (+/- %0.4f)" % (cv_score_nb_J.mean(), cv_score_nb_J.std() * 2))


# all into one plot 

plt.plot(fpr3, tpr3, label='Naive Bayes ROC')
plt.plot(fpr2, tpr2, label='kNN ROC, NN = {}'.format(neighbours))
plt.plot(fpr1, tpr1, label='Random Forest ROC, {} trees'.format(trees))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.yscale('log')
plt.title('ROC curve')
plt.legend(loc='best')
#plt.show()

#plt.savefig('/plots/ROC.pdf')
plt.close()