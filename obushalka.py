import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, auc, roc_curve, confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score
from sklearn.neural_network import MLPRegressor
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import shap


# ///MPG Classification///
# df = pd.read_csv('mpg_car_dataset.csv', sep=' ')
#
# print(df.head(10))
#
# df.info()
#
# df.describe()
#
# X = df[['displacement', 'horsepower', 'weight']]
# y = df['mpg']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# print(f'TRAIN: {X_train.shape}, {y_train.shape} TEST: {X_test.shape},{y_test.shape}')
#
# treeModel = DecisionTreeRegressor(max_depth=5)
# treeModel.fit(X_train, y_train)
#
# featuresNames = X.columns
# print(featuresNames)

# plt.figure(figsize=(20, 15))
# tree.plot_tree(treeModel, feature_names=featuresNames, filled=True)
# # plt.show()
# # plt.savefig('treeModelIndianDataSet.png')
# #
# y_prediction = treeModel.predict(X_test)
#
# print(f'Average absolute error: {round(mean_absolute_error(y_test, y_prediction), 2)} Average squared error: {round(mean_squared_error(y_test, y_prediction), 2)}')
#
# print(confusion_matrix(y_true=y_test, y_pred=y_prediction))
# print(classification_report(y_true=y_test, y_pred=y_prediction))

# ///MPG prediction rfModel///
# rfModel = RandomForestRegressor(n_estimators=3, max_depth=5, n_jobs=-1, random_state=1)
#
# rfModel.fit(X_train, y_train)
#
# y_prediction_rf = rfModel.predict(X_test)
#
# pickle.dump(rfModel, open('rfModel_mpg.pkl', 'wb'))
#
# print(f'Average absolute error: {round(mean_absolute_error(y_test, y_prediction_rf), 2)} Average squared error: {round(mean_squared_error(y_test, y_prediction_rf), 2)}')
#
# print(confusion_matrix(y_true=y_test, y_pred=y_prediction_rf))
# print(classification_report(y_true=y_test, y_pred=y_prediction_rf))

# ///Regressor///
# mlpRegressor = MLPRegressor(hidden_layer_sizes=(9, 3), solver="lbfgs", random_state=1, max_iter=5000)
# mlpRegressor.fit(X_train, y_train)
#
# y_prediction_mlp = mlpRegressor.predict(X_test)
#
# print(f'Average absolute error: {round(mean_absolute_error(y_test, y_prediction_mlp), 2)} Average squared error: {round(mean_squared_error(y_test, y_prediction_mlp), 2)}')

# ///28.04.2022 Improved classifictor///
features = pd.read_csv('attributeNames_covertypes-1.csv')
columnNames = features.columns.tolist()

# //Very big file, gonna work with copy of original data//
originalDf = pd.read_csv('covtype-1.data',
                         names=columnNames, header=None)

# print(originalDf.head())

# print(originalDf['Cover_Type'].value_counts())

# //Changed data//
changedDf = originalDf.loc[(originalDf['Cover_Type'] == 1) | (originalDf['Cover_Type'] == 2)]

changedDf_smaller = resample(changedDf, n_samples=10000, stratify=changedDf['Cover_Type'])

finallyDf = changedDf_smaller

# //Make a tree and get a result//
dfX = finallyDf.drop(['Cover_Type'], axis=1)
dfy = finallyDf['Cover_Type']

dfy = dfy-1

X_train, X_test, y_train, y_test = train_test_split(dfX, dfy, test_size=0.3, stratify=dfy, random_state=1)

treeModel = DecisionTreeClassifier(criterion='entropy', max_depth=15,
                                   min_samples_split=8, random_state=1)
treeModel.fit(X_train, y_train)

y_pred = treeModel.predict(X_test)

print(classification_report(y_true=y_test,y_pred=y_pred))

confmat_tree=confusion_matrix(y_true=y_test,y_pred=y_pred)
print(confmat_tree)

# //Probs//
probsTree = treeModel.predict_proba(X_test)
predsTree = probsTree[:, 1]
fprTree, tprTree, thresholdTree = roc_curve(y_test,predsTree)
roc_aucTree = auc(fprTree, tprTree)
print(f'AUC = {roc_aucTree}')

# //Plot//
# plt.figure(figsize=(7,5))
# plt.title('ROC curve for Decision Tree')
# plt.plot(fprTree, tprTree, color='indigo',linestyle='--', linewidth=2, label=f'AUC = {roc_aucTree}')
# plt.legend(loc='lower right')
# plt.xlabel('False positive')
# plt.ylabel('True positive')
# plt.show()

# //GridSearch: Searching for better result//
model = DecisionTreeClassifier()
parameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 12, 14, 16, 18],
    'min_samples_split': [6, 8, 10, 12]
}

gridTree = GridSearchCV(estimator=model, param_grid=parameters, n_jobs=-1)
gridTree.fit(X_train, y_train)

print(f'Accuracy Best model: {gridTree.best_score_}')
print(f'Best params: {gridTree.best_params_}')
bestTreemodel = gridTree.best_estimator_

y_predBest = bestTreemodel.predict(X_test)
print(classification_report(y_true=y_test,y_pred=y_predBest))

# //Probs for best tree//
probsBestTree = bestTreemodel.predict_proba(X_test)
predsBestTree = probsBestTree[:, 1]
fprBestTree, tprBestTree, thresholdBestTree = roc_curve(y_test,predsBestTree)
roc_aucBestTree = auc(fprBestTree, tprBestTree)
print(f'Best Tree AUC = {roc_aucBestTree}')

# //Plot for best tree//
# plt.figure(figsize=(7,5))
# plt.title('ROC curve for Best Decision Tree')
# plt.plot(fprTree, tprTree, color='indigo',linestyle='--', linewidth=2, label=f'AUC = {roc_aucTree}')
# plt.plot(fprBestTree, tprBestTree, color='tomato', linestyle='--', linewidth=2, label=f'AUC = {roc_aucBestTree}')
# plt.legend(loc='lower right')
# plt.xlabel('False positive')
# plt.ylabel('True positive')
# plt.show()

# //StandartScaler//
stdscalModel = StandardScaler()
stdscalModel.fit(dfX)
X_train_std = stdscalModel.transform(X_train)
X_test_std = stdscalModel.transform(X_test)

# //Logistic regression//
logregModel = LogisticRegression()
logregModelparams = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100]
}
gridLogReg = GridSearchCV(estimator=logregModel, param_grid=logregModelparams, n_jobs=-1)
gridLogReg.fit(X_train_std, y_train)
bestLogReg = gridLogReg.best_estimator_

y_pred_best = bestLogReg.predict(X_test_std)

print(f'Accuracy Best model: {gridLogReg.best_score_}')
print(f'Best params: {gridLogReg.best_params_}')
print(classification_report(y_true=y_test,y_pred=y_pred_best))

# //Probs for best log reg//
probsBestLogReg = bestLogReg.predict_proba(X_test_std)
predsBestLogReg = probsBestLogReg[:, 1]
fprBestLogReg, tprBestLogReg, thresholdBestLogReg = roc_curve(y_test, predsBestLogReg)
roc_aucBestLogReg = auc(fprBestLogReg, tprBestLogReg)
print(f'Best Tree AUC = {roc_aucBestLogReg}')

# //General plot//
# plt.figure(figsize=(7, 5))
# plt.title('ROC curve for Models')
# plt.plot(fprTree, tprTree, color='indigo',linestyle='--', linewidth=2, label=f'AUC = {roc_aucTree}')
# plt.plot(fprBestTree, tprBestTree, color='tomato', linestyle='--', linewidth=2, label=f'AUC = {roc_aucBestTree}')
# plt.plot(fprBestLogReg, tprBestLogReg, color='green', linestyle='--', linewidth=2, label=f'AUC = {roc_aucBestLogReg}')
# plt.legend(loc='lower right')
# plt.xlabel('False positive')
# plt.ylabel('True positive')
# plt.show()

# ///05.05.2022 Importances///
# //Importances DataFrame//
# print(treeModel.feature_importances_)
treeModelImportances = pd.DataFrame(treeModel.feature_importances_, index=dfX.columns, columns=['Tree features importance']).sort_values('Tree features importance')
print(treeModelImportances)
# treeModelImportances.plot.barh(figsize=(8, 5), color='green')

# //Model accuracy after mix features values//
perm_importance = permutation_importance(treeModel, dfX, dfy, n_repeats=10, n_jobs=-1, random_state=1)
for i in perm_importance.importances_mean.argsort()[::-1]:
    print(f'Feature: {dfX.columns[i]}, Avg: {perm_importance.importances_mean[i]:.3f} +/- {perm_importance.importances_std[i]:.3f}')

# fig, ax = plt.subplots(figsize=(6, 6))
# sorted_idx = perm_importance.importances_mean.argsort()
# ax.boxplot(perm_importance.importances[sorted_idx].T, vert=False, labels=dfX.columns[sorted_idx])
# ax.set_title('Importance of swapping')
# plt.show()

# //Using Shap//
X_summary = shap.kmeans(dfX, 10)
treeModel_explainer = shap.KernelExplainer(treeModel.predict, X_summary)
treeModel_1_shap_value = treeModel_explainer.shap_values(dfX.iloc[10, :])
treeModel_15_shap_value = treeModel_explainer.shap_values(dfX.iloc[10:25, :])
treeModel_50_shap_value = treeModel_explainer.shap_values(dfX.iloc[10:60, :])

# shap.summary_plot(treeModel_50_shap_value, dfX.iloc[10:60, :])
# plt.show()

shap.decision_plot(treeModel_explainer.expected_value, treeModel_15_shap_value, feature_names=dfX.columns.tolist(),
                   title='Decision for 15 regions')
plt.show()