# Cure the princess

# Por Ramiro Padilla

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc

#%%
### Analisis del dataset ###

df = pd.read_csv('data.csv')
df.head()
df.info()

highest_corr = df.corr().iloc[-1,:-2].idxmax()

cured_vs_not = sns.countplot(data=df, x='Cured')

correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap de Correlación')
plt.show()

#%%
### Datos test-training ###

from sklearn.model_selection import train_test_split

X = df.iloc[:,:-1].values
y = df['Cured'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0) 

#%%
### Seleccion de modelo: Support Vector Classifier (SVM) ###

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

p_svc = make_pipeline(StandardScaler(),
                             SVC(random_state=1))

param_range = [0.001, 0.1, 1 , 10, 100, 1000]

param_grid = [{'svc__C' : param_range,
               'svc__kernel' : ['linear']},
              {'svc__C' : param_range,
              'svc__gamma' : [1, 0.1 ,0.01, 0.001, 0.0001],
              'svc__kernel' : ['rbf']}]

inner_cv = KFold(n_splits=4, shuffle=True, random_state=2)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=2)

# bucle interior
clf_svm = GridSearchCV(estimator=p_svc, 
                   param_grid=param_grid,
                   scoring='accuracy',
                   cv=inner_cv,
                   n_jobs=3) # 3 nucleos del CPU

# bucle exterior, 
scores_svm = cross_val_score(clf_svm, X_train, y_train,
                         scoring='accuracy',
                         cv=outer_cv,
                         n_jobs=3)

CV_accuracy_svm = np.mean(scores_svm)
CV_acc_std_svm = np.std(scores_svm)

#%%
### Seleccion de modelo: Random Forest Classifier ###

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=1)

number_trees = [5, 10, 15, 20, 25, 30]
depth_range = [3, 5, 10, 15, 30, 35]

param_grid = [{'n_estimators' : number_trees,
               'max_depth' : depth_range}]

# bucle interior
clf_rfc = GridSearchCV(estimator=rfc, 
                   param_grid=param_grid,
                   scoring='accuracy',
                   cv=inner_cv,
                   n_jobs=3) # 3 nucleos del CPU

# bucle exterior,
scores_rfc = cross_val_score(clf_rfc, X_train, y_train,
                         scoring='accuracy',
                         cv=outer_cv,
                         n_jobs=3)

CV_accuracy_rfc = np.mean(scores_rfc)
CV_acc_std_rfc = np.std(scores_rfc)

#%%
### Curva de aprendizaje

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

clf_svm = clf_svm.fit(X_train, y_train)
SVC = clf_svm.best_estimator_

train_sizes, train_scores, test_scores = learning_curve(estimator=SVC,
                                             X=X_train,
                                             y=y_train,
                                             train_sizes=np.linspace(0.1, 1.0, 10),
                                             cv = 5,
                                             n_jobs=3)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='precisión training')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='precisión validación')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xlabel('Numero de muestras de entrenamiento')
plt.ylabel('Precisión')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.tight_layout()
plt.show()
#%%
### Curva de validacion

#%%
### Seleccion de caracteristicas

clf_rfc = clf_rfc.fit(X_train, y_train)
forest = clf_rfc.best_estimator_

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

plt.title('Importancia de ingredientes')

plt.bar(range(indices.size), 
        importances[indices],
        align='center')

plt.xticks(ticks=range(indices.size),
           labels=df.columns[:-1],
           rotation=90)

plt.tight_layout()
plt.show()

#%%

# ### Entrenamiento modelo final ###

# final_svm = SVC(**best_params, probability=True) # paso los mejores parametros como keywords arguments
# final_svm.fit(X_train_sd, y_train_1d)


# y_pred_train = final_svm.predict(X_train_sd)
# y_pred_test = final_svm.predict(X_test_sd)


# # Resultados del mejor modelo
# final_report = classification_report(y_test, y_pred_test)
# print(final_report)
# # Precision = cuan utiles son los resultados de busqueda
# # Recall = cuan completos son los resultados

# ##### f1_score_test = 96% #####

#%%

### Cuva ROC ###

# y_prob_1 = final_svm.predict_proba(X_test_sd)[:,-1] # probabilidades de target = 1 (positivas)

# fpr, tpr, thresholds = roc_curve(y_test, y_prob_1)
# # fpr = FP / (FP + TN)
# # tpr = TP / (TP + FN)
# roc_auc = auc(fpr, tpr) # Area Under the Curve


# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('Tasa de Falsos Positivos')
# plt.ylabel('Tasa de Verdaderos Positivos')
# plt.title('Curva ROC')
# plt.legend(loc='lower right')
# plt.show()



