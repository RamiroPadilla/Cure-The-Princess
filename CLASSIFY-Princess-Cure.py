## Cure the princess

# Librerias y metodos utilizados
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import classification_report, roc_curve, auc

# In[1]

### Analisis del dataset ###

df = pd.read_csv('data.csv')
df.head()
df.info()

highest_corr = df.corr().iloc[-1,:-2].idxmax()

# Conteo de casos donde se cura y donde no
cured_vs_not = sns.countplot(data=df, x='Cured')

correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap de Correlación')
plt.show()

# In[2]

### Datos test-training ###

X = df.iloc[:,:-1]
y = df[['Cured']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

# Estandarizacion de datos
sc = StandardScaler()
sc.fit(X_train)
X_train_sd = sc.transform(X_train)
X_test_sd = sc.transform(X_test)

# Transformo en array de 1d
y_train_1d = column_or_1d(y_train) 

# In[3]

### Busqueda de los mejores parametros ###

svm = SVC()
param_grid = {'C' : [0.1, 1 , 5, 10, 100, 1000],
              'gamma' : [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel' : ['rbf', 'linear']
              }

# GridSearch = para c/combinación de hiperparámetros, la evalua haciendo cross-validation (cv)
clf = GridSearchCV(svm, param_grid, cv=5, n_jobs=10) 
# n_jobs = Number of jobs to run in parallel, depende del CPU 
# otra manera podria ser utilizar RandomizedSearchCV

clf.fit(X_train_sd, y_train_1d)
best_params = clf.best_params_

# In[4]

### Entrenamiento modelo final ###

final_svm = SVC(**best_params, probability=True) # paso los mejores parametros como keywords arguments
final_svm.fit(X_train_sd, y_train_1d)


y_pred_train = final_svm.predict(X_train_sd)
y_pred_test = final_svm.predict(X_test_sd)


# Resultados del mejor modelo
final_report = classification_report(y_test, y_pred_test)
print(final_report)
# Precision = cuan utiles son los resultados de busqueda
# Recall = cuan completos son los resultados

##### f1_score_test = 96% #####

# In[5]

### Cuva ROC ###

y_prob_1 = final_svm.predict_proba(X_test_sd)[:,-1] # probabilidades de target = 1 (positivas)

fpr, tpr, thresholds = roc_curve(y_test, y_prob_1)
# fpr = FP / (FP + TN)
# tpr = TP / (TP + FN)
roc_auc = auc(fpr, tpr) # Area Under the Curve


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()



