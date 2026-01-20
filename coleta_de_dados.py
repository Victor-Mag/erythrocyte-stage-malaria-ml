# %%
# Chembl possui uma biblioteca oficial em python
# o que facilita as requisições
# pip install rdkit
# pip install scikit-learn
from funcoes import morgan_fp
from funcoes import fp_to_array
from funcoes import val_cruzada

import pandas as pd
import numpy as np

from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split


# %%
data_xlsx = pd.read_excel(
    'dataset\dataset-malaria-johnhopkins.xlsx', index_col=None)
data_xlsx.to_csv('dataset\dataset_malaria_jh.csv', encoding='utf-8')

# %%
# Transformando o dataset criado em csv num dataframe e manipulando só o essencial

df_malaria = pd.read_csv("dataset\dataset_malaria_jh.csv")
df_malaria.tail()


# %%
# Temos que dropar a coluna PROJECT e DataSet
# Depois devemos manter apenas o Canonical Smiles e Mudar a coluna Library para 'atividade'
# E mudar o status 'Active' para '1' e Inactive para '0', drogas marcadas com '1'
# Apresentam atividade anti-malária. Active -> valor de EC50 < 10 µM

df_malaria = df_malaria.drop(
    columns=['Index', 'PROJECT', 'DataSet', 'Unnamed: 0'])
df_malaria
# %%
# Mudando o nome da coluna library pro status
df_malaria.rename(columns={'LIBRARY': 'status',
                  'Canonical_Smiles': 'canonical_smiles'}, inplace=True)
df_malaria.head()
# %%
# Compreendendo quantidade de moléculas ativas e inativas
contagem = df_malaria['status'].value_counts()['Active']
# %%
print(
    f'Existem {contagem} compostos ativos e {len(df_malaria)-contagem} inativos')
print(
    f"Razão de compostos ativos: {(contagem/(len(df_malaria)-contagem)):.2f}")


# %%
# Transformando canonical_smiles no objeto 'Mol' que pode ser interpretado
# facilmente pelo RDKit
df_malaria['mol'] = df_malaria['canonical_smiles'].apply(Chem.MolFromSmiles)
df_malaria['mol'].head()


# %%
# Criando 6 dataframes com hiperparametros de morgan diferentes
df_raio2_512 = df_malaria.copy()
df_raio2_1024 = df_malaria.copy()
df_raio2_2048 = df_malaria.copy()

df_raio3_512 = df_malaria.copy()
df_raio3_1024 = df_malaria.copy()
df_raio3_2048 = df_malaria.copy()

# %%
# Calculando hiperparametros de morgan diferentes para cada um dos respectivos dataframes
df_raio2_512['morgan'] = df_raio2_512['mol'].apply(morgan_fp)
df_raio2_1024['morgan'] = df_raio2_1024['mol'].apply(morgan_fp, fpsize=1024)
df_raio2_2048['morgan'] = df_raio2_2048['mol'].apply(morgan_fp, fpsize=2048)

df_raio3_512['morgan'] = df_raio3_512['mol'].apply(morgan_fp, raio=3)
df_raio3_1024['morgan'] = df_raio3_1024['mol'].apply(morgan_fp, raio=3, fpsize=1024)
df_raio3_2048['morgan'] = df_raio3_2048['mol'].apply(morgan_fp, raio=3, fpsize=2048)


# %%
# Verificando se os dataframes funcionaram usando a função GetNumBits
print(df_raio2_512.iloc[0, 3].GetNumBits())
print(df_raio3_1024.iloc[0, 3].GetNumBits())
print(df_raio3_2048.iloc[0, 3].GetNumBits())

# %%
# Agora temos que transformar essas subestruturas numa frequencia para o scikit
X_r2_512  = np.vstack(df_raio2_512['morgan'].apply(fp_to_array))
X_r2_1024 = np.vstack(df_raio2_1024['morgan'].apply(fp_to_array))
X_r2_2048 = np.vstack(df_raio2_2048['morgan'].apply(fp_to_array))
X_r3_512  = np.vstack(df_raio3_512['morgan'].apply(fp_to_array))
X_r3_1024 = np.vstack(df_raio3_1024['morgan'].apply(fp_to_array))
X_r3_2048 = np.vstack(df_raio3_2048['morgan'].apply(fp_to_array))

# %%

print(f"Raio 2, 512 bits: {X_r2_512.shape}")
print(f"Raio 3, 512 bits: {X_r3_2048.shape}")

# %%
# Mudando as expressoes de active == 1 e inactive == 0
mapa = {'Active': 1, 'Inactive': 0}

y = df_malaria['status'].map(mapa)

# %%
print(y)

# %%
# Inicialmente vamos dividir o primeiro conjunto de dados 
# (RAIO 2 E 512 bits   )

X_train, X_test, y_train, y_test = train_test_split(
    X_r2_512, y, test_size=0.2, random_state=42 
)

print (f"Tamanho da amostra de treino: {X_train.shape}")
print (f"Tamanho da amostra de teste: {X_test.shape}")

# %% 
# Treinando inicialmente usando Random Forest
from sklearn.ensemble import RandomForestClassifier

modelo_r2_512 = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_r2_512.fit(X_train, y_train)

# %%
from sklearn.model_selection import cross_val_score

n_scores = cross_val_score(modelo_r2_512, X_train, y_train, scoring='accuracy',cv=10, n_jobs=-1)

print("Acurácias:", n_scores)
print("\nMédia das acurácias obtidas:", n_scores.mean())
print("\nDesvio padrão das acurácias obtidas:", n_scores.std())

# %%
# Fine tuning do random forest

par_grid={}


# %%
# Verificando a performance do modelo com raio 2 e 512 bits para 
# outros modelos de classificação

lista_algoritmos = ['SVM', 'KNN', 'XGBoost', 'MLP', 'Regressao Logistica']

# %%
val_cruzada(lista_algoritmos, X_train, y_train)
# SVM E KNN Saem na frente aqui, com uma leve vantagem para o SVM
# Uma vez que ao aumentar a dimensionalidade, o KNN perde força
# e esses dados tratados foram apenas de 512 bits.


# %%
# Fazendo um grid search para o SVM já que obteve os melhores 
# resultados de benchmark
# Os hiperparametros selecionados tiveram com base a alta quantidade
# de amostras negativas (90%)

parametros_grid ={'C':[0.1,1,10],
                  'kernel': ['rbf', 'linear'],
                  'gamma': [0.1, 0.01, 0.001],
                  'class_weight': ['balanced']}

# Agora um objeto deve ser instanciado com os parametros nele

from sklearn.model_selection import GridSearchCV
from sklearn import svm

grid_search = GridSearchCV(svm.SVC(),parametros_grid, cv=10, scoring='f1', n_jobs=-1,verbose = 0)

grid_search.fit(X_train, y_train)

# %%
print("Melhor combinação de hiperparâmetros:")
print(grid_search.best_params_)

print("\nMelhor recall obtida:")
print(grid_search.best_score_)

# Aparentemente os valores default do svm 
# superaram os hiperparametros selecionados
# Porém o hiperparâmetro 'class_weight' default tem menos valor
# É válido rodar uma matriz de confusão para as duas formas
# %%
modelo_svm = svm.SVC(C=0.5,class_weight = 'balanced',gamma=0.1, kernel='rbf')
modelo_svm.fit(X_train,y_train)

# %%
previsoes = modelo_r2_512.predict(X_test)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

conf_matrix = confusion_matrix(y_test, previsoes)
sns.heatmap(conf_matrix, fmt='d', annot=True, cmap='Blues')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

## Existem muitos falsos negativos
# Porém, o modelo não deu nenhum falso positivo o que é um bom sinal
# Devemos ajustar o gridsearch para trazer tentar otimizar isso ou buscar equilibrar
# O dataset



