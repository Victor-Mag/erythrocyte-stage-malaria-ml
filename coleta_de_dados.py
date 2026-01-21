# %%
# Chembl possui uma biblioteca oficial em python
# o que facilita as requisições
# pip install rdkit
# pip install scikit-learn
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
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

# %%
data_xlsx_jh = pd.read_excel(
    'dataset\dataset-malaria-johnhopkins.xlsx', index_col=None)
data_xlsx_jh.to_csv('dataset\dataset_malaria_jh.csv', encoding='utf-8')

data_xlsx_stj = pd.read_excel(
    'dataset\dataset-malaria-stjude.xlsx', index_col=None
)
data_xlsx_stj.to_csv('dataset\dataset_malaria_stj.csv', encoding='utf-8')


# %%
# Aqui faremos merge de dados positivos de stjude que não sejam iguais aos já
# Existentes do John Hopkins -> 21/01
# Para ter uma proporção de 40% de moleculas ativas, são necessarias
# 656 moleculas ativas adicionadas
df_origem = pd.read_csv('dataset\dataset_malaria_stj.csv')
df_transferencia = df_origem.head(656)


# %%
# Transformando o dataset criado em csv num dataframe e manipulando só o essencial

df_malaria_jh = pd.read_csv("dataset\dataset_malaria_jh.csv")
# Transferindo as moleculas ativas para o df contendo todas
df_malaria = pd.concat([df_malaria_jh, df_transferencia], ignore_index=True)
df_malaria.tail()

# %%
df_malaria.shape

# %%
boolean = df_malaria.duplicated(subset=['Canonical_Smiles']).any()
print(boolean)

# %%
df_malaria.drop_duplicates(subset=['Canonical_Smiles'], inplace=True)
df_malaria.shape

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
contagem = df_malaria[df_malaria['status'].isin(
    ['Active', 'Actives'])]['status'].value_counts().sum()

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
df_raio3_1024['morgan'] = df_raio3_1024['mol'].apply(
    morgan_fp, raio=3, fpsize=1024)
df_raio3_2048['morgan'] = df_raio3_2048['mol'].apply(
    morgan_fp, raio=3, fpsize=2048)


# %%
# Verificando se os dataframes funcionaram usando a função GetNumBits
print(df_raio2_512.iloc[0, 3].GetNumBits())
print(df_raio3_1024.iloc[0, 3].GetNumBits())
print(df_raio3_2048.iloc[0, 3].GetNumBits())

# %%
# Agora temos que transformar essas subestruturas numa frequencia para o scikit
X_r2_512 = np.vstack(df_raio2_512['morgan'].apply(fp_to_array))
X_r2_1024 = np.vstack(df_raio2_1024['morgan'].apply(fp_to_array))
X_r2_2048 = np.vstack(df_raio2_2048['morgan'].apply(fp_to_array))
X_r3_512 = np.vstack(df_raio3_512['morgan'].apply(fp_to_array))
X_r3_1024 = np.vstack(df_raio3_1024['morgan'].apply(fp_to_array))
X_r3_2048 = np.vstack(df_raio3_2048['morgan'].apply(fp_to_array))

# %%

print(f"Raio 2, 512 bits: {X_r2_512.shape}")
print(f"Raio 3, 2048 bits: {X_r3_2048.shape}")

# %%
# Mudando as expressoes de active == 1 e inactive == 0
mapa = {'Active': 1, 'Inactive': 0, 'Actives': 1}

y = df_malaria['status'].map(mapa)

# %%
print(y)

# %%
# Inicialmente vamos dividir o primeiro conjunto de dados
# (RAIO 2 E 512 bits   )

X_train, X_test, y_train, y_test = train_test_split(
    X_r2_512, y, test_size=0.2, random_state=42
)

print(f"Tamanho da amostra de treino: {X_train.shape}")
print(f"Tamanho da amostra de teste: {X_test.shape}")

# %%
# Treinando inicialmente usando Random Forest

modelo_r2_512 = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_r2_512.fit(X_train, y_train)

# %%
scores = ['accuracy', 'roc_auc', 'f1', 'recall']

n_scores = cross_validate(modelo_r2_512, X_train,
                          y_train, scoring=scores, cv=10, n_jobs=-1)

for i in n_scores.keys():
    print(f"{i} = {np.mean(n_scores[i])}")

# %%
# Verificando a performance do modelo com raio 2 e 512 bits para
# outros modelos de classificação

lista_algoritmos = ['SVM', 'KNN', 'XGBoost', 'MLP', 'Regressao Logistica']

# %%
val_cruzada(lista_algoritmos, X_train, y_train)
# MLP e XGBoost apresentam as melhores métricas de recall
# Portanto seguiremos com eles, já que um dos problemas na primeira versão
# foi um recall extremamente baixo -> 0,30

# %%
# Antes de realizar qualquer fine-tuning, devemos verificar qual dataframe
# tem os melhores resultados -> linha 125; linha 130
from xgboost import XGBClassifier
# Criação do dicionario para facilitar a identificação dos resultados
# após iterações
scoring = ['recall', 'accuracy', 'f1', 'roc_auc']

dict_fingerprints = {'Morgan R2 512': X_r2_512,
                                'Morgan R2 1024': X_r2_1024,
                                'Morgan R2 2048': X_r2_2048,
                                'Morgan R3 512': X_r3_512,
                                'Morgan R3 1024': X_r3_1024,
                                'Morgan R3 2048': X_r3_2048}

melhor_recall = {'nome': '', 'score_recall': 0 }
melhor_f1 = {'nome': '', 'score_f1': 0}
for nome, x_atual in dict_fingerprints.items():
    X_train, X_test, y_train, y_test = train_test_split(
    x_atual, y, test_size=0.2, random_state=42)
    
    modelo = XGBClassifier(scale_pos_weight =(((len(df_malaria)-contagem)/contagem))) # Treinamento do modelo
    # Validação do modelo dentro do loop verificando o nome pelo par
    n_scores = cross_validate(
            modelo, X_train, y_train, scoring=scoring, cv=10, n_jobs=-1) 
    print(f"Modelo {nome}")

    if melhor_recall['score_recall'] < np.mean(n_scores['test_recall']):
        melhor_recall['nome'] = nome
        melhor_recall['score_recall'] = np.mean(n_scores['test_recall']) 

    if melhor_f1['score_f1'] < np.mean(n_scores['test_f1']):
        melhor_f1['nome'] = nome
        melhor_f1['score_f1'] = np.mean(n_scores['test_f1'])

    for i in n_scores.keys():
        print(f"AVG {i} = {np.mean(n_scores[i])}")
        print(f"NP {i} = {np.std(n_scores[i])}\n")

# %%
# Apresentando melhores modelos de recall e f1-score
# R2 2048 é o mais interessante
print(melhor_f1)
print(melhor_recall)

# %%
# Realização de três métodos de fine-tuning de hiperparametros
# 1. Grid search
# 2. Randomized Search
# 3. Bayesian Optimization

parametros_grid = {'C': [0.1, 1, 10],
                   'kernel': ['rbf', 'linear'],
                   'gamma': [0.1, 0.01, 0.001],
                   'class_weight': ['balanced']}

# Agora um objeto deve ser instanciado com os parametros nele

grid_search = GridSearchCV(xgb.XGBClassifier(), parametros_grid,
                           cv=10, scoring='f1', n_jobs=-1, verbose=0)

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
modelo_svm = svm.SVC(C=0.5, class_weight='balanced', gamma=0.1, kernel='rbf')
modelo_svm.fit(X_train, y_train)

# %%
previsoes = modelo_r2_512.predict(X_test)

conf_matrix = confusion_matrix(y_test, previsoes)
sns.heatmap(conf_matrix, fmt='d', annot=True, cmap='Blues')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# Existem muitos falsos negativos
# Porém, o modelo não deu nenhum falso positivo o que é um bom sinal
# Devemos ajustar o gridsearch para trazer tentar otimizar isso ou buscar equilibrar
# O dataset
