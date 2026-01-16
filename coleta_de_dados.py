# %%
# Chembl possui uma biblioteca oficial em python
# o que facilita as requisições
# pip install rdkit
# pip install scikit-learn
import pandas as pd
from chembl_webresource_client.new_client import new_client


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
print(f'Existem {contagem} compostos ativos e {len(df_malaria)-contagem} inativos')
print(f"Razão de compostos ativos: {(contagem/(len(df_malaria)-contagem)):.2f}")
