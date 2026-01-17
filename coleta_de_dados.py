# %%
# Chembl possui uma biblioteca oficial em python
# o que facilita as requisições
# pip install rdkit
# pip install scikit-learn
from funcoes import morgan_fp
from funcoes import fp_to_array

import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import AllChem


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
df_raio3_1024['morgan'] = df_raio3_1024['mol'].apply(
    morgan_fp, raio=3, fpsize=1024)
df_raio2_2048['morgan'] = df_raio2_2048['mol'].apply(
    morgan_fp, raio=3, fpsize=2048)


# %%
# Verificando se os dataframes funcionaram usando a função GetNumBits
print(df_raio2_512.iloc[0, 3].GetNumBits())
print(df_raio3_1024.iloc[0, 3].GetNumBits())
print(df_raio3_2048.iloc[0, 3].GetNumBits())

# %%
# Agora temos que transformar essas subestruturas numa frequencia para o scikit
