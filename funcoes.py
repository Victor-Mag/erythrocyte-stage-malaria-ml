import numpy as np
import xgboost as xgb
from rdkit.Chem import AllChem
from rdkit import DataStructs


def morgan_fp(mol, raio=2, fpsize=512):
    morgan = AllChem.GetMorganGenerator(radius=raio, fpSize=fpsize)
    return morgan.GetFingerprint(mol)


def fp_to_array(fp):
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def val_cruzada(lista_algoritmos, X, y):
    from tqdm.notebook import tqdm
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    from sklearn import svm

    print("Valores de acurácia média para validação cruzada 10x:")
    for a in tqdm(lista_algoritmos):
        if a == 'SVM':
            modelo = svm.SVC()
        elif a == 'KNN':
            modelo = KNeighborsClassifier()
        elif a == 'XGBoost':
            modelo = xgb.XGBClassifier()
        elif a == 'MLP':
            modelo = MLPClassifier()
        elif a == 'Regressao Logistica':
            modelo = LogisticRegression()

# Executando a validacao cruzada 10x
        n_scores = cross_val_score(modelo, X, y, scoring='accuracy', cv= 10, n_jobs=-1)
        media_acuracias = n_scores.mean()
        desvio_padrao = n_scores.std()
    
        print(f"{a}: {media_acuracias} (Desvio Padrão: {desvio_padrao})")


