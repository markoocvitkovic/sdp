import os
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile, mutual_info_classif, f_classif
from sklearn.metrics import accuracy_score, f1_score
from mlxtend.feature_selection import SequentialFeatureSelector

def ucitaj_arff(putanja_do_datoteke):
    data, meta = arff.loadarff(putanja_do_datoteke)
    df = pd.DataFrame(data)
    return df

def obradi_podatke(df, ciljna_varijabla):
    X = df.drop(columns=[ciljna_varijabla])
    y = df[ciljna_varijabla]

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    return X.values, y

def pearson_korelacija(X, y):
    korelacije = []
    for i in range(X.shape[1]):
        korelacija = np.corrcoef(X[:, i], y)[0, 1]
        korelacije.append(abs(korelacija))
    return korelacije

def evaluiraj_model(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier()  
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    točnost = accuracy_score(y_test, y_pred)
    f1_mjera = f1_score(y_test, y_pred, average='weighted')  
    return točnost, f1_mjera

ciljna_varijabla = 'Defective' 
broj_značajki = [2, 5, 10, 20]  

putanja_do_direktorija = '/content'
for ime_datoteke in os.listdir(putanja_do_direktorija):
    if ime_datoteke.endswith('.arff'):
        putanja_do_datoteke = os.path.join(putanja_do_direktorija, ime_datoteke)
        print(f"\nObrada skupa podataka: {ime_datoteke}")

        df = ucitaj_arff(putanja_do_datoteke)
        X, y = obradi_podatke(df, ciljna_varijabla)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 1. Slijedna pretraga unaprijed (SFS)
        sfs_rezultati = {}
        for k in broj_značajki:
            model = KNeighborsClassifier()
            sfs = SequentialFeatureSelector(model, k_features=k, forward=True, scoring='accuracy', cv=5)
            sfs = sfs.fit(X_train, y_train)
            X_train_sfs = sfs.transform(X_train)
            X_test_sfs = sfs.transform(X_test)
            točnost_sfs, f1_sfs = evaluiraj_model(X_train_sfs, X_test_sfs, y_train, y_test)
            sfs_rezultati[k] = (točnost_sfs, f1_sfs)

        print("Rezultati SFS-a:")
        for k, (točnost, f1_mjera) in sfs_rezultati.items():
            print(f'Točnost s {k} značajki: {točnost:.4f}, F1-mjera: {f1_mjera:.4f}')

        korelacije = pearson_korelacija(X_train, y_train)
        sortirani_indeksi = np.argsort(korelacije)[::-1]  

        print("Rezultati Pearsonove korelacije:")
        for k in broj_značajki:
            top_indeksi = sortirani_indeksi[:k]
            X_train_pearson = X_train[:, top_indeksi]
            X_test_pearson = X_test[:, top_indeksi]
            točnost_pearson, f1_pearson = evaluiraj_model(X_train_pearson, X_test_pearson, y_train, y_test)
            print(f'Točnost s {k} značajki (Pearsonova korelacija): {točnost_pearson:.4f}, F1-mjera: {f1_pearson:.4f}')

        mi_rezultati = {}
        for k in broj_značajki:
            selector_mi = SelectKBest(mutual_info_classif, k=k)
            selector_mi.fit(X_train, y_train)
            X_train_mi = selector_mi.transform(X_train)
            X_test_mi = selector_mi.transform(X_test)
            točnost_mi, f1_mi = evaluiraj_model(X_train_mi, X_test_mi, y_train, y_test)
            mi_rezultati[k] = (točnost_mi, f1_mi)

        print("Rezultati zajedničke informacije:")
        for k, (točnost, f1_mjera) in mi_rezultati.items():
            print(f'Točnost s {k} značajki: {točnost:.4f}, F1-mjera: {f1_mjera:.4f}')

        anova_rezultati_k = {}
        for k in broj_značajki:
            selector_anova = SelectKBest(f_classif, k=k)
            selector_anova.fit(X_train, y_train)
            X_train_anova = selector_anova.transform(X_train)
            X_test_anova = selector_anova.transform(X_test)
            točnost_anova, f1_anova = evaluiraj_model(X_train_anova, X_test_anova, y_train, y_test)
            anova_rezultati_k[k] = (točnost_anova, f1_anova)

        print("Rezultati ANOVA F-testa (SelectKBest):")
        for k, (točnost, f1_mjera) in anova_rezultati_k.items():
            print(f'Točnost s {k} značajki: {točnost:.4f}, F1-mjera: {f1_mjera:.4f}')

       
