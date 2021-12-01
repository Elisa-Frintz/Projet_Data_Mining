#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programme 2
Objectif : Produire le score d'appartenance à la classe "m16" de la variable cible 'V200'.
"""

# --------------------- Importation des librairies ---------------------
import os 
import pandas
import numpy 
import scipy
import sklearn # Version de sklearn : 0.23.1
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics

# --------------------- Fonctions ---------------------

""" Fonction qui importe les données à partir d'un chemin menant à un dataframe : Importation_data(dataframe)
 Input :
   - dataframe : le chemin et le nom du dataframe que l'on veut importé
 Output : 
   - data : le dataframe importé
   - chemin : le répertoire courant (c'est le répertoire où se trouve le dataframe)
"""
def Importation_data(dataframe):
    data = pandas.read_table(dataframe, sep="\t", header=0)
    chemin = os.path.dirname(dataframe) # On retrouve le répertoire parent du fichier
    os.chdir(chemin) # On se place dans ce répertoire parent
    return data, chemin

""" Fonction qui supprime les variables qui ont un nombre de modalité <= 1 car elles n'apportent 
 aucune information : suppr_var_sans_information(dataframe)
 Input :
   - dataframe : le nom du dataframe que l'on veut traiter 
 Output : 
   - data : le dataframe traité
"""

def suppr_var_sans_information(dataframe):
    # Parcourir les colonnes du dataframe :
    for i in dataframe.columns:
        # Si nb de modalité <= 1, on supprime la variable :
        if len(dataframe[i].unique()) <= 1 : 
            dataframe.drop([i], axis='columns', inplace=True)
    return dataframe

""" Fonction pour définir la variable cible (y) et les variables explicatives (X) : Var_cible_expl(dataframe)
 Input :
   - dataframe : le dataframe à définir
 Output : 
   - X : le dataframe des variables explicatives
   - y : la variable cible
"""
def Var_cible_expl(dataframe):
    # On récupère tout les noms de colonnes du dataframe dans une liste col
    col = [i for i in dataframe.columns]
    
    # On enlève "V200" de la liste col
    col.remove("V200")
    
    # La colonne "V200" est la variable cible, les autres sont les variables explicatives
    X = dataframe.loc[:,col]
    y = dataframe.V200
    return X, y

#dataframe = "/home/elisa/Documents/M1_Info/Semestre_1/Data_Mining/Projet_Data_Mining/data_essai.txt"

# --------------------- Fonction principale ---------------------


def main():
    # ---------- Paramètres fixés ----------
    taille_test = 0.25
    nb_var = 15
    
    dataframe = zone_text_importation.get()
    print(dataframe)
    # ---------- Importation des données et définition du répertoire courant ----------
    
    # Importation des données et définition du répertoire courant (celui où se trouve "data_avec_etiquettes.txt")
    deploiement, chemin = Importation_data(dataframe)
    print("Le répertoire courant est : " , chemin)
    
    # --------------------- Entrainement du modèle ---------------------

    # Récupération du dataframe "data_avec_etiquettes.txt" dans le répertoire courant
    data_brute = pandas.read_table("data_avec_etiquettes.txt", sep="\t", header=0)
    
    # ---------- Traitement des données "data_avec_etiquettes.txt" ----------
    
    # Recodage des variables qualitatives V160, V161 et V162 en quantitatives
    data = pandas.get_dummies(data_brute.iloc[:,0:199], prefix=['V160', 'V161', 'V162'])
    data = data.join(data_brute["V200"])
    #data.head(5)
    
    # Recodage de la variable cible : « m16 » (les positifs)
    data.V200 = pandas.get_dummies(data_brute.V200).m16

    # Supprimer les variables sans information (c'est-à-dire nombre de modalité <= 1)
    data = suppr_var_sans_information(data)
    
    # ---------- Subdivision en deux dataset (dataTrain et dataTest) ----------
    dataTrain, dataTest = model_selection.train_test_split(data, test_size = taille_test, random_state = 0)
    
    # Verification
    print("Dimentions dataTrain :", dataTrain.shape)
    print("Dimentions dataTest :", dataTest.shape)
    
    # ---------- Définir la variable cible (y) et les variables explicatives (X) ----------
    # Pour l'échantillon dataTrain
    XTrain, yTrain = Var_cible_expl(dataTrain)
    # Pour l'échantillon dataTest
    XTest, yTest = Var_cible_expl(dataTest)
    
    # ---------- Selection de variables ----------
    
    # On séléctionne les k = 15 variables les plus pertinentes à l'aide d'un test du chi2
    selector = SelectKBest(chi2, k = nb_var)

    # On applique la séléction de variables sur le dataset d'entrainement 
    selector.fit_transform(XTrain, yTrain) 

    selector.get_support() # Renvoie un booléen

    print("Les variables séléctionnées sont : \n", XTrain.columns[selector.get_support()])
    
    # ---------- Entrainement du modèle (Arbre de décision)¶ ----------
    
    # Instanciation de l'arbre de décision (pour avoir les mêmes valeurs à chaque fois)
    tree = DecisionTreeClassifier(random_state = 0)
    
    # Transformer les données d'apprentissage à l'aide des variables séléctionnées
    XTrain = XTrain.loc[:,selector.get_support()]
    
    # Application du modèle sur les données d'apprentissage
    tree.fit(XTrain,yTrain)
    
    # --------------------- Prédictions sur la base de déploiement ---------------------
 
    # ---------- Traitement des données de deploiement ----------
    # Recodage des variables qualitatives V160, V161 et V162 en quantitatives
    deploiement = pandas.get_dummies(deploiement, prefix=['V160', 'V161', 'V162'])

    # Supprimer les variables sans information (c'est-à-dire nombre de modalité <= 1)
    deploiement = suppr_var_sans_information(deploiement)
    
    Xdéploiement = deploiement.loc[:,selector.get_support()]
    
    # ---------- Scoring et Prédiction ----------
    # Probas de prédictions
    probas_de_prédictions = tree.predict_proba(Xdéploiement)
    
    # Prediction
    prédictions = tree.predict(Xdéploiement)
    #print(prédictions)
    
    # Score de 'positif'
    score = probas_de_prédictions[:,1]
    
    # --------------------- Sauvegarder "scores.txt" dans le répertoire courant ---------------------
    # Création du dataframe à enregister
    df = pandas.DataFrame({'index' : Xdéploiement.index, 'score': [i for i in score] , 'Prédictions': [j for j in prédictions]})
    
    # Ecriture du fichier "scores.txt" dans le répertoire courant 
    df.to_csv("scores.txt", sep="\t", encoding="utf-8", index=False)
    
    print("--> Le fichier 'scores.txt' est dans votre répertoire courant : ", chemin)


    



#main()

""" 
------------------------------- Création d'une interface avec Tkinter -------------------------------
"""

# Importation de Tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename


###################################         INTERFACE UTILISATEUR      ######################################  

##### Fenetre principale avec ses caractéristiques
fenetre = Tk()
fenetre.title("Programme 2 - Data Mining")
fenetre.config(background='skyblue')

##### Fonction permettant d'ouvrir l'explorateur de fichier
def chercherFichier():
    filename = filedialog.askopenfilename(filetypes=(("Fichiers TXT","*.txt"),)) # Ouvre l'explorateur de fichier
    zone_text_importation.delete(0, END) # Vide le champs de texte
    zone_text_importation.insert(0, filename) # Puis le rempli par le chemin du nouveau fichier

# Grand titre 
Indic_crit = Label(fenetre, text="Programme 2 - Data Mining \n",fg = "darkblue", bg='skyblue', font = ("Helvetica", 20 , "bold"))
Indic_crit.grid(row=0, column=0, columnspan=2)

##### Objectif
label_objectif = Label(fenetre, justify=LEFT, text="Objectif : Produire le score d'appartenance à la classe \"m16\" de la variable cible 'V200'. \n",fg = "DodgerBlue4", bg='skyblue', font = ("Arial", 18 , "normal"))
label_objectif.grid(row=1, column=0, columnspan=2, sticky="w")

#### Importation du fichier
label_importation_desc = Label(fenetre, text="Importation de la base de déploiement :", fg = "DodgerBlue4", bg='skyblue', font = ("Arial", 18 , "underline"))
label_importation_desc.grid(row=2, column=0, sticky="w")

zone_text_importation = Entry(fenetre, width=100)
zone_text_importation.grid(row=3, column=0, sticky="we")

# Bouton pour chercher un fichier
btn_chercher_fichier = Button(fenetre, text="Chercher un fichier", command=chercherFichier)
btn_chercher_fichier.grid(row=3, column=1, sticky="w")

#### Bouton valider pour lancer le traitement
btn_valider = Button(fenetre, text="Valider", command=main, width=10)
btn_valider.grid(row=4, column=0)

#### Méthode
label_methode = Label(fenetre, justify=LEFT, text="Avec la méthode de scoring Arbre de Décision, le nombre de positifs parmi les 10 000 observations \n qui présentent les scores les plus élevés dans une base de 4 898 424 observations est annoncé à 267. \n",fg = "DodgerBlue4", bg='skyblue', font = ("Arial", 18 , "normal"))
label_methode.grid(row=5, column=0, columnspan=2, sticky="w")

#### Information sur le nouveau fichier créé
label_info = Label(fenetre, text="Le fichier \"scores.txt\" sera disponible dans votre répertoire courant à la fin du \n traitement.\n",
                         fg = "DodgerBlue4", bg='skyblue', font = ("Arial", 18 , "bold"))
label_info.grid(row=6, column=0, columnspan=2)

#### Pied de page
label_footer = Label(fenetre, borderwidth=1, relief="groove", text="Bourdin Yvan & Frintz Elisa - Data Mining - M1 Informatique - Promo 2020-2021", fg = "darkblue", bg='skyblue', font = ("Arial", 15 , "italic"))
label_footer.grid(row=7, column=0, columnspan=2, sticky="nesw")

# On démarre la boucle Tkinter qui s'interompt quand on ferme la fenêtre
fenetre.mainloop()



