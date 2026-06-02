# 🚀 PROJET JAVA - THS : GUIDE TECHNIQUE ET COMPTE-RENDU (ÉTAPE 1)

Ce fichier centralise l'arborescence, les commandes d'installation, la procédure de compilation et l'analyse scientifique des tests du neurone sur les fonctions logiques.

---

## 📂 1. Architecture du Projet
Pour que les commandes fonctionnent, l'arborescence dans VS Code doit être exactement celle-ci :

```text
PROJET JAVA - THS/
└── Support/
    ├── Image.java
    └── neurone/
        ├── iNeurone.java
        ├── Neurone.java
        ├── NeuroneHeavyside.java
        └── testNeurone.java
```

---

## 🛠️ 2. Configuration Initiale (Environnement WSL Ubuntu)
Si la commande de compilation ne fonctionne pas, c'est que Java n'est pas installé sur votre terminal Linux. 
Exécutez cette commande une seule fois :

```bash
sudo apt update && sudo apt install default-jdk -y
```

---

## 💻 3. Protocole de Test des Fonctions Logiques
Pour tester le neurone, il faut modifier le fichier `Support/neurone/testNeurone.java`. 
À l'intérieur, décommentez **une seule ligne `resultats` à la fois** 
selon le test que vous voulez lancer, et sauvegardez le fichier (Ctrl+S).

**Pour tester le ET :**
```java
final float[] resultats = {0, 0, 0, 1}; // S'allume si les deux entrées valent 1
// final float[] resultats = {0, 1, 1, 1}; 
// final float[] resultats = {0, 1, 1, 0}; 
```

**Pour tester le OU :**
```java
// final float[] resultats = {0, 0, 0, 1}; 
final float[] resultats = {0, 1, 1, 1}; // S'allume si au moins une entrée vaut 1
// final float[] resultats = {0, 1, 1, 0}; 
```

**Pour tester le XOR :**
```java
// final float[] resultats = {0, 0, 0, 1}; 
// final float[] resultats = {0, 1, 1, 1}; 
final float[] resultats = {0, 1, 1, 0}; // S'allume si une seule entrée vaut 1
```

---

## 🚀 4. Commandes d'Exécution
À faire dans le terminal après CHAQUE modification du code Java. Lancez ces trois commandes à la suite :

**1. Se placer dans le bon dossier :**
```bash
cd Support/neurone
```

**2. Nettoyer les anciens fichiers et recompiler :**
```bash
rm -f *.class && javac *.java
```

**3. Lancer le test :**
```bash
java testNeurone
```

---

## 📊 5. Résultats et Interprétation Scientifique (Analyse)

### 🟢 Résultat du test ET
* **Validation :** L'erreur globale (`mse`) converge et tombe à `0.000000`.
* **Interprétation :** Le neurone trouve un **biais fortement négatif**. 
Il exige donc que les deux entrées soient actives simultanément pour que leur somme dépasse le seuil d'activation (0) et renvoie 1.

### 🟢 Résultat du test OU
* **Validation :** L'erreur globale (`mse`) converge et tombe à `0.000000`.
* **Interprétation :** Le neurone trouve un **biais très proche de zéro**. 
Ainsi, la moindre activation (un seul 1 en entrée) génère une somme positive suffisante pour déclencher la fonction Heaviside.

### 🔴 Résultat du test XOR
* **Validation :** L'algorithme boucle à l'infini (le `mse` reste bloqué à 0.25).
* **Interprétation :** Comportement normal. Un perceptron simple est mathématiquement incapable 
d'apprendre un problème non-linéaire (on ne peut pas tracer une seule ligne droite pour séparer les réponses du XOR).

### 🔬 Conclusion Générale (Robustesse et Poids)
À chaque exécution du ET ou du OU, le programme trouve des valeurs de synapses et de biais différentes tout en gardant 100% de réussite. Ces fonctions étant linéairement séparables, il existe une infinité de plans de séparation possibles. 
L'algorithme s'arrête simplement dès qu'il franchit l'un d'eux au hasard de son initialisation.