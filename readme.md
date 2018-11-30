## Données
Les données d'entraînement et de test devraient être placées dans un dossier data à la racine du projet.

## CNNs Ensemble
### Entraîner les CNNs
Dans le dossier CNNs, un fichier nommé models.config contient les spécifications des modèles que nous voulons entraîner.

Pour exécuter l'entraînement
```sh
python CNNs/training.py
```

### Faire une prédiction
Les modèles se retrouvant dans models.config vont être utilisés pour faire une prédiction. Si le fichier de config contient plus d'un modèle, une moyenne des prédictions sera faite pour obtenir une prédiction d'ensemble.

Pour produire un fichier csv contenant les prédictions
```sh
python CNNs/predict.py
```

Le fichier résultant devrait se situer dans le dossier dans le dossier suivant:

.
+-- CNNsEnsemble
|   +-- log
|       +-- quickdraw
|           +-- predictions.csv

## Baseline SVM


