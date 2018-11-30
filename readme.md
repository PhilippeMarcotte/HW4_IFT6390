## Données
Les données d'entraînement et de test devraient être placées dans un dossier data à la racine du projet.

## CNNs Ensemble
### models.config
Dans le dossier CNNsEnsemble, un fichier nommé models.config contient les spécifications des modèles que nous voulons entraîner ou utiliser pour faire une prédiction.

### Entraîner les CNNs
Pour exécuter l'entraînement
```sh
cd CNNsEnsemble
python training.py
```

### Faire une prédiction
Les modèles se retrouvant dans models.config vont être utilisés pour faire une prédiction. Si le fichier de config contient plus d'un modèle, une moyenne des prédictions sera faite pour obtenir une prédiction d'ensemble. Ce script requiert les checkpoints des modèles dans un dossier log/quickdraw/nom_modele/pth

Pour produire un fichier csv contenant les prédictions
```sh
cd CNNsEnsemble
python predict.py
```

Le fichier résultant devrait se situer dans le dossier ./CNNsEnsemble/log/quickdraw/predictions.csv

## Baseline SVM


