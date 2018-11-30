## Données
Les données d'entraînement et de test devraient être placées dans un dossier data à la racine du projet.

## CNNs Ensemble
### Requis
Installer la librairie pretrainedmodels. Nous n'utilisons pas de modèles préentrainés mais celle-ci contient l'implémentation pytorch de SENet-154.

```sh
pip install pretrainedmodels
```

### models.config
Dans le dossier CNNsEnsemble, un fichier nommé models.config contient les spécifications des modèles que nous voulons entraîner ou utiliser pour faire une prédiction.

### Entraîner les CNNs
Pour exécuter l'entraînement
```sh
python CNNsEnsemble/training.py
```

### Faire une prédiction
Les modèles se retrouvant dans models.config vont être utilisés pour faire une prédiction. Si le fichier de config contient plus d'un modèle, une moyenne des prédictions sera faite pour obtenir une prédiction d'ensemble.

Pour produire un fichier csv contenant les prédictions
```sh
python CNNsEnsemble/predict.py
```

Le fichier résultant devrait se situer dans le dossier ./CNNsEnsemble/log/quickdraw/predictions.csv

## Baseline SVM


