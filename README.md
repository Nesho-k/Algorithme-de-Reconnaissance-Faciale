# Reconnaissance Faciale

## Introduction 

Dans ce projet, un modèle de reconnaissance faciale a été développé en utilisant TensorFlow et des techniques avancées de deep learning. L’objectif principal est de concevoir un réseau de neurones convolutifs capable de détecter et d'identifier des visages humains à partir d'images, avec une précision optimale, même en présence de variations d'éclairage, de poses ou d'expressions faciales.

Pour ce projet, nous avons créé notre propre jeu de données composé d'images de visages, à l'aide des bibliothèques OpenCV et Albumentations. Nous avons ensuite conçu un modèle de réseau de neurones convolutifs (CNN) en nous basant sur l'architecture VGG16, que nous avons adaptée à notre cas spécifique de reconnaissance faciale. Cette adaptation inclut des modifications au niveau des couches finales afin de mieux correspondre à la tâche de classification des visages.

Enfin, une fois le modèle défini, nous avons procédé à son entraînement, puis à son évaluation sur un ensemble de test afin de mesurer ses performances.

## Tâches à accomplir 

Dans ce projet de reconnaissance faciale, il y a deux types de tâches à accomplir :

**Classification (Présence ou absence d'un visage)**

Dans cette tâche, le modèle doit déterminer si un visage est présent dans l'image ou non. Ce type de problème est appelé classification binaire, car le modèle doit attribuer l'image à l'une des deux classes suivantes :

- Présence d'un visage : l'image contient un ou plusieurs visages.
- Absence de visage : aucun visage n'est détecté dans l'image.
Ici, le modèle génère une probabilité indiquant dans quelle mesure l’image appartient à l’une des deux classes. Par exemple, s’il renvoie une probabilité de 0,85, cela signifie qu’il est relativement sûr que l’image contient un visage.

**Régression (Localisation du visage avec une Bounding Box)**

Une fois que le modèle a détecté la présence d’un visage dans l’image, il doit également localiser précisément où il se trouve. Pour ce faire, le modèle utilise un processus de régression, où il prédit les coordonnées de la bounding box (ou boîte de délimitation) entourant le visage.

Une bounding box est un rectangle qui encadre le visage dans l’image. Elle est généralement définie par :

- x_min et y_min : les coordonnées du coin supérieur gauche du rectangle.
- x_max et y_max : les coordonnées du coin inférieur droit du rectangle.
Le modèle prédit ces valeurs pour chaque visage détecté, ce qui permet de le localiser précisément dans l’image.

**Deux tâches au sein d’un même réseau**

La variable à expliquer y contiendra deux éléments : la classe (visage ou non) ainsi que les coordonnées de la bounding box.

Ainsi, nous pouvons considérer qu’il y a deux modèles intégrés dans le même réseau :

- Un modèle chargé de la classification (présence ou absence de visage). 
- Un modèle chargé de la régression (prédiction des coordonnées de la bounding box).

## Création, annotation et chargement des images 

Nous devons dans un premier temps créer nos propres données. Pour cela nous capturons 120 images à l'aide d'une webcam, ces 120 images peuvent contenir des visages ou non.

Nous ne créons que 120 images, car nous devons ensuite les annoter manuellement à l'aide de la bibliothèque **Labelme**, ce qui peut être une tâche longue. Cependant, pour l'entraînement d'un CNN comme VGG16, 120 images ne sont pas suffisantes. C'est là qu'intervient la bibliothèque Albumentations, qui permet d’augmenter artificiellement le jeu de données en appliquant diverses transformations aux images existantes.

![Capture d’écran 2025-03-06 150625](https://github.com/user-attachments/assets/49a45366-b43a-4082-becd-383c91e89fd3)

## Augmentation des images 

Une fois avoir séparer le jeu de données en trois ensembles : train set, validation et test set, nous devons ensuite augmenter le nombre d'images de notre jeu de données en utilsant la bibliothèque  **Albumentations**. Nous appliquons des transformations aléatoires (recadrage, flips, ajustements de luminosité, contraste, gamma et décalage des canaux RGB) tout en gérant les bounding box associées aux images annotées.

![Capture d’écran 2025-03-06 153547](https://github.com/user-attachments/assets/68f91146-3bfd-4a97-af8b-fead3b8f69ca)

Pour chaque image, nous appliquons 60 transformations aléatoires afin d'augmenter artificiellement notre jeu de données. Sachant que nous disposions initialement de 118 images, nous obtenons un total de 118 * 60 = **7 080 images**.

Cette augmentation est essentielle, car un plus grand nombre d'images améliore la capacité du CNN à généraliser et à apprendre des variations naturelles des visages (éclairage, rotation, échelle, ...), ce qui conduit à de meilleures performances lors de l'entraînement.

Nous sommes maintenant prêt pour la construction du modèle. 

## Modèle 

Nous construisons un modèle de réseau de neurones convolutifs (CNN) en nous basant sur l'architecture VGG16, que nous avons adaptée à notre cas spécifique de reconnaissance faciale. Pour cela, nous avons retiré la partie fully connected afin de permettre la prédiction de deux éléments : la classe du visage et les coordonnées de la bounding box.

Ainsi, nous pouvons considérer qu’il y a "deux modèles" à construire au sein du même réseau : l’un chargé de la classification et l’autre de la régression des coordonnées de la bounding box.

**Remarque :** Pour la construction du modèle, nous avons besoin des bibliothèque **Numpy** et **TensorFlow**.

![Capture d’écran 2025-03-06 154249](https://github.com/user-attachments/assets/d12b54ac-68bb-4b49-b4d7-78a9309ad327)

Le modèle contient 16 826 181 paramètres. 

## Fonctions coûts 

Comme nous l'avons dit précédemment, il y a "deux modèles" à construire. Pour le modèle de classification (visage ou non), la fonction coût utilisée est un Binary Cross Entropy classique. 

En revanche, pour le modèle de régression, nous avons dû créer notre propre fonction coût. Elle s'inspire de la MSE. Elle prend en compte les erreurs des coordonnées de la bouding box : x_min, y_min, x_max et y_max, mais aussi les erreurs de largeur et longueur de la bouding box. 

**Pourquoi séparer les erreurs de coordonnées et les erreurs de dimensions pour le modèle de régression ?**

Même s'ils sont liés, ces deux erreurs n'ont pas exactement le même impact sur la localisation de l'objet. Une bounding box peut avoir un bon coin supérieur gauche mais de mauvaises dimensions, ou inversement.

Les séparer permet de pondérer différemment ces erreurs et d'assurer que le modèle apprend correctement à prédire à la fois l'emplacement et la taille des bounding boxes. Cela évite qu’une erreur dans la position des coins masque une erreur dans la taille, ou inversement.

## Entraînement du modèle 

Nous avons entraîné le modèle avec le train set mais aussi le validation set. Le val set va nous permettre d'ajuster les hyperparamètres (comme le taux d’apprentissage ou la complexité du réseau) et à détecter un éventuel sur-ajustement (overfitting). 

![Capture d’écran 2025-03-06 173246](https://github.com/user-attachments/assets/8ca2722c-c283-4dd6-95b4-cae7f3d6bb0e)

**Classification loss**

- La courbe bleue (train loss) est relativement stable et basse, ce qui indique que le modèle apprend bien sur l'ensemble d'entraînement.
- La validation loss (orange) démarre haute, diminue rapidement et devient presque nulle après quelques itérations. Cela peut indiquer que le modèle devient performant pour la classification, mais il pourrait aussi y avoir un sur-ajustement (overfitting) si la validation loss atteint zéro trop rapidement.

**Regréssion loss**

- La courbe bleue (train loss) est relativement stable et basse, ce qui indique que le modèle apprend bien sur l'ensemble d'entraînement.
- La courbe orange (val loss) commence élevée avec de fortes fluctuations avant de diminuer progressivement. Cela peut suggérer une variance élevée au début, mais elle semble se stabiliser.

**Total loss**

On sait que total loss = regress loss + 0.5*class loss. Ainsi, tout comme la régression loss :

- La courbe d'entraînement (bleue) reste relativement stable et basse.
- La validation loss (orange) a un comportement similaire à la regress loss : variations au début, puis stabilisation à un niveau faible.

**Conclusion**

Le modèle semble bien apprendre sur l'entraînement, car les pertes lors de l'entraînement restent faibles. Les pertes lors de la validation sont plus instables au début mais finissent par converger.

## Prédiction 

Le modèle a ensuite été évalué avec le test set pour vérifier si le modèle est fiable et précis : 

![Capture d’écran 2025-03-06 174235](https://github.com/user-attachments/assets/ccb0bdec-396f-48ac-b575-457b8a118357)

Les résultats montrent que le modèle fonctionne correctement : en l'absence de visage dans l’image, aucune bounding box n’est générée, ce qui indique une bonne capacité de filtrage des images non pertinentes. Lorsqu’un visage est présent, le modèle parvient à le détecter avec précision et à le localiser correctement à l’aide d’une bounding box bien ajustée. Ces observations confirment la fiabilité et la précision du modèle.

## Détection en temps réel

https://github.com/user-attachments/assets/ccc2853c-06cb-4f6f-b922-ef6bd8156e9f

## Conclusion 

Les performances obtenues avec le modèle sont globalement bonnes, avec des résultats convaincants sur l'ensemble d'entraînement, de validation et de test. Toutefois, certaines limitations apparaissent, notamment des bugs lorsque l’utilisateur s’éloigne de la caméra. Cela pourrait être dû à un manque de données représentatives pour ces cas spécifiques ou à des contraintes liées à la robustesse du modèle en conditions réelles.

Une amélioration des performances aurait pu être faite en poursuivant l'entraînement sur un plus grand nombre d'itérations. Cependant, cela impliquerait un coût en calcul plus élevé et une augmentation significative du temps nécessaire à l'entraînement. Un compromis entre précision et efficacité a donc été trouvé dans ce projet.

Des optimisations futures, comme l'utilisation de techniques d'augmentation de données ou un ajustement des hyperparamètres, pourraient permettre de corriger les instabilités observées tout en limitant le surcoût en calcul.




