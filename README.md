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
