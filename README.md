# Emotion Recognition Using 3D FaceMesh: Graph Neural Networks for Mental Health
> In partnership with the [Stanford Human Perception Lab](https://med.stanford.edu/hpl.html)

This repository is the official implementation of [Emotion Recognition Using 3D FaceMesh: Graph Neural Networks for Mental Health](https://medium.com/@arnavm30/emotion-recognition-using-3d-face-mesh-ml-for-mental-health-744f822f8e41). 

> This is the code for an emotion recognition (7 emotion multiclass classification) problem. 
The attached notebook first creates a dataset of 3D face mesh (468 landmarks) using [Mediapipe Face Mesh](https://arxiv.org/abs/2006.10962) from the images of the dataset. A graph is constructed from the 3D face mesh, using both `FACE_CONTOURS` and `FACE_TESSELATIONS` as edges and the landmarks as nodes.
In the next section, supervised learning techniques like Logistic Regression, Ridge Classifier, Random Forests, and XGBoost models are used to obtain a baseline. In the section after, an MLP with a Resnet50 in the first stage to extract features from each image for context and concatenate them to the face meshes is trained. In the final section, A GNN is created and trained on the graphs.
The models were hyperparameter-tuned initially using Grid Search and then Bayesian Optimization.

## Requirements
> The Real-world Affective Faces Database (RAF-DB) dataset can be downloaded from http://www.whdeng.cn/raf/model1.html#dataset.

## Training

To train the model(s), run the cell with:

```train
python train.py --input-data <path_to_data> 
```

## Evaluation

To evaluate the model(s), run the cell with:

```eval
python eval.py --model-file mymodel.pth
```

## Results

Our model achieves the following performance on RAF-DB:

| Model name  | Top 1 Accuracy |
| ----------- |--------------- |
| MResNet50   |     66%        |
|    GNN      |     57%        |
