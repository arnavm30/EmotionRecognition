# Emotion Recognition Using 3D Face Mesh: ML for Mental Health
> In partnership with the [Stanford Human Perception Lab](https://med.stanford.edu/hpl.html)

This repository is the official implementation of [Emotion Recognition Using 3D Face Mesh: ML for Mental Health](https://medium.com/@arnavm30/emotion-recognition-using-3d-face-mesh-ml-for-mental-health-744f822f8e41). 

> The attached notebook first creates a dataset of a 3D face mesh using [Mediapipe Face Mesh](https://arxiv.org/abs/2006.10962) and restructures the images of the dataset into folders with the corresponding label. In the next section, Logistic Regression, Ridge Classifier, Random Forests, and XGBoost models are used to obtain a baseline. In the following section, a basic MLP is trained on the face meshes. In the section after, an MLP with a Resnet50 in the first stage to extract features from each image for context and concatenate them to the face meshes is trained.
The MLPs were hyperparameter-tuned initially using Grid Search and then Bayesian Optimization.

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
