GNN-Enhanced Image Classification for Domain Generalization
This project implements an image classification model that integrates a Graph Neural Network (GNN) architecture with a standard deep learning backbone, MobileNetV3-Small.

The core idea is to leverage the GNN to perform information aggregation among data samples (images) that belong to the same class within a batch. This process allows harder-to-learn samples to utilize information from easier-to-learn samples, ultimately improving the model's overall generalization performance. This approach is particularly effective for Domain Generalization tasks, such as those presented by the PACS Dataset.

Some sample of Dataset PACS
//
<img width="698" height="359" alt="image" src="https://github.com/user-attachments/assets/156cff52-9975-4364-be46-4c67ae42fbea" />
