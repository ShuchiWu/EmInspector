# EmInspector
This repository contains the code of the paper ["EmInspector: Combating Backdoor Attacks in Federated Self-Supervised Learning Through Embedding Inspection"](https://arxiv.org/abs/2405.13080), an unpolished yet functional version created by a novice :see_no_evil:. EmInspector is a defense mechanism designed to combat backdoor attacks in federated self-supervised learning by inspecting the representation/embedding space of uploaded local models and excluding suspicious ones. Datasets in `.npz` format and some of the models we trained can be accessed at [datasets & models](https://drive.google.com/drive/folders/1iNftJa5essO25ajgXT9dSHgLuLePmjJA?usp=drive_link).

<img src="https://github.com/ShuchiWu/EmInspector/blob/main/framework.jpg">

## Pre-train an encoder
Following [BadEncoder](https://arxiv.org/pdf/2108.00352), we pre-train the image encoder using the widely-adopted [contrastive learning](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf) approach, and build our federated system based on their implementation. Once the training data (primarily CIFAR10, STL10, and CIFAR100 as discussed in our paper) is prepared, you can initiate training with the following script:
```script
python fed_simclr.py
```

## Backdoor the pre-trained model
Since BadEncoder is one of the most renowned and effective backdoor attacks targeting self-supervised models, we employ it to evaluate the robustness of our federated self-supervised learning system. Additionally, considering the distributed nature of the federated system, we also consider another well-known backdoor attack scheme tailored for distributed machine learning systems, named [DBA](https://openreview.net/pdf?id=rkgyS0VFvr). In our paper, the original BadEncoder is termed as a single-pattern attack, while the DBA-incorporated version is termed as a coordinated-pattern attack. To execute the backdoor attack, simply modify the trigger used by each malicious client and run the following script:
```script
python backdoor_fssl.py
```
## Downstream evaluation
For evaluations of the trained image encoders (main task accuracy and backdoor attack success rate), we adopt the linear probing protocol. This approach keeps the parameters of the pre-trained encoder frozen and trains an additional, specific classifier head (an MLP) for each dataset.
```script
python downstream_classifier.py
```
