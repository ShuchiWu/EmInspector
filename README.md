# EmInspector
This repository contains the code of the paper ["EmInspector: Combating Backdoor Attacks in Federated Self-Supervised Learning Through Embedding Inspection"](https://arxiv.org/abs/2405.13080), a inelegant but runnable version written by a newbie :see_no_evil:. EmInspector, a defense mechanism devised for combatting backdoor attack in federated self-supervised learning through inspecting the representation/embedding space of uploaded local models and excluding the suspicious ones. Datasets in `.npz` format and part of models we trained can be accessed at [datasets & models](https://drive.google.com/drive/folders/1iNftJa5essO25ajgXT9dSHgLuLePmjJA?usp=drive_link).

<img src="https://github.com/ShuchiWu/EmInspector/blob/main/framework.jpg">

## Pre-train an encoder
Following [BadEncoder](https://arxiv.org/pdf/2108.00352), we pre-train image encoder use the widely-adopted [contrastive learning](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf), and train our federated system based on their implementaion.

## Backdoor the pre-trained model
Again, since BadEncoder is one of the most known and effective backdoor attacks targeting self-supervised models, we employ it to evaluate the robustness of the our federated self-supervised learning system.
