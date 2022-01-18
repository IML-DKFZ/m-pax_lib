<br />
<p align="center">
  <a href="https://github.com/lukaskln/Interpretability-of-Disentangled-Representations-by-Explanatory-Methods">
    <img src="https://polybox.ethz.ch/index.php/s/rgCwPsnIfT2p928/download" alt="Logo" width="650"> 
  </a>

  <h3 align="center">Improving Explainability of Disentangled Representations using Multipath-Attribution Mappings</h3>

  <p align="center">
    <a href="https://openreview.net/forum?id=3uQ2Z0MhnoE"><strong>Read the paper »</strong></a>
    <br />

  </p>
</p>

## 🔎&nbsp;&nbsp;Table of Contents
* [Introduction](#introduction)
* [Project Structure](#project-structure)
* [Usage](#usage)
  * [Run the code](#run-the-code)
  * [Reproducing the results](#reproducing-the-results)
* [How to cite this code](#how-to-cite-this-code)
* [Acknowledgements](#acknowledgements)

## 📌&nbsp;&nbsp;Introduction

<p align="center">
  <img src="https://polybox.ethz.ch/index.php/s/qzYLmEENEwqT66W/download" width="650"> 
</p>

Explainable AI aims to render model behavior understandable by humans, which can be seen as an intermediate step in extracting causal relations from correlative patterns. Due to the high risk of possible fatal decisions in image-based clinical diagnostics, it is necessary to integrate explainable AI into these safety-critical systems. Current explanatory methods typically assign attribution scores to pixel regions in the input image, indicating their importance for a model's decision. However, they fall short when explaining why a visual feature is used. We propose a framework that utilizes interpretable disentangled representations for downstream-task prediction. Through visualizing the disentangled representations, we enable experts to investigate possible causation effects by leveraging their domain knowledge. Additionally, we deploy a multi-path attribution mapping for enriching and validating explanations. We demonstrate the effectiveness of our approach on a synthetic benchmark suite and two medical datasets. We show that the framework not only acts as a catalyst for causal relation extraction but also enhances model robustness by enabling shortcut detection without the need for testing under distribution shifts.

## 🗂&nbsp;&nbsp;Project Structure
```
├── README.md                                
├── LICENSE.txt                             
├── requirements.txt            - txt file with the environment
├── run_eval.py                 - Main script to execute for evaluation
├── run_head.py                 - Main script to execute for supervised training
├── run_tcvae.py                - Main script to execute for unsupervised pre-training
├── configs                     - Hydra configs
│   ├── config_eval.yaml
│   ├── config_head.yaml
│   ├── config_tcvae.yaml
│   ├── callbacks
│   ├── datamodule
│   ├── evaluation
│   ├── experiment
│   ├── hydra
│   ├── logger
│   ├── model
│   └── trainer
├── data                        - Data storage folders (each filled after first run)
│   ├── DiagVibSix
│   ├── ISIC
│   ├── MNIST
│   ├── models                  - Trained and saved models
│   │   └── dataset_beta        - Copied checkpoints per dataset and beta value
│   │       └── images          - Image export folder 
│   └── OCT
├── logs                        - Logs and Checkpoints saved per run and date
│   └── runs
│       └── date
│           └── timestamp
│               ├── checkpoints
│               ├── .hydra
│               └── tensorboard
└── src
    ├── evaluate.py             - Evaluation pipeline
    ├── train.py                - Training pipeline
    ├── datamodules             - Datamodules scripts
    ├── evaluation              - Evaluation scripts
    ├── models                  - Lightning modules
    └── utils                   - Various utility scripts (beta-TCVAE loss etc.)
                         

```

## 🚀&nbsp;&nbsp;Usage

All essential libraries for the execution of the code are provided in the requirements.txt file from which a new environment can be created (Linux only). For the R script, please install the corresponding libraries beforehand. 

### Run the code

Once the virtual environment is activated, the code can be run as follows:

### Reproducing the results

<p align="center">
    <img src="https://polybox.ethz.ch/index.php/s/kRVJcPFubIW1JXy/download" width="257"> &nbsp;
    <img src="https://polybox.ethz.ch/index.php/s/Nv6yV7LnwlJdrsa/download" width="250"> &nbsp;
    <img src="https://polybox.ethz.ch/index.php/s/Zm2v8XybCy7awvS/download" width="250"> 
</p>

## 📝&nbsp;&nbsp;How to cite this code

Please cite the original publication:

```
@article{klein2021improving,
  title={Improving Explainability of Disentangled Representations using Multipath-Attribution Mappings},
  author={Klein, Lukas and Carvalho, Jo{\~a}o BS and El-Assady, Mennatallah and Penna, Paolo and Buhmann, Joachim M and Jaeger, Paul F},
  year={2021}
}
```

## Acknowledgements

The code is developed by the authors of the paper. However, it does also contain pieces of code from the following packages:

- Lightning-Hydra-Template by Zalewski, Łukasz et al: https://github.com/ashleve/lightning-hydra-template
- Disentangled VAE by Dubois, Yann et al: https://github.com/YannDubs/disentangling-vae


