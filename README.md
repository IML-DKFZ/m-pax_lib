<br />
<p align="center">
  <a href="https://github.com/IML-DKFZ/m-pax_lib">
    <img src="https://polybox.ethz.ch/index.php/s/8y400QVO0cWAdtW/download" alt="Logo" width="650"> 
  </a>

  <h3 align="center">Improving Explainability of Disentangled Representations using Multipath-Attribution Mappings</h3>

  <p align="center">
    <a href="https://openreview.net/forum?id=3uQ2Z0MhnoE"><strong>Read the paper ยป</strong></a>
    <br />

  </p>
</p>

## ๐&nbsp;&nbsp;Table of Contents
* [Introduction](#Introduction)
* [Project Structure](#project-structure)
* [Usage](#usage)
  * [Run the code](#run-the-code)
  * [Reproduce the results](#reproduce-the-results)
* [How to cite this code](#how-to-cite-this-code)
* [Acknowledgements](#acknowledgements)

## <a id="Introduction"></a> ๐&nbsp;&nbsp;Introduction 

<p align="center">
  <img src="https://polybox.ethz.ch/index.php/s/rpuUqMnANAQTmKC/download" width="650"> 
</p>

Explainable AI aims to render model behavior understandable by humans, which can be seen as an intermediate step in extracting causal relations from correlative patterns. Due to the high risk of possible fatal decisions in image-based clinical diagnostics, it is necessary to integrate explainable AI into these safety-critical systems. Current explanatory methods typically assign attribution scores to pixel regions in the input image, indicating their importance for a model's decision. However, they fall short when explaining why a visual feature is used. We propose a framework that utilizes interpretable disentangled representations for downstream-task prediction. Through visualizing the disentangled representations, we enable experts to investigate possible causation effects by leveraging their domain knowledge. Additionally, we deploy a multi-path attribution mapping for enriching and validating explanations. We demonstrate the effectiveness of our approach on a synthetic benchmark suite and two medical datasets. We show that the framework not only acts as a catalyst for causal relation extraction but also enhances model robustness by enabling shortcut detection without the need for testing under distribution shifts.

## <a id="project-structure"></a> ๐&nbsp;&nbsp;Project Structure
```
โโโ README.md                                
โโโ LICENSE                             
โโโ requirements.txt            - txt file with the environment
โโโ run_eval.py                 - Main script to execute for evaluation
โโโ run_head.py                 - Main script to execute for supervised training
โโโ run_tcvae.py                - Main script to execute for unsupervised pre-training
โโโ configs                     - Hydra configs
โ   โโโ config_eval.yaml
โ   โโโ config_head.yaml
โ   โโโ config_tcvae.yaml
โ   โโโ callbacks
โ   โโโ datamodule
โ   โโโ evaluation
โ   โโโ experiment
โ   โโโ hydra
โ   โโโ logger
โ   โโโ model
โ   โโโ trainer
โโโ data                        - Data storage folders (each filled after first run)
โ   โโโ DiagVibSix
โ   โโโ ISIC
โ   โโโ MNIST
โ   โโโ models                  - Trained and saved models
โ   โ   โโโ dataset_beta        - Copied checkpoints per dataset and beta value
โ   โ       โโโ images          - Image export folder 
โ   โโโ OCT
โโโ logs                        - Logs and Checkpoints saved per run and date
โ   โโโ runs
โ       โโโ date
โ           โโโ timestamp
โ               โโโ checkpoints
โ               โโโ .hydra
โ               โโโ tensorboard
โโโ src
    โโโ evaluate.py             - Evaluation pipeline
    โโโ train.py                - Training pipeline
    โโโ datamodules             - Datamodules scripts
    โโโ evaluation              - Evaluation scripts
    โโโ models                  - Lightning modules
    โโโ utils                   - Various utility scripts (beta-TCVAE loss etc.)
                         

```

## <a id="usage"></a> ๐&nbsp;&nbsp;Usage

All essential libraries for the execution of the code are provided in the `requirements.txt` file from which a new environment can be created (Linux only). For the R script, please install the corresponding libraries beforehand. Setup package in a conda environment:

```
git clone https://github.com/IML-DKFZ/m-pax_lib
cd m-pax_lib
conda create -n m-pax_lib python=3.7
source activate m-pax_lib
pip install -r requirements.txt
````
Depending on your GPU, change the torch and torchvision version in the `requirements.txt` file to the respective CUDA supporting version. For CPU only support add `trainer.gpus=0` behind every command.

### Run the code

Once the virtual environment is activated, the code can be run as follows:

Running the scripts without any experiment files will start the training and evaluation on mnist. All parameters are defined in the hydra config files and not overwritten by any experiment files. The following commands will first, train the &#946;-TCVAE loss based model with &#946; = 4, second train the downstream classification head, and at last evaluate the model. The `run_tcvae.py` script also automatically initializes the download and extraction of the dataset at `./data/MNIST`.

```
python run_tcvae.py
python run_head.py
python run_eval.py
```
Before training the head, place one of the encoder checkpoints (best or last epoch) from `./logs/runs/date/timestamps/checkpoints` at `./models/mnist_beta=4` and rename them to `encoder.ckpt`. Folder can be renamed, but then has to be changed in the `config/model/head_model.yaml` and  `config/evaluation/default.yaml` files. Place the head checkpoint in the same folder and rename it to `head.ckpt`. The evaluation script will create automatically an image folder inside, and export all graphics to this location.

### Reproduce the results

For all other experiments in the paper, respective experiment files to overwrite the default parameters were created. The following configurations reproduce the results from the paper for each dataset. You can also add your own experiment yaml files
or change the existing ones. For more information see [here](https://github.com/ashleve/lightning-hydra-template).

The ISIC and OCT evaluation need a rather large RAM size of ~80Gb. Reduce the batch size in the `isic/oct_eval.yaml` file to get less accurate but more RAM sparing results.

#### DiagViB-6

```
python run_tcvae.py +experiment=diagvibsix_tcvae.yaml
python run_head.py +experiment=diagvibsix_head.yaml
python run_eval.py +experiment=diagvibsix_eval.yaml seed=43
```

These commands run the experiment for the ZGO study. For the other two studies change ZGO to FGO_05 or FGO_20 in the three experiment files. 

#### UCSD OCT Retina Scans

```
python run_tcvae.py +experiment=oct_tcvae.yaml
python run_head.py +experiment=oct_head.yaml
python run_eval.py +experiment=oct_eval.yaml seed=48
```

#### ISIC Skin Lesion Images

```
python run_tcvae.py +experiment=isic_tcvae.yaml
python run_head.py +experiment=isic_head.yaml
python run_eval.py +experiment=isic_eval.yaml seed=47
```

GIFs traversing the ten latent space features for five observations of each of the three datasets:

<p align="center">
    <img src="https://polybox.ethz.ch/index.php/s/yAzDc0fqDiyud3i/download" width="154"> &nbsp;
    <img src="https://polybox.ethz.ch/index.php/s/CQPQFhSfBtiuUad/download" width="150"> &nbsp;
    <img src="https://polybox.ethz.ch/index.php/s/53nZuouyBQKjznn/download" width="150"> 
</p>

## <a id="how-to-cite-this-code"></a> ๐&nbsp;&nbsp;How to cite this code

Please cite the original publication:

```
@inproceedings{
  klein2022improving,
  title={Improving Explainability of Disentangled Representations using Multipath-Attribution Mappings},
  author={Lukas Klein and Jo{\~a}o B. S. Carvalho and Mennatallah El-Assady and Paolo Penna and Joachim M. Buhmann and Paul F Jaeger},
  booktitle={Medical Imaging with Deep Learning},
  year={2022},
  url={https://openreview.net/forum?id=3uQ2Z0MhnoE}
}
```

## Acknowledgements

The code is developed by the authors of the paper. However, it does also contain pieces of code from the following packages:

- Lightning-Hydra-Template by Zalewski, ลukasz et al: https://github.com/ashleve/lightning-hydra-template
- Disentangled VAE by Dubois, Yann et al: https://github.com/YannDubs/disentangling-vae

<br>

____

<br>

<p align="left">
  <img src="https://polybox.ethz.ch/index.php/s/I6VJEPrCDW9zbEE/download" width="190"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://polybox.ethz.ch/index.php/s/kqDrOTTIzPFYPU7/download" width="91"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Deutsches_Krebsforschungszentrum_Logo.svg/1200px-Deutsches_Krebsforschungszentrum_Logo.svg.png" width="270"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ3A8RAM4OeLrG3XXi263zSvM_GVKbJQooDzS6VecJSA_ncBW1wbhrtg6vwqH3Odx5sWQ&usqp=CAU" width="200">  
</p>

The m-pax_lib is developed and maintained by the Interactive Machine Learning Group of [Helmholtz Imaging](https://www.helmholtz-imaging.de/) and the [DKFZ](https://www.dkfz.de/de/index.html), as well as the Information Science and Engineering Group at [ETH Zรผrich](https://ise.inf.ethz.ch/).



