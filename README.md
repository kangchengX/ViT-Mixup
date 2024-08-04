<p align="center">
    <h1 align="center">VIT-MIXUP</h1>
</p>
<p align="center">
    <em>Superior Image Classification.</em>
</p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=default&logo=TensorFlow&logoColor=white" alt="TensorFlow">
	<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=default&logo=Jupyter&logoColor=white" alt="Jupyter">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
</p>

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data](#data)
- [Modules](#modules)
- [Getting Started](#getting-started)
   - [Installation](#installation)
   - [Usage](#usage)
      - [Command Line Arguments](#command-line-arguments)
      - [Example](#example)
</details>
<hr>

##  Overview

ViT-Mixup is an image classification project that enhances model generalization through advanced data augmentation techniques, specifically integrating MixUp with Vision Transformers (ViT). The project encompasses complete data processing, model training, and evaluation workflows. Core functionalities include dataset splitting, image normalization, and visualization tools that facilitate comprehensive result analysis.

---

##  Repository Structure

```sh
└── ViT-Mixup/
    ├── data_utilis.py
    ├── display_tools.py
    ├── main.ipynb
    ├── main.py
    ├── models.py
    ├── process.py
    ├── README.md
    └── requirements.txt
```

---

## Data

The exapmle data used for this project is a public dataset named [CIFAR-10](https://www.cs.toronto.edu/%7Ekriz/cifar.html).

---

##  Modules

<details open><summary>.</summary>

| File                                 | Summary |
| ---                                  | --- |
| [data_utilis.py](data_utilis.py)     | Splits the dataset into training, validation, and test sets, ensuring proper distribution of images and labels for each subset. Normalizes image data and provides flexibility in setting the ratios for development and training sets, facilitating effective model evaluation and training within the repositorys data processing framework.|
| [display_tools.py](display_tools.py) | Display tools enhance visualization capabilities, allowing for the creation of montages of mixed-up images, visual comparisons of true and predicted labels from trained models, and detailed performance summaries for training, validation, and test datasets. |
| [main.ipynb](main.ipynb)             | This shows some usage of the projectm and workflow of model training, validation, test, as well as new model setting. |
| [main.py](main.py)                   | Demonstrate the entire machine learning workflow for the project, encompassing data loading, model initialization, training, and evaluation. Various args can be passed using command line, integrating configurable parameters to hyperperameter tuning process.|
| [models.py](models.py)               | Implement Vision Transformer (ViT) from scrach and its augmented version, ViT with MixUp. The core functionality includes building MLP blocks, Transformer blocks, and the complete ViT model, facilitating image classification tasks. The augmentation method MixUp enhances training by mixing image samples for improved generalization.|
| [process.py](process.py)             | Implements a class to train, validate and test the VitMixup model. |
| [requirements.txt](requirements.txt) | Outline dependencies essential for the project. |

</details>

---

##  Getting Started

**System Requirements:**

* **Python**: `version 3.10.8`

### Installation

<h4>From <code>source</code></h4>

> 1. Clone the repository:
>
> ```console
> $ git clone https://github.com/kangchengX/ViT-Mixup.git
> ```
>
> 2. Change to the project directory:
> ```console
> $ cd ViT-Mixup
> ```
>
> 3. Install the dependencies:
> ```console
> $ pip install -r requirements.txt
> ```

###  Usage

<h4>From <code>source</code></h4>

> ```console
> $ python main.py [OPTIONS]
> ```

#### Command Line Arguments

| Option | Type | Description | Default Value |
|--------|------|-------------|---------------|
| --ratio_dev | Float | Ratio for development set (i.e. training and validation) in the whole data set. | `0.8` |
| --ratio_train | Float | Ratio for train set in the whole development set. | `0.9` |
| --sampling_method | String | Method to generate lambda. Choices are `beta` or `uniform`. | `uniform` |
| --image_size | Integer | Width or height of input images. | `32` |
| --patch_size | Integer | Width or height of patches. | `4` |
| --num_classes | Integer | Number of the classes. | `10` |
| --dim | Integer | Dimension of the word vectors. | `256` |
| --depth | Integer | Number of transformer blocks. | `8` |
| --num_heads | Integer | Number of heads in the transformer. | `8` |
| --mlp_dim | Integer | Hidden dimension of MLP blocks. | `512` |
| --dropout | Float | Dropout percentage. | `0.5` |
| --alpha | Float | Parameter for beta distribution (used if sampling_method is `beta`). | `None` |
| --uniform_range | Tuple | Predefined range to generate lambda uniformly (used if sampling_method is 'uniform'). | `(0.0, 1.0)` |
| --learning_rate | Float | Learning rate during training. | `0.001` |
| --batch_size | Integer | Batch size during training. | `64` |
| --num_epochs | Integer | Number of epochs during training. | `40` |
| --monitor_on_validation | Boolean | Indicates if assess model on the validation set during training. | `True` |
| --path_root | String | Path root to save models and log if not `None`. | string of the current time |
| --save_model | Boolean | Indicates if save the final model. `path_root` should not be `None` if this is `True`. | `False` |
| --save_period | Integer | Save the model every save_period of epochs if not `None`. `path_root` should not be `None` if this is not `None`. | `None` |
| --save_log | Boolean | Indicates if log will be saved. `path_root` should not be None if this is not `True`. | `True` |

#### Example

```console
$ python main.py --monitor_on_validation --save_period 5 --save_log
```