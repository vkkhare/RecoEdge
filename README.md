# FedDLRM
A reimplementation of Facebook AI's [DLRM](https://arxiv.org/abs/1906.00091) for Federated Learning

The repository refactors the [original code](https://github.com/facebookresearch/dlrm) to allow easy exploration with new datasets, model architectures and hyper-parameters.

# Code Structure
```bash
├── fedrec
│   ├── datasets
│   │   ├── criteo.py (Kaggle Dataset)
│   │   └── criteo_processor.py (Pre-processing)
│   │
│   ├── modules
│   │   ├── dlrm.py (DLRM pytorch module) 
│   │   ├── embeddings.py
│   │   └── sigmoid.py (activation units)
│   │
│   ├── optimization
│   │   ├── optimizer.py (Adagrad)
│   │   └── corrected_sgd.py (Pytorch sparse SGD bug)
│   │   └── schedulers.py
│   │
│   └── utilities
│       └── **
│       
├── scripts
├── configs (Your custom YAML configs)
├── test.py
├── train.py (Run standard DLRM)
├── README.md
├── conda_requirements.txt
└── .gitignore
```

## Running Standard DLRM Training

**Download and extract** the Kaggle Criteo dataset from [Google Drive](https://drive.google.com/file/d/17K5ntN30LbMWJ2gHHSkwCGHEcAjShm2_/view?usp=sharing)

```bash
mkdir criteo; cd criteo
gdown https://drive.google.com/file/d/17K5ntN30LbMWJ2gHHSkwCGHEcAjShm2_
tar -xf dac.tar.gz
```

Clone this repo and change the argument `datafile` in [configs/dlrm.yml](configs/dlrm.yml) to the above path.
```bash
git clone https://github.com/NimbleEdge/fedDLRM
```
```yml
model :
  name : 'dlrm'
  ...
  preproc :
    datafile : "<Path to Criteo>/criteo/train.txt"
 
```
Install the dependencies with conda or pip
```bash
conda create --name recsys python=3.8 --file conda_requirements.txt
conda activate recsys
``` 

Run data preprocessing with [preprocess_data](preprocess_data.py) and supply the config file. You should be able to generate per-day split from the entire dataset as well a processed data file
```bash
python preprocess_data.py --config configs/dlrm.yml --logdir $HOME/logs/kaggle_criteo/exp_1
```

**Begin Training**
```bash
python train.py --config configs/dlrm.yml --logdir $HOME/logs/kaggle_criteo/exp_3 --num_eval_batches 1000 --devices 0
```

Run tensorboard to view training loss and validation metrics at [localhost:8888](http://localhost:8888/)
```bash
tensorboard --logdir $HOME/logs/kaggle_criteo --port 8888
```

# Federated Training
coming soon


# Customization
## Training Configuration
There are two ways to adjust training hyper-parameters:
- **Set values in config/*.yml** persistent settings which are necessary for reproducibility eg randomization seed
- **Pass them as CLI argument** Good for non-persistent and dynamic settings like gpu device  

*In case of conflict, CLI argument supercedes config file parameter.*
For further reference, check out [training config flags](configs/flags.md)

## Model Architecture
### Adjusting DLRM model params 
Any parameter needed to instantiate the pytorch module can be supplied by simply creating a key-value pair in the config file.

For example DLRM requires `arch_feature_emb_size`, `arch_mlp_bot`, etc 
```yml
model: 
  name : 'dlrm'
  arch_sparse_feature_size : 16
  arch_mlp_bot : [13, 512, 256, 64]
  arch_mlp_top : [367, 256, 1]
  arch_interaction_op : "dot"
  arch_interaction_itself : False
  sigmoid_bot : "relu"
  sigmoid_top : "sigmoid"
  loss_function: "mse"
```

### Adding new models
Model architecture can only be changed via `configs/*.yml` files. Every model declaration is tagged with an appropriate name and loaded into registry.
```python
@registry.load('model','<model_name>')
class My_Model(torch.nn.Module):
    def __init__(num):
        ... 
```

You can define your own modules and add them in the [fedrec/modules](fedrec/modules). Finally set the `name` flag of `model` tag in config file
```yml
model : 
  name : "<model name>"
```