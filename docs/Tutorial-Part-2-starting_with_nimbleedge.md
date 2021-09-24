
- [The Simulation](#the-simulation)
- [Model Definition](#model-definition)
- [Hooking the registry](#hooking-the-registry)
- [Standard Training](#standard-training)

# The Simulation

Before we put a code into production we need to evaluate the models and run benchmarks to get the expected accuracy gains.

There is a [simulator](https://github.com/NimbleEdge/RecoEdge) created by NimbleEdge exactly for this purpose. 

- The FL simulator is designed in a way to make the architecture as close to real world deployments as possible.
- You can simulate both the normal ML training and FL training with the simulator.
- The design is scalable to hit 10000+ workers running in the simulation.

Let's take an example of FB AI's [DLRM](https://arxiv.org/abs/1906.00091). This is one of the standard baselines for recommendation engines. We will be training this model on [Kaggle Criteo Ads Data](https://www.kaggle.com/c/criteo-display-ad-challenge). 


# Model Definition
All the model descriptions go into [fedrec/modules](https://github.com/NimbleEdge/RecoEdge/tree/main/fedrec/modules). You can add your own folder of models as well and hook the registry with it.

We will create a file dlrm.py and write its implementation in standard pytorch code.

```python
from torch import nn


class DLRM_Net(nn.Module):
    
    def __init__(self, arg1, arg2, arg3):
        # Your model description comes here.
    
    def forward(inputs):
        # process inputs
        return output 
```

To see the real implementation of DLRM, please check out the [dlrm implementation in the repository](../fedrec/modules/dlrm.py)

# Hooking the registry
The simulator makes it easy to experiment with different model architectures, hyper parameters, optimizers and other components of an ML pipeline.

We define a [registry class](../fedrec/utilities/registry.py) which records all the model definitions, optimizers and attaches a configuration file to the top.

For all your experiments simply define the config file and you are done.

In our DLRM model description, we record it in the registry by annotating it with `@registry.load(<Class Type>, <Name>)`

```python
from torch import nn
from fedrec.utilities import registry

@registry.load('model','dlrm')
class DLRM_Net(nn.Module):
    
    def __init__(self, arg1, arg2, arg3):
        # Your model description comes here.
    
    def forward(inputs):
        # process inputs
        return output 

```

Now create a [config.yml](../configs/dlrm.yml) file to pass the arguments and hyper parameters. 

```yaml
model: # The <Class Type> annotated in registry
    name : 'dlrm' # The unique identifier key 
```

# Standard Training

Training your model in the normal non-FL settting requires you to write the implementations for `train` and `test` methods. You can also implement `validate` method if you want and all these methods will automatically be serialized into FL plans when we move into FL deployment.

The [BaseTrainer](../fedrec/trainers/base_trainer.py) abstracts away the basic methods needed to implemented. 

Simply subclass the `BaseTrainer` and create your own trainer object. We will call this DLRMTrainer

```python
@registry.load('trainer', 'dlrm')
class DLRMTrainer(BaseTrainer):

    def __init__(
            self,
            config_dict: Dict,
            train_config: DLRMTrainConfig,
            logger: BaseLogger, 
            model_preproc: PreProcessor,) -> None:

        self.train_config = train_config
        super().__init__(config_dict, train_config, logger, model_preproc)

```

Next implement the data loaders. These are standard PyTorch dataloaders and return them in the Trainer class.

```python
@property
def dataloaders(self):
    return {
            'train': train_data_loader,
            'train_eval': train_eval_data_loader,
            'val': val_data_loader
        }

```

Define the train and test methods of `BaseTrainer` in `DLRMTrainer`.

With this you are ready to train your model. Till now we have been doing what you usually do to train your ML models. We have been writing standard PyTorch code and developing our ML pipeline.

In the [next section](./simulating_fl_cycle.md) we will see how easy it is to convert the normal ML pipeline into an FL pipeline.

