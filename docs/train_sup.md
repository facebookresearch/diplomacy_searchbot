# Training a supervised model
The model code is in [fairdiplomacy/models/dipnet/](../fairdiplomacy/models/dipnet/). The model architecture is defined in [dipnet.py](../fairdiplomacy/models/dipnet/dipnet.py)
We use config instead of argparse to run training and evaluation jobs. A config is a text proto message of type `MetaCfg` as defined in [conf/conf.proto](../conf/conf.proto).
Specific configs are stored in [conf/](../conf/) folder with [conf/common](../conf/common) containing config chunks that could be included into other configs.

To run a config simply run:
```
python run.py --adhoc --cfg conf/c02_sup_train/sl.prototxt
```

This will start training of a supervised agent locally using config from [conf/c02_sup_train/sl.prototxt](../conf/c02_sup_train/sl.prototxt).
To redefine any parameters in the config simply pass them via command line arguments using the following syntax: `<key>=<value>`. Example:
```
python run.py --adhoc --cfg conf/c02_sup_train/sl.prototxt batch_size=10
```

To run on cluster one has to set job params in `launcher.slurm`. The simplest way to do this is to use a predefined message. run.py allows to include any partial config into the main config using the following syntax: `I=<config_path>` or `I.<mount_point>=<config_path>`. In the latter case, the subconfig will be merged into the specified place of the root config. For instace, the following will include a launcher to run on cluster on a single node with 8 gpus (as defined in [conf/common/launcher/slurm_8gpus.prototxt](../conf/common/launcher/slurm_8gpus.prototxt)):

```
python run.py --adhoc --cfg conf/c02_sup_train/sl.prototxt I.launcher=slurm_8gpus
```

One can combine includes and scalar redefines, e.g., to run the job on 2 machines:
```
python run.py --adhoc --cfg conf/c02_sup_train/sl.prototxt I.launcher=slurm_8gpus launcher.slurm.num_gpus=16
```

To train a model on the cluster, see the scripts in [slurm/](../slurm/), specifically [example_train_sl.sh](../slurm/example_train_sl.sh).
