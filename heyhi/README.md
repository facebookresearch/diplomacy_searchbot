# HeyHi

HeyHi is a sceleton config library to facilate rapid research. It provides the following:

 * Support for parsing and composing configs
 * Launch and track (e.g., restart) jobs on slurm
 * API for launch and get results of sweeps and group runs
 * GSheet export

## Intro

We use config instead of argparse to run tasks.
A config is a text proto message of type `MetaCfg` as defined in [conf/conf.proto](../conf/conf.proto).
The proto files are mostly self-explanatory, if you ignore `optional` and  `= <digit>` parts.
See [protobuf docs](https://developers.google.com/protocol-buffers/docs/proto#simple) for details.
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


## FAQ

### Nothing works
If you get error about failed import `conf.conf_pb2` or about missing fields in the config, run:
```
protoc conf/*.proto --python_out ./
```

### How do I do `--help`
Go to `conf/conf.proto` and you'll see all flags with docs.

### Redefine output folder
Use `--exp_id_pattern_override=<outdir>`.

### Adding new flags
Just add them into the proto. The field id (number after "=") doesn't matter as we don't use binary format. Just increment the last one. Don't forget to run `protoc` after this.

### Running group runs


Sometimes you wants to do several runs, e.g., several training runs with different hyper parameters. One way to do this, is to run a bash script that will call `run.py` several time:
```(bash)
for lr in 0.1 0.001; do
  python run.py --cfg conf/c02_sup_train/sl.prototxt I.launcher=slurm_8gpus \
    lr=$lr
done
```

Note, there is no `--adhoc` flag. This means that the output folder will be a function of the config name and redefines. As a result, if such experiment is already running, run.py will not start a new one. So it's safe to run the script several times. Or add new values for `lr` in the script.


If you want to control the output path manully, you can do so:
```(bash)
for lr in 0.1 0.001; do
  python run.py --cfg conf/c02_sup_train/sl.prototxt I.launcher=slurm_8gpus \
    lr=$lr \
    --exp_id_pattern_override /checkpoint/$USER/custom_path/lr_${lr}
done
```

Alternative to bash is python, i.e. to make a call to run.py from `heyhi` module. The benefits of this approach is that one may check the status of the running (or died) jobs or get a mapping from experiment path to hyper parameter dict without having to reverse engineer the output folder.
Example of such approach could be found in [conf/c02_sup_train/rs_yolo_01_example.py](conf/c02_sup_train/rs_yolo_01_example.py). The script does launching and results aggregation with optional export to google-sheets.

Code walk through:

  * `yield_sweep` function yields pairs `(cfg_path, dict_with_redefines)` that has to be evaluated.
  * `get_logs_data` takes an experiment handle (a thing such that `handle.exp_path` is where outputs are) and returns dictionary of key metrics
  * `main`: launches runs from `yield_sweep` if there are not running already, goes through handles to collect slurm statuses (DEAD/RUNNING/DONE) of the jobs and current metrics (whatever `get_logs_data` extracts), prints the resulting DataFrame or import it google sheet.

To have google sheet support one has to install `pygsheets`. On the first run, the package will ask to open a link to authentify the heyhi app. Otherwise, you can dump the DataFrame on disk to explore it in a notebook or something else.

Scripts like this are expected to be treated as bash scripts, i.e., in read only mode (do not modify scripts after the experiment is done) and without too much includes. The scripts are expected to be submitted for documentation purposes.

Caveat: the scripts are expected to be run as modules in order for imports to work, e.g., `python -m conf.c02_sup_train.rs_yolo_01_example`.
If you have google-sheet export installed, you can add watch to export data there every 10m:
```
watch -n 600 "python -m conf.c02_sup_train.rs_yolo_01_example  2>&1 | tail -n3;   "
```

