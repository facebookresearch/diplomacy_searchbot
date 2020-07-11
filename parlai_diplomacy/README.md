# ParlAI Diplomacy

Folder for all of the language related diplomacy code.

I import parlai here so there are a few caveats/changes/tricks to using ParlAI inside this repo, please read below. All commands should be run from inside the `parlai_diplomacy` directory.

**NOTE: the dialogue chunk teacher (for faster streaming) does not currently work as the ParlAI pip package needs to be updated to include my latest changes. It should be updated sometime next week (Week of July 13).**

## Creating tasks and agents
We use the "register" syntax so that new agents and tasks are visible to the parlai module. Note, however,
that in any script that uses these tasks or agents, you will have to make a call to the functions `register_all_agents()` and `register_all_tasks()` which can be found in `utils/loading.py`.


## Viewing data
NOTE: I used an arbitrary train/valid split for now so we probably want to talk about how to split

We cannot use the `parlai` syntax any longer, as we need to make a call to register all tasks and agents found in this repository. For this purpose, I put scripts inside the `scripts` folder for all basic ParlAI functions.

### Dialogue teacher

To view the train data using the dialogue teacher that loads all of the data into memory at once, run:
```
python scripts/display_data.py -t dialogue
```
To do the same thing, but view with metadata (verbose), add the flag `-v`:
```
python scripts/display_data.py -t dialogue -v
```
To view only the data with a minimum of 3 dialogue turns, try:
```
python scripts/display_data.py -t dialogue -v --min-turns 3
```

The regular dialogue teacher loads four large JSON files consecutively (directly exported from the SQL table), which can be found in this directory:
```
/checkpoint/fairdiplomacy/processed_chat_jsons/game_phases/
```

### Message order teacher
To view the train data for the MessageOrder task:
```
python scripts/display_data.py -t message_order
```

### Dialogue chunk teacher
**(CURRENTLY NOT WORKING) TODO: update this once the parlai pip package is update.**
To view the train data using the CHUNK teacher (streams in small files to load the data quickly) run:
```
python scripts/display_data.py -t dialogue_chunk -dt train:stream
```

The chunk teacher loads small JSON files organized by game from this directory:
```
/checkpoint/fairdiplomacy/chat_messages_jsons/
```

## Training models

### Launching a sweep

I added an example script that launches a sweep on the FAIR cluster to finetune a 400M parameter reddit model on the diplomacy data, which can be found in: `scripts/sweeps/400M_chatdata_basic_baseline.py`. Note that this does not include optimal hyperparameters and is merely for a quick demonstration.

Other sweeps can be found in this folder.

### Training from the commandline

Example command for training a model on your devfair that is initialized with weights from a pre-trained Reddit model:
```
python -u scripts/train.py -t dialogue --min-turns 3 -veps 0.1 --attention-dropout 0.0 --dropout 0.1 --fp16 True --init-model /checkpoint/edinan/diplomacy/400M_0701/model --dict-file /checkpoint/edinan/diplomacy/400M_0701/model.dict -m transformer/generator --embedding-size 1024 --ffn-size 4096 --attention-dropout 0.1 --n-heads 16 --n-positions 2048 --variant prelayernorm --activation gelu --n-encoder-layers 2 --n-decoder-layers 22 --skip-generation True --fp16 True --fp16-impl mem_efficient --force-fp16-tokens True --optimizer mem_eff_adam --truncate 128 --dict-tokenizer bytelevelbpe --bpe-vocab /checkpoint/edinan/20200625/reddit-400M-baseline/de8/model.dict-vocab.json --bpe-merge /checkpoint/edinan/20200625/reddit-400M-baseline/de8/model.dict-merges.txt --label-truncate 128 --log_every_n_secs 10 -lr 7e-06 --lr-scheduler reduceonplateau --lr-scheduler-patience 3 --optimizer adam --relu-dropout 0.0 --activation gelu --model-parallel True --save-after-valid True --text-truncate 128 --truncate 128 --warmup_updates 100 --fp16-impl mem_efficient --update-freq 2 --gradient-clip 0.1 --skip-generation True -vp 10 --max-train-time 27647.999999999996 -vmt ppl -vmm min -stim 360 -vme 10000 -bs 4 -mf /tmp/fairdip_traintest
```

## Talking to models that are already trained (interactive)

This is a model that was not trained for very long (DO NOT USE FOR ANYTHING ELSE); I'm simply providing a command to talk to it here as a demo:

**400M**
```
parlai interactive -mf /checkpoint/edinan/20200629/diplomacy_basic_baseline/3db/model --skip-generation False --inference beam --beam-size 10 --beam-min-length 15 --beam-block-ngram 3 --beam-context-block-ngram 3
```

Note the `--skip-generation False` above: this is important as by default we skip generation during training for speed.

## Displaying model predictions

This is a model that was not trained for very long (DO NOT USE FOR ANYTHING ELSE); I'm simply providing a command to talk to it here as a demo:

**400M**
This shows an example of the models predictions on 15 examples from the train set of the dialogue task:
```
python scripts/display_model.py -t dialogue -mf /checkpoint/edinan/20200629/diplomacy_basic_baseline/3db/model --skip-generation False --inference beam --beam-size 10 --beam-min-length 15 --beam-block-ngram 3 --beam-context-block-ngram 3 -ne 15 -dt train:evalmode
```

Note the `--skip-generation False` above: this is important as by default we skip generation during training for speed. Also note `-dt train:evalmode`: this is necessary to indicate to the model that we do not want it to train on these examples.


## Evaluating models

Evaluate a model on the validation set of the dialogue task, using beam search with a beam size of 10 and 3-gram blocking:
```
python scripts/eval.py -t dialogue -mf /checkpoint/edinan/20200629/diplomacy_basic_baseline/3db/model --skip-generation False --inference beam --beam-size 10 --beam-min-length 15 --beam-block-ngram 3 --beam-context-block-ngram 3 -ne 15 -dt valid
```
