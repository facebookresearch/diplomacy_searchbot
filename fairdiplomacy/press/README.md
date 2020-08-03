# ParlDipNet - Full-Press DipNet Model

To build press dataset:
```
python run.py --adhoc --cfg conf/c08_build_press_db_cache/build_press_db_cache.prototxt
```

In addition to preprocessing the dipnet inputs, we also tokenize the language data using the HuggingFace BPE Tokenizer.


## Listener Press Variant

This model consists of a ParlAI Transformer Encoder (pre-trained) and DipNet Encoder, Value Decoder and Policy Decoder
Essentially, encodes the language using the Tranformer, encodes the state using DipNet Encoder and combines them.
The combined vector is used in the Value and Policy decoders. 

To run ParlDipNet 

```
python run.py --adhoc --cfg conf/c09_press_sup_train/press_sl.prototxt dipnet_train_params.debug_no_mp=1 dipnet_train_params.dataset_params.num_dataloader_workers=-1
```
