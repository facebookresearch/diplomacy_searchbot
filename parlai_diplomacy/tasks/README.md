# Dialogue and State Teachers

* `-t` or `--task` indicates the task teacher. The name shows what's included in the input and output sequences, e.g. `message_state_order` means the `input_sequence = (message + state)`, and the `output_sequence = order`; `order_history_message_order` means the `input_sequence = (order_history + message)`, and the `output_sequence = order`

* `--include-message-from` indicates the message source, if `speaker_msg_only`, only messages from that speaker are included; if `partner_msg_only`, only messages from that speaker's partners are included; if `all_msg`, messages from both sides are included.


## Get tokenized sequence length statistics
To view the valid data sequence stats for the StateMessageOrderTeacher task, where input sequence = (state + all messages).
```
python scripts/display_data.py -t state_message_order -dt valid -v --include-message-from all_msg --with-seq-len-stat --train-valid-split-by data_point_with_msg --which-tokenizer-for-seq-stat bart --debug
```

## Build and save a new joined_json dump
```
python scripts/display_data.py -t state_message_order -dt valid -v --include-message-from all_msg
```

## Train different models
```
python scripts/sweeps/*.py
```
