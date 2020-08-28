# Streaming Dialogue/Order Data

Data for streaming dialogue and order information.

To view for example, the teacher that takes message and state information as input, and the order for a given power as ouput, run:
```
python scripts/display_data.py -t state_order_chunk -dt train:stream
```

Other teacher variants include:
- state_order_chunk: state --> order
- order_history_order_chunk: order history --> order
- speaker_token_order_chunk: power --> order
- dummy_token_order_chunk: UNK --> order
