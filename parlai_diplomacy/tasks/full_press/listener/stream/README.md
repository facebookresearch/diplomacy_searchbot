# Streaming Dialogue/Order Data

Data for streaming dialogue and order information.

To view for example, the teacher that takes message and state information as input, and the order for a given power as ouput, run:
```
python scripts/display_data.py -t state_message_order_chunk -dt train:stream
```

Other teacher variants include:
- message_state_order_chunk: message + state --> order
- order_history_message_order_chunk: order history + message --> order
- message_order_history_order_chunk: message + order history --> order
- message_order_chunk: message --> order