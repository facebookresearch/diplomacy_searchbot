# Diplomacy Dialogue Data

Contains variants of dialogue models.
To view the data with streaming, for example:
```
python scripts/display_data.py -t dialogue_chunk -dt train:stream
```

Other teacher variants: 
 - dialogue_chunk: message -> message
 - message_history_state_dialogue_chunk: Message History, State -> message
 - state_message_history_dialogue_chunk: State, Message History -> Message
 - message_history_order_history_dialogue_chunk: Message History, Order History -> Message
 - message_history_order_history_state_dialogue_chunk: Message History, Order History, State -> Message