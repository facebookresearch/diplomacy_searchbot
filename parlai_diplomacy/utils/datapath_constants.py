"""
Datapath constants.

NOTE: if you edit this file, please include the date that a datapath was constructed
"""

######################################################
## Metadata
######################################################
GAME_METADATA_PATH = "/checkpoint/alerer/fairdiplomacy/facebook_notext/metadata.json"

######################################################
## Message data
######################################################

# Chat message JSONs split into many chunk files
CHUNK_DIALOGUE_PATH = (
    "/checkpoint/fairdiplomacy/press_diplomacy/chat_messages/chat_messages_jsons/"  # 2020-08-13
)
# Chat messages not split into many files
NONCHUNK_DIALOGUE_PATH = "/checkpoint/fairdiplomacy/press_diplomacy/processed_chat_jsons/game_phases/redacted_messages_runthree_*.json"  # 2020-08-01

######################################################
## "Order" (or game)  data
######################################################

# Order data pulled from build_dataset.py with SQL table /checkpoint/fairdiplomacy/facebook_notext.sqlite3
GAME_JSONS_PATH = "/checkpoint/fairdiplomacy/processed_orders_jsons_new/game_*.json"  # 2020-08-13
# TODO: deprecate old paths
GAME_JSONS_PATH_OLD_0801 = (
    "/checkpoint/fairdiplomacy/processed_orders_jsons/game_*.json"  # 2020-08-01
)

######################################################
## Joined message and order data
######################################################

# Joined order and message data, DIPCC format, 250 chunks
CHUNK_MESSAGE_ORDER_250_PATH = "/checkpoint/fairdiplomacy/press_diplomacy/joined_jsons/include_msg=all_msg-special_tokens=False-format=dipcc/extra_zip/data*.gz"  # 2020-09-01
# Joined order and message data, DIPCC format, 1000 chunks
CHUNK_MESSAGE_ORDER_PATH = "/checkpoint/fairdiplomacy/press_diplomacy/joined_jsons/include_msg=all_msg-special_tokens=False-format=dipcc/all/data*.gz"  # 2020-08-13
# TODO: deprecate/delete old paths
CHUNK_MESSAGE_ORDER_PATH_OLD_0813 = "/checkpoint/fairdiplomacy/press_diplomacy/joined_jsons/include_msg=all_msg-special_tokens=False-format=dipcc/all/data*.gz"  # 2020-08-13
# Joined order and message data, DIP format
CHUNK_MESSAGE_ORDER_PATH_OLD_0812 = "/checkpoint/fairdiplomacy/press_diplomacy/joined_jsons/dumps_State_OrderHistory_MessageHistory-all-msg-SpecialToken-False_order/*.json"  # 2020-08-12
