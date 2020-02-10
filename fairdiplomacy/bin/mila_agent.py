#!/usr/bin/env python
import diplomacy

from fairdiplomacy.agents.mila_sl_agent import MilaSLAgent

if __name__ == '__main__':
    print(MilaSLAgent(host="localhost").get_orders(diplomacy.Game(), "ITALY"))
