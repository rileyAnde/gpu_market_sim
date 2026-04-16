'''
RILEY ANDERSON, HANNAH SMITH

This file is intended to be changed to reflect the desired configuration of the simulator
'''

CFG = {
    # market / sim variables
    "num_gpus": 20,
    "num_epochs": 50,
    "num_rounds": 10,

    # agent counts
    "num_agents": 100,
    "num_bid_cancellers": 0,
    "num_griefers": 0,
    "num_sybil_controllers": 0,
    "sybil_accounts_per_controller": 3,

    # pricing
    "bid_increment": 1,
    "max_price": 500,
    "min_value": 1,
    "max_value": 500,
    "griefer_bid_boost_min": 15,
    "griefer_bid_boost_max": 60,
    "sybil_bid_floor_fraction": 0.5,

    # probs
    "arrival_rate": 0.7,
    "cancel_prob": 0.8,
    "sybil_bid_prob": 0.7,
    "sybil_cancel_prob": 1.0,
    "griefer_cancel_prob": 1.0,

    # misc
    "seed": None,
    "plot": True,
    "verbose": True,
}