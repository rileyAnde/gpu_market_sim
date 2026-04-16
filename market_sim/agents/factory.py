'''
RILEY ANDERSON, HANNAH SMITH

This file handles initialization of all agents
'''
class Agent:
    def __init__(self, agent_id, kind="honest", group_id=None):
        self.agent_id = agent_id
        self.kind = kind
        self.group_id = group_id
        self.waiting = False
        self.wait_epochs = 0
        self.total_jobs = 0
        self.total_wins = 0
        self.total_cancels = 0

    def is_malicious(self):
        return self.kind != "honest"


def build_agents(cfg):
    agents = []
    aid = 0

    num_honest = cfg["num_agents"]
    for _ in range(num_honest):
        agents.append(Agent(aid, kind="honest"))
        aid += 1

    num_cancellers = cfg["num_bid_cancellers"]
    for _ in range(num_cancellers):
        agents.append(Agent(aid, kind="canceller"))
        aid += 1

    num_griefers = cfg["num_griefers"]
    for _ in range(num_griefers):
        agents.append(Agent(aid, kind="griefer"))
        aid += 1

    num_sybil = cfg["num_sybil_controllers"]
    accts_per_sybil = cfg["sybil_accounts_per_controller"]
    for group in range(num_sybil):
        for _ in range(accts_per_sybil):
            agents.append(Agent(aid, kind="sybil", group_id=group))
            aid += 1

    return agents
