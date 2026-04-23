'''
RILEY ANDERSON, HANNAH SMITH

This file contains the market simulation itself. It handles initialization
of all agent behaviour, tracks market variables, and optionally plots a number of statistics.
'''
# IMPORTS
import random
import matplotlib.pyplot as plt
from agents import build_agents
from cfg import CFG

# CONSTANTS

# store market data
prices = {"clearing": []}

# GPU CLASS
class GPU():
    def __init__(self):
        self.max_price = CFG['max_price']
        self.owner = None
        self.last_price = 0


# MAIN SIM CLASS
class Sim():
    def __init__(self):
        self.rng = random.Random(CFG["seed"])

        # store agent and gpu class instances
        self.agents = build_agents(CFG)
        self.gpus = [GPU() for gpu in range(CFG['num_gpus'])]

        # store auction parameters
        self.num_epochs = CFG['num_epochs']
        self.bid_increment = CFG["bid_increment"]
        self.max_price = CFG["max_price"]
        self.arrival_rate = CFG["arrival_rate"]
        self.min_value = CFG["min_value"]
        self.max_value = CFG["max_value"]

        # probability controls for malicious behavior
        self.cancel_prob = CFG["cancel_prob"]
        self.sybil_bid_prob = CFG["sybil_bid_prob"]
        self.sybil_cancel_prob = CFG["sybil_cancel_prob"]
        self.griefer_cancel_prob = CFG["griefer_cancel_prob"]

        # malicious bid-shaping controls
        self.griefer_bid_boost_min = CFG["griefer_bid_boost_min"]
        self.griefer_bid_boost_max = CFG["griefer_bid_boost_max"]
        self.sybil_bid_floor_fraction = CFG["sybil_bid_floor_fraction"]

        # metrics
        self.metrics = {
            "served_honest": [],
            "served_total": [],
            "cancelled_wins": [],
            "clearing_price": [],
            "avg_honest_delay": [],
        }

    def _value_bid(self, agent):
        base = self.rng.randint(self.min_value, self.max_value)

        if agent.kind == "griefer":
            # Griefers overbid to increase chance of blocking real users.
            boost = self.rng.randint(self.griefer_bid_boost_min, self.griefer_bid_boost_max)
            base = min(self.max_price, base + boost)

        if agent.kind == "sybil":
            # Sybil accounts bid near the top of the market to inflate prices.
            high_floor = max(int(self.max_price * self.sybil_bid_floor_fraction), self.min_value)
            base = self.rng.randint(high_floor, self.max_price)

        return min(self.max_price, max(self.min_value, base))

    def _collect_bids(self):
        bids = []

        for agent in self.agents:
            should_bid = agent.waiting or (self.rng.random() < self.arrival_rate)

            if agent.kind == "sybil" and self.rng.random() > self.sybil_bid_prob:
                should_bid = False

            if not should_bid:
                continue

            bid_price = self._value_bid(agent)
            bid_price = bid_price - (bid_price % self.bid_increment)
            bid_price = max(self.min_value, bid_price)

            # If this is new demand, mark the agent as waiting for service.
            if not agent.waiting:
                agent.waiting = True

            bids.append({"agent": agent, "price": bid_price})

        return bids

    def _should_cancel(self, winner, has_honest_loser):
        agent = winner["agent"]

        if agent.kind == "canceller":
            return self.rng.random() < self.cancel_prob

        if agent.kind == "sybil":
            # Sybil wins are generally synthetic demand and get cancelled.
            return self.rng.random() < self.sybil_cancel_prob

        if agent.kind == "griefer":
            # Griefers cancel primarily when they successfully delayed honest users.
            if not has_honest_loser:
                return False
            return self.rng.random() < self.griefer_cancel_prob

        return False

    def _resolve_auction(self, bids):
        if not bids:
            return [], [], 0
        #HANNAH this is where the winners are selected 
        bids_sorted = sorted(bids, key=lambda x: x["price"], reverse=True)
        winners = bids_sorted[:len(self.gpus)]
        losers = bids_sorted[len(self.gpus):]

        clearing = winners[-1]["price"] if winners else 0

        honest_losers = any(item["agent"].kind == "honest" for item in losers)

        accepted = []
        cancelled = []
        for winner in winners:
            if self._should_cancel(winner, honest_losers):
                cancelled.append(winner)
            else:
                accepted.append(winner)

        return accepted, cancelled, clearing

    def _update_delays(self, served_agents):
        served_ids = {agent.agent_id for agent in served_agents}

        for agent in self.agents:
            if not agent.waiting:
                continue

            if agent.agent_id in served_ids:
                agent.waiting = False
                agent.wait_epochs = 0
                agent.total_jobs += 1
            else:
                agent.wait_epochs += 1
    #measure of resourse allocation, where 0 is perfectly unequal and 1 is perfectly equal
    def _jain_index(self, values):
        n = len(values)
        if n == 0:
            return 1.0
        sum_vals = sum(values)
        sum_squares = sum(v ** 2 for v in values)
        if sum_squares == 0:
            return 1.0
        return (sum_vals ** 2) / (n * sum_squares)

    def _fairness_snapshot(self):
        #fairness among honest agents only
        honest = [a for a in self.agents if a.kind == "honest"]
        service_vals = [a.total_jobs for a in honest]
        jains_honest = self._jain_index(service_vals)
        #overall market fairness, including malicious agents 
        service_vals_all = [a.total_jobs for a in self.agents]
        jains_all = self._jain_index(service_vals_all)

        #measures how long honest agents have been waiting without service, to capture starvation
        starvation_threshold = CFG["starvation_threshold"] * self.num_epochs
        starved = sum(1 for a in honest if a.waiting and a.wait_epochs >= starvation_threshold)
        starvation_rate = starved / len(honest) if honest else 0.0

        return jains_honest, jains_all, starvation_rate

    def start_auction_round(self, epoch):
        bids = self._collect_bids()
        accepted, cancelled, clearing = self._resolve_auction(bids)

        served_agents = []
        for i, gpu in enumerate(self.gpus):
            if i < len(accepted):
                winner = accepted[i]
                gpu.owner = winner["agent"].agent_id
                gpu.last_price = winner["price"]
                winner["agent"].total_wins += 1
                served_agents.append(winner["agent"])
            else:
                gpu.owner = None
                gpu.last_price = 0

        for item in cancelled:
            item["agent"].total_wins += 1
            item["agent"].total_cancels += 1

        self._update_delays(served_agents)
        honest_fairness, overall_fairness, starvation_rate = self._fairness_snapshot()

        honest_waits = [a.wait_epochs for a in self.agents if a.kind == "honest" and a.waiting]
        avg_wait = (sum(honest_waits) / len(honest_waits)) if honest_waits else 0.0

        self.metrics["served_honest"].append(sum(1 for a in served_agents if a.kind == "honest"))
        self.metrics["served_total"].append(len(served_agents))
        self.metrics["cancelled_wins"].append(len(cancelled))
        self.metrics["clearing_price"].append(clearing)
        self.metrics["avg_honest_delay"].append(avg_wait)

        prices["clearing"].append(clearing)

        return {
            "epoch": epoch,
            "num_bids": len(bids),
            "served": len(served_agents),
            "cancelled": len(cancelled),
            "clearing_price": clearing,
            "avg_honest_delay": avg_wait,
            "honest_fairness": honest_fairness,
            "overall_fairness": overall_fairness,
            "starvation_rate": starvation_rate,
        }

    def run(self, plot=False, verbose=True):
        logs = []
        for epoch in range(self.num_epochs):
            stats = self.start_auction_round(epoch)
            logs.append(stats)
            if verbose:
                print(
                    f"epoch={epoch:03d} bids={stats['num_bids']:02d} "
                    f"served={stats['served']:02d} cancelled={stats['cancelled']:02d} "
                    f"price={stats['clearing_price']:03d} delay={stats['avg_honest_delay']:.2f}"
                )

        if plot:
            self.plot_metrics()

        return logs

    def plot_metrics(self):
        epochs = list(range(self.num_epochs))

        fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

        ax[0][0].plot(epochs, self.metrics["clearing_price"], color="tab:blue")
        ax[0][0].set_title("Clearing Price")
        ax[0][0].set_xlabel("Epoch")

        ax[0][1].plot(epochs, self.metrics["cancelled_wins"], color="tab:red")
        ax[0][1].set_title("Cancelled Winning Bids")
        ax[0][1].set_xlabel("Epoch")

        ax[1][0].plot(epochs, self.metrics["served_honest"], color="tab:green")
        ax[1][0].set_title("Honest Jobs Served")
        ax[1][0].set_xlabel("Epoch")

        ax[1][1].plot(epochs, self.metrics["avg_honest_delay"], color="tab:orange")
        ax[1][1].set_title("Average Honest Delay")
        ax[1][1].set_xlabel("Epoch")

        plt.show()


def plot_multi_round_averages(
    avg_clearing_by_epoch,
    avg_delay_by_epoch,
    avg_served_by_epoch,
    avg_cancelled_by_epoch,
):
    epochs = list(range(len(avg_clearing_by_epoch)))

    fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

    ax[0][0].plot(epochs, avg_clearing_by_epoch, color="tab:blue")
    ax[0][0].set_title("Avg Clearing Price Across Rounds")
    ax[0][0].set_xlabel("Epoch")
    ax[0][0].set_ylabel("Price")

    ax[0][1].plot(epochs, avg_cancelled_by_epoch, color="tab:red")
    ax[0][1].set_title("Avg Cancelled Wins Across Rounds")
    ax[0][1].set_xlabel("Epoch")
    ax[0][1].set_ylabel("Cancelled wins")

    ax[1][0].plot(epochs, avg_served_by_epoch, color="tab:green")
    ax[1][0].set_title("Avg Jobs Served Across Rounds")
    ax[1][0].set_xlabel("Epoch")
    ax[1][0].set_ylabel("Served jobs")

    ax[1][1].plot(epochs, avg_delay_by_epoch, color="tab:orange")
    ax[1][1].set_title("Avg Honest Delay Across Rounds")
    ax[1][1].set_xlabel("Epoch")
    ax[1][1].set_ylabel("Delay (epochs)")

    plt.show()

def plot_fairness_comparison(honest_jains, overall_jains):
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    ax.bar(["Honest", "Overall"], [honest_jains, overall_jains], color=["tab:green", "tab:blue"])
    ax.set_title("Jain's Fairness Index")
    ax.set_ylim(0, 1)

    plt.show()

def plot_starvation_rate(starvation_rates):
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    ax.plot(starvation_rates, color="tab:red")
    ax.set_title("Starvation Rate Among Honest Agents")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Starvation Rate")

    plt.show()


if __name__ == "__main__":
    num_rounds = CFG["num_rounds"]
    num_epochs = CFG["num_epochs"]

    clearing_sum_by_epoch = [0.0] * num_epochs
    delay_sum_by_epoch = [0.0] * num_epochs
    served_sum_by_epoch = [0.0] * num_epochs
    cancelled_sum_by_epoch = [0.0] * num_epochs
    honest_fairness_sum_by_epoch = [0.0] * num_epochs
    overall_fairness_sum_by_epoch = [0.0] * num_epochs
    starvation_sum_by_epoch = [0.0] * num_epochs


    for _ in range(num_rounds):
        sim = Sim()
        round_stats = sim.run(plot=False, verbose=CFG["verbose"])

        for epoch_idx, auction_cycle in enumerate(round_stats):
            clearing_sum_by_epoch[epoch_idx] += auction_cycle["clearing_price"]
            delay_sum_by_epoch[epoch_idx] += auction_cycle["avg_honest_delay"]
            served_sum_by_epoch[epoch_idx] += auction_cycle["served"]
            cancelled_sum_by_epoch[epoch_idx] += auction_cycle["cancelled"]
            honest_fairness_sum_by_epoch[epoch_idx] += auction_cycle["honest_fairness"]
            overall_fairness_sum_by_epoch[epoch_idx] += auction_cycle["overall_fairness"]
            starvation_sum_by_epoch[epoch_idx] += auction_cycle["starvation_rate"]


    avg_clearing_by_epoch = [val / num_rounds for val in clearing_sum_by_epoch]
    avg_delay_by_epoch = [val / num_rounds for val in delay_sum_by_epoch]
    avg_served_by_epoch = [val / num_rounds for val in served_sum_by_epoch]
    avg_cancelled_by_epoch = [val / num_rounds for val in cancelled_sum_by_epoch]
    avg_honest_fairness_by_epoch = [val / num_rounds for val in honest_fairness_sum_by_epoch]
    avg_overall_fairness_by_epoch = [val / num_rounds for val in overall_fairness_sum_by_epoch]
    avg_starvation_by_epoch = [val / num_rounds for val in starvation_sum_by_epoch]

    avg_clearing_overall = sum(avg_clearing_by_epoch) / num_epochs
    avg_delay_overall = sum(avg_delay_by_epoch) / num_epochs

    print(f"Avg clearing price (overall): {avg_clearing_overall:.2f}")
    print(f"Avg honest delay (overall): {avg_delay_overall:.2f}")
    print(f"Avg honest fairness (overall): {avg_honest_fairness_by_epoch[-1]:.4f}")
    print(f"Avg overall fairness (overall): {avg_overall_fairness_by_epoch[-1]:.4f}")
    print(f"Avg starvation rate (overall): {avg_starvation_by_epoch[-1]:.4f}")

    if CFG["plot"]:
        plot_multi_round_averages(
            avg_clearing_by_epoch,
            avg_delay_by_epoch,
            avg_served_by_epoch,
            avg_cancelled_by_epoch,
        )
        plot_fairness_comparison(
            avg_honest_fairness_by_epoch[-1],      
            avg_overall_fairness_by_epoch[-1], 
        )
        plot_starvation_rate(avg_starvation_by_epoch)