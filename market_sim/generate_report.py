"""
Generate an HTML report comparing market behavior across many malicious-agent distributions.

Usage:
    python generate_report.py
"""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

from cfg import CFG
from sim import Sim


def total_accounts(cfg):
    return (
        cfg["num_agents"]
        + cfg["num_bid_cancellers"]
        + cfg["num_griefers"]
        + cfg["num_sybil_controllers"] * cfg["sybil_accounts_per_controller"]
    )


def make_scenario_overrides(base_cfg, cancel_share=0.0, griefer_share=0.0, sybil_share=0.0):
    total = total_accounts(base_cfg)
    apc = base_cfg["sybil_accounts_per_controller"]

    cancellers = int(total * cancel_share)
    griefers = int(total * griefer_share)

    sybil_accounts_requested = int(total * sybil_share)
    sybil_controllers = sybil_accounts_requested // apc
    sybil_accounts = sybil_controllers * apc

    used = cancellers + griefers + sybil_accounts
    if used > total:
        overflow = used - total
        griefers = max(0, griefers - overflow)
        used = cancellers + griefers + sybil_accounts

    honest = max(0, total - used)

    return {
        "num_agents": honest,
        "num_bid_cancellers": cancellers,
        "num_griefers": griefers,
        "num_sybil_controllers": sybil_controllers,
    }


def default_scenarios(base_cfg):
    scenarios = [
        {
            "name": "Honest Baseline",
            "description": "No malicious bidders.",
            "overrides": make_scenario_overrides(base_cfg, 0.0, 0.0, 0.0),
        },
        {
            "name": "Light Cancellers",
            "description": "About 10% cancellers in the market.",
            "overrides": make_scenario_overrides(base_cfg, cancel_share=0.10),
        },
        {
            "name": "Heavy Cancellers",
            "description": "About 25% cancellers in the market.",
            "overrides": make_scenario_overrides(base_cfg, cancel_share=0.25),
        },
        {
            "name": "Light Sybil",
            "description": "About 15% sybil-controlled accounts.",
            "overrides": make_scenario_overrides(base_cfg, sybil_share=0.15),
        },
        {
            "name": "Heavy Sybil",
            "description": "About 30% sybil-controlled accounts.",
            "overrides": make_scenario_overrides(base_cfg, sybil_share=0.30),
        },
        {
            "name": "Light Griefers",
            "description": "About 10% griefing bidders.",
            "overrides": make_scenario_overrides(base_cfg, griefer_share=0.10),
        },
        {
            "name": "Mixed Moderate",
            "description": "Balanced malicious mix at moderate intensity.",
            "overrides": make_scenario_overrides(base_cfg, 0.10, 0.10, 0.10),
        },
        {
            "name": "Mixed Aggressive",
            "description": "Balanced malicious mix at higher intensity.",
            "overrides": make_scenario_overrides(base_cfg, 0.20, 0.20, 0.20),
        },
    ]
    return scenarios


def run_one_scenario(scenario_name, description, cfg_override, base_cfg):
    original_cfg = deepcopy(CFG)
    try:
        CFG.clear()
        CFG.update(deepcopy(base_cfg))
        CFG.update(cfg_override)

        num_rounds = CFG["num_rounds"]
        num_epochs = CFG["num_epochs"]

        clearing_sum = [0.0] * num_epochs
        delay_sum = [0.0] * num_epochs
        served_sum = [0.0] * num_epochs
        cancelled_sum = [0.0] * num_epochs

        # Keep report generation quiet and avoid pop-up plots for each run.
        CFG["verbose"] = False
        CFG["plot"] = False

        for _ in range(num_rounds):
            sim = Sim()
            round_stats = sim.run(plot=False, verbose=False)

            for epoch_idx, cycle in enumerate(round_stats):
                clearing_sum[epoch_idx] += cycle["clearing_price"]
                delay_sum[epoch_idx] += cycle["avg_honest_delay"]
                served_sum[epoch_idx] += cycle["served"]
                cancelled_sum[epoch_idx] += cycle["cancelled"]

        avg_clearing = [value / num_rounds for value in clearing_sum]
        avg_delay = [value / num_rounds for value in delay_sum]
        avg_served = [value / num_rounds for value in served_sum]
        avg_cancelled = [value / num_rounds for value in cancelled_sum]

        overall = {
            "avg_clearing_price": sum(avg_clearing) / num_epochs,
            "avg_honest_delay": sum(avg_delay) / num_epochs,
            "avg_served": sum(avg_served) / num_epochs,
            "avg_cancelled": sum(avg_cancelled) / num_epochs,
        }

        distribution = {
            "honest": CFG["num_agents"],
            "cancellers": CFG["num_bid_cancellers"],
            "griefers": CFG["num_griefers"],
            "sybil_controllers": CFG["num_sybil_controllers"],
            "sybil_accounts": CFG["num_sybil_controllers"] * CFG["sybil_accounts_per_controller"],
            "total_accounts": total_accounts(CFG),
        }
        distribution["malicious_accounts"] = (
            distribution["cancellers"] + distribution["griefers"] + distribution["sybil_accounts"]
        )
        if distribution["total_accounts"] > 0:
            distribution["malicious_share"] = (
                distribution["malicious_accounts"] / distribution["total_accounts"]
            )
        else:
            distribution["malicious_share"] = 0.0

        return {
            "name": scenario_name,
            "description": description,
            "distribution": distribution,
            "overall": overall,
            "series": {
                "avg_clearing": avg_clearing,
                "avg_delay": avg_delay,
                "avg_served": avg_served,
                "avg_cancelled": avg_cancelled,
            },
        }
    finally:
        CFG.clear()
        CFG.update(original_cfg)


def build_html(report_title, generated_at, baseline_cfg, scenario_results):
    scenario_payload = json.dumps(scenario_results)
    base_cfg_payload = json.dumps(baseline_cfg)

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{report_title}</title>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
  <style>
    :root {{
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #52606d;
      --line: #d9e2ec;
      --accent: #2f6fed;
      --good: #1f9d55;
      --warn: #c05621;
      --bad: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background: radial-gradient(circle at top right, #e8eefb, var(--bg) 40%);
      color: var(--ink);
    }}
    .container {{
      max-width: 1240px;
      margin: 24px auto;
      padding: 0 16px 32px;
    }}
    .header {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 16px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 1.65rem; }}
    .meta {{ color: var(--muted); font-size: 0.95rem; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
      gap: 14px;
      margin: 16px 0;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
    }}
    .card h3 {{ margin: 0 0 8px; font-size: 1rem; }}
    .small {{ color: var(--muted); font-size: 0.9rem; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
    }}
    th, td {{ padding: 10px 9px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ background: #f8fafc; font-weight: 600; }}
    tr:last-child td {{ border-bottom: none; }}
    .section {{ margin-top: 18px; }}
    .chart-wrap {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      margin-top: 10px;
    }}
    .two-col {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    }}
    .pill {{
      display: inline-block;
      margin-right: 6px;
      margin-bottom: 6px;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 0.8rem;
      border: 1px solid var(--line);
      color: var(--muted);
      background: #f8fafc;
    }}
    .delta {{
      margin-left: 6px;
      font-weight: 600;
      font-size: 0.84rem;
      white-space: nowrap;
    }}
    .delta.positive {{
      color: var(--bad);
    }}
    .delta.negative {{
      color: var(--good);
    }}
    .delta.positive.good {{
      color: var(--good);
    }}
    .delta.negative.bad {{
      color: var(--bad);
    }}
    .delta.zero {{
      color: var(--muted);
      font-weight: 500;
    }}
  </style>
</head>
<body>
  <div class=\"container\">
    <div class=\"header\">
      <h1>{report_title}</h1>
      <div class=\"meta\">Generated: {generated_at}</div>
      <div class=\"meta\">Multi-round market simulation report comparing malicious-agent distributions.</div>
    </div>

    <div class=\"grid\">
      <div class=\"card\">
        <h3>Baseline Runtime</h3>
        <div class=\"small\">Epochs per round: <strong>{baseline_cfg["num_epochs"]}</strong></div>
        <div class=\"small\">Rounds per scenario: <strong>{baseline_cfg["num_rounds"]}</strong></div>
        <div class=\"small\">GPUs: <strong>{baseline_cfg["num_gpus"]}</strong></div>
      </div>
      <div class=\"card\">
        <h3>Price Model</h3>
        <div class=\"small\">Min value: <strong>{baseline_cfg["min_value"]}</strong></div>
        <div class=\"small\">Max value: <strong>{baseline_cfg["max_value"]}</strong></div>
        <div class=\"small\">Bid increment: <strong>{baseline_cfg["bid_increment"]}</strong></div>
      </div>
      <div class=\"card\">
        <h3>Behaviour Probabilities</h3>
        <span class=\"pill\">arrival={baseline_cfg["arrival_rate"]}</span>
        <span class=\"pill\">cancel={baseline_cfg["cancel_prob"]}</span>
        <span class=\"pill\">sybil_bid={baseline_cfg["sybil_bid_prob"]}</span>
        <span class=\"pill\">sybil_cancel={baseline_cfg["sybil_cancel_prob"]}</span>
        <span class=\"pill\">griefer_cancel={baseline_cfg["griefer_cancel_prob"]}</span>
      </div>
    </div>

    <div class=\"section\">
      <h2>Scenario Summary</h2>
      <table>
        <thead>
          <tr>
            <th>Scenario</th>
            <th>Malicious Share</th>
            <th>Avg Clearing Price</th>
            <th>Avg Honest Delay</th>
            <th>Avg Served</th>
            <th>Avg Cancelled</th>
          </tr>
        </thead>
        <tbody id=\"summaryBody\"></tbody>
      </table>
    </div>

    <div class=\"section\">
      <h2>Scenario Details</h2>
      <div class=\"grid\" id=\"scenarioCards\"></div>
    </div>
  </div>

  <script>
    const scenarioData = {scenario_payload};
    const baselineCfg = {base_cfg_payload};

    const colors = [
      "#2f6fed", "#1f9d55", "#b42318", "#c05621",
      "#2d9cdb", "#7f56d9", "#0f766e", "#bc4c00"
    ];

    function round2(x) {{
      return Number.parseFloat(x).toFixed(2);
    }}

    function signed2(x) {{
      const value = Number.parseFloat(x);
      const prefix = value >= 0 ? "+" : "";
      return `${{prefix}}${{value.toFixed(2)}}`;
    }}

    function metricWithDelta(value, baselineValue, invertSignColors = false) {{
      const delta = value - baselineValue;
      const zeroish = Math.abs(delta) < 1e-9;

      if (zeroish) {{
        return `${{round2(value)}}<span class=\"delta zero\">(0.00%)</span>`;
      }}

      let pctText = "n/a";
      if (Math.abs(baselineValue) >= 1e-9) {{
        const pctDelta = (delta / baselineValue) * 100;
        pctText = `${{signed2(pctDelta)}}%`;
      }}

      const negative = delta < 0;
      let deltaClass = negative ? "delta negative" : "delta positive";

      if (invertSignColors) {{
        // For delay column, negative change should be red and positive should be green.
        deltaClass += negative ? " bad" : " good";
      }}

      return `${{round2(value)}}<span class=\"${{deltaClass}}\">(${{pctText}})</span>`;
    }}

    function metricWithNumberDelta(value, baselineValue, invertSignColors = false) {{
      const delta = value - baselineValue;
      const zeroish = Math.abs(delta) < 1e-9;

      if (zeroish) {{
        return `${{round2(value)}}<span class=\"delta zero\">(0.00)</span>`;
      }}

      const negative = delta < 0;
      let deltaClass = negative ? "delta negative" : "delta positive";

      if (invertSignColors) {{
        deltaClass += negative ? " bad" : " good";
      }}

      return `${{round2(value)}}<span class=\"${{deltaClass}}\">(${{signed2(delta)}})</span>`;
    }}

    function makeSummaryTable() {{
      const body = document.getElementById("summaryBody");
      const baseline = scenarioData.length > 0 ? scenarioData[0].overall : null;
      const rows = scenarioData.map((item) => {{
        const baseClearing = baseline ? baseline.avg_clearing_price : item.overall.avg_clearing_price;
        const baseDelay = baseline ? baseline.avg_honest_delay : item.overall.avg_honest_delay;
        const baseServed = baseline ? baseline.avg_served : item.overall.avg_served;
        const baseCancelled = baseline ? baseline.avg_cancelled : item.overall.avg_cancelled;

        return `<tr>
          <td><strong>${{item.name}}</strong><div class=\"small\">${{item.description}}</div></td>
          <td>${{round2(item.distribution.malicious_share * 100)}}%</td>
          <td>${{metricWithDelta(item.overall.avg_clearing_price, baseClearing)}}</td>
          <td>${{metricWithDelta(item.overall.avg_honest_delay, baseDelay)}}</td>
          <td>${{metricWithDelta(item.overall.avg_served, baseServed, true)}}</td>
          <td>${{metricWithNumberDelta(item.overall.avg_cancelled, baseCancelled)}}</td>
        </tr>`;
      }}).join("");
      body.innerHTML = rows;
    }}

    function makeScenarioCards() {{
      const host = document.getElementById("scenarioCards");
      host.innerHTML = scenarioData.map((item) => {{
        return `<div class=\"card\">
          <h3>${{item.name}}</h3>
          <div class=\"small\">${{item.description}}</div>
          <div class=\"small\">Honest: <strong>${{item.distribution.honest}}</strong></div>
          <div class=\"small\">Cancellers: <strong>${{item.distribution.cancellers}}</strong></div>
          <div class=\"small\">Griefers: <strong>${{item.distribution.griefers}}</strong></div>
          <div class=\"small\">Sybil controllers: <strong>${{item.distribution.sybil_controllers}}</strong></div>
          <div class=\"small\">Sybil accounts: <strong>${{item.distribution.sybil_accounts}}</strong></div>
          <div class=\"small\">Total accounts: <strong>${{item.distribution.total_accounts}}</strong></div>
        </div>`;
      }}).join("");
    }}

    function buildBarChart(canvasId, label, values, color) {{
      const labels = scenarioData.map((s) => s.name);
      new Chart(document.getElementById(canvasId), {{
        type: "bar",
        data: {{
          labels,
          datasets: [{{
            label,
            data: values,
            backgroundColor: color,
          }}]
        }},
        options: {{
          responsive: true,
          plugins: {{ legend: {{ display: false }} }}
        }}
      }});
    }}

    function buildLineChart(canvasId, title, seriesKey, yLabel) {{
      const epochs = Array.from({{ length: baselineCfg.num_epochs }}, (_, i) => i);
      const datasets = scenarioData.map((s, idx) => ({{
        label: s.name,
        data: s.series[seriesKey],
        borderColor: colors[idx % colors.length],
        tension: 0.25,
        pointRadius: 0,
        borderWidth: 2,
      }}));

      new Chart(document.getElementById(canvasId), {{
        type: "line",
        data: {{ labels: epochs, datasets }},
        options: {{
          responsive: true,
          plugins: {{ title: {{ display: true, text: title }} }},
          scales: {{
            x: {{ title: {{ display: true, text: "Epoch" }} }},
            y: {{ title: {{ display: true, text: yLabel }} }}
          }}
        }}
      }});
    }}

    makeSummaryTable();
    makeScenarioCards();
  </script>
</body>
</html>
"""


def generate_report(output_path=None):
    base_cfg = deepcopy(CFG)
    scenarios = default_scenarios(base_cfg)

    results = []
    for scenario in scenarios:
        result = run_one_scenario(
            scenario_name=scenario["name"],
            description=scenario["description"],
            cfg_override=scenario["overrides"],
            base_cfg=base_cfg,
        )
        results.append(result)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_title = "GPU Spot Market Multi-Scenario Report"
    html = build_html(report_title, ts, base_cfg, results)

    if output_path is None:
        out_dir = Path("reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"market_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        output_path = out_dir / file_name
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(html, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    report_file = generate_report()
    print(f"Report written to: {report_file}")
