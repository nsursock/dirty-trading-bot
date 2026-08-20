# General

Start every reply with "Yes master!!"

## Writing Commit Messages

Imperative Mood in Subject: Write the summary line in the imperative present tense (e.g., Add user authentication or Fix memory leak in buffer, rather than Added or Fixes). This aligns with standard Git generator convention (If applied, this commit will <subject>).

Subject Line Formatting: Limit the first line to 50–72 characters, capitalize the first letter, and do not end the line with a period.

Structured Body: Separate the subject line from the body with a blank line. Use the body to explain what changed and why, rather than how (the diff shows how).

Reference Issue Tracking: Link relevant issue numbers or pull request references in the commit body (e.g., Closes #142 or Ref #88).

## Libraries / Development

Every python process launch should use venv/bin/python.
Run python scripts unsandboxed in IDEs.
Use caffeinate -dims when launching a python process.
This trading bot should use Apple MLX in the hot path.
When executing costly python processes, always use tqdm feedback.

# Troubleshooting

If the only nan in the training CSVs happen on the last row, ignore it: it means the episode didn't complete.

# IDE

Don't create a canvas unless I explicitly ask you to.

# Goal of the project

# Goal

A profitable equity curve, plotted cumulatively over time, looks like a **staircase climbing up and to the right**—not a straight line, but a persistent upward drift with periodic setbacks.

Visually, you'd see:

- **Upward trajectory** — the line starts lower left and ends higher right, reflecting net gains over the period.
- **Shallow pullbacks (drawdowns)** — small dips where the curve briefly bends downward before resuming its climb. The depth and frequency of these dips reveal the strategy's risk: a good curve has *shallow, short-lived* dips rather than deep, prolonged ones.
- **Smoothness vs. jaggedness** — a smooth, gently curving line suggests consistent, low-volatility returns; a jagged, sawtooth line means high variance—big wins followed by big losses.
- **Compounding curvature** — ideally, the slope gently *steepens* over time as profits compound, creating a subtle convex shape.
- **Recovery speed** — after each dip, the curve recovers to a new high relatively quickly; long flat stretches or slow recoveries signal fragility.

In short: it should look like a **healthy mountain ridge**—ascending with occasional valleys, not a roller coaster or a flat line with one sudden spike.

For a curve that *looks* good—smooth, steadily climbing, with shallow and brief dips—**the best match is actually the Ulcer Performance Index (UPI), also called the Martin Ratio**, not Calmar.

Here's why, and how the metrics map to what you see:

| Metric | What it captures | Visual correspondence |
|--------|-----------------|----------------------|
| **Calmar** | Annual return ÷ Max Drawdown | Only the **single deepest valley**. A curve could look terrible day-to-day but score well if it avoided one catastrophic drop. |
| **Sharpe** | Return ÷ volatility (std dev) | Penalizes *all* volatility, including upside spikes. A jagged but upward curve might score worse than a flat one. |
| **Sortino** | Return ÷ downside volatility | Better than Sharpe, but still treats each down day equally—doesn't care if you recover quickly or stay underwater for months. |
| **Ulcer Performance Index (Martin)** | Return ÷ Ulcer Index | **Both depth AND duration of drawdowns**. Shallow, short dips = low Ulcer Index = high UPI. This directly rewards the "smooth staircase" look. |

**The Ulcer Index** is essentially the root-mean-square of percentage drawdowns over time. So if your equity curve spends *any* time in a hole—deep or prolonged—the Ulcer Index spikes, and UPI drops. That aligns perfectly with visual intuition: a curve that *looks* painless scores well.

**Calmar is useful** as a quick "worst-case sanity check," but it's a blunt instrument. If you want one number that says *"this curve looks healthy and climbs smoothly,"* **UPI / Martin Ratio** is the answer.

If you can only use common metrics, **Sortino** is the next best proxy.