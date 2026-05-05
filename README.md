# Casualty Catastrophe Model — Agentic Explorer

A probabilistic casualty catastrophe model with an agentic AI interface for natural-language scenario exploration, sensitivity analysis, and uncertainty quantification.

This project simulates losses from emerging and systemic liability risks (mass torts, product-defect cascades, latent exposures) using the standard cat-modeling architecture — frequency / severity / financial modules — and exposes the model through an OpenAI-function-calling agent backed by a schema-defined tool registry, with a deterministic eval harness for agent reliability.

## Why this exists

Casualty catastrophe models help insurers price and capitalize against liability tail risks. They produce highly non-intuitive outputs (heavy-tailed distributions, return-period PMLs, parameter-uncertainty bands) that benefit from interactive exploration. Rather than asking an analyst to drive a notebook, this project lets a user ask questions in natural language — *"how does the 250-year PML change if claim severity is 30% higher?"* — and have an agent translate that into the right model invocation, return the numbers with confidence intervals, and explain the result.

## Architecture

```
                 ┌─────────────────────────────┐
   user query ─► │       OpenAI agent          │
                 │  (function-calling loop,    │
                 │   max-iter guardrail,       │
                 │   structured trace)         │
                 └──────────────┬──────────────┘
                                │  tool calls
                 ┌──────────────▼──────────────┐
                 │  Schema-defined tool        │
                 │  registry (6 tools):        │
                 │  describe / metrics /       │
                 │  pml_with_ci / sensitivity /│
                 │  tornado / update_assumption│
                 └──────────────┬──────────────┘
                                │  Python calls
                 ┌──────────────▼──────────────┐
                 │  Casualty cat model         │
                 │  ┌────────┐ ┌────────────┐  │
                 │  │Freq    │ │Severity    │  │
                 │  │NegBin  │ │Lognormal   │  │
                 │  └────────┘ └────────────┘  │
                 │  ┌──────────────────────┐   │
                 │  │ Exposure portfolio   │   │
                 │  │ Financial module     │   │
                 │  │ (deductibles+limits) │   │
                 │  └──────────────────────┘   │
                 │  ┌──────────────────────┐   │
                 │  │ Monte Carlo engine   │   │
                 │  │ AAL / EP curve / PML │   │
                 │  │ TVaR / bootstrap CIs │   │
                 │  └──────────────────────┘   │
                 └─────────────────────────────┘
```

## Modeling approach

The cat model uses standard insurance-industry methodology:

- **Frequency:** Negative Binomial (Poisson nested inside, with overdispersion to allow for event clustering)
- **Severity:** Lognormal (parameterized by mean + CV for domain-friendly assumption setting)
- **Exposure:** Portfolio of policies with industry segments, exposure weights, deductibles, and per-event limits
- **Financial module:** Vectorized application of contract terms — `min(max(allocated_loss − deductible, 0), limit)` summed across the portfolio
- **Monte Carlo engine:** Runs ~10⁵ simulated years, vectorized across events; produces the standard cat-model output set:
  - **AAL** — Average Annual Loss (with Monte Carlo standard error)
  - **EP curve** — Exceedance Probability across standard return periods (2y → 1000y)
  - **PML** — Probable Maximum Loss at any return period, with **bootstrap 95% confidence intervals** to quantify simulation uncertainty in the tail
  - **TVaR** — Tail Value at Risk for capital-adequacy framing
- **Sensitivity:** One-at-a-time and tornado analysis ranking parameters by impact on chosen metric

## Agent design

- **Schema-defined tool registry** — each tool declares its OpenAI function-calling JSON Schema, making the registry composable and easy to extend (mirrors the MCP pattern)
- **Stateful model context** — shared across tool calls, with **result caching** keyed on the model configuration so follow-up questions don't re-simulate when assumptions haven't changed
- **What-if support** — `update_assumption` mutates the shared context, so subsequent tool calls reflect the new scenario
- **Guardrails** — max-iteration cap on the tool-use loop; malformed tool arguments are returned to the model as error tool-results rather than crashing the agent
- **Structured trace** — every LLM call, tool invocation, and tool result is captured with timestamps, token counts, and elapsed time, enabling post-hoc debugging and evaluation

## Evaluation harness

Beyond unit tests, the project includes a **deterministic eval suite** for agent reliability:

- **Tool-selection correctness** — did the agent dispatch the right tool(s) for the query?
- **Forbidden-tool detection** — did the agent correctly refuse unsupported requests rather than hallucinate a capability?
- **Numerical correctness** — do the numbers in the response match the ground-truth tool outputs (within tolerance)?

Run with:

```bash
export OPENAI_API_KEY=sk-...
python -m scripts.run_eval
```

## Project layout

```
casualty-cat-explorer/
├── src/
│   ├── cat_model/         # Probabilistic model: distributions, exposure,
│   │                      # financial, simulation engine, sensitivity
│   └── agent/             # OpenAI function-calling agent: tools registry,
│                          # runtime, eval harness
├── tests/                 # 43 unit + integration tests (pytest)
├── scripts/
│   ├── demo.py            # Interactive REPL
│   └── run_eval.py        # Run the deterministic eval suite
├── notebooks/             # Worked example walkthrough
└── .github/workflows/     # CI: ruff + pytest on Python 3.10/3.11/3.12
```

## Quick start

```bash
git clone https://github.com/<user>/casualty-cat-explorer.git
cd casualty-cat-explorer
pip install -e ".[dev]"

# Run tests (no API key needed)
pytest tests/ -v

# Try the agent (requires OpenAI API key)
export OPENAI_API_KEY=sk-...
python -m scripts.demo
```

## Example session

```
You> What's the 250-year PML, and how confident should I be in that estimate?

Agent> The 250-year PML is approximately $305,000,000. Bootstrap 95% confidence
       interval is [$281M, $343M], a width of about 20% of the point estimate —
       indicating moderate Monte Carlo uncertainty in the tail. For tighter
       bounds, increase n_years on the simulation.

[1 tool calls, 2 LLM calls, 3.4s]

You> Which assumption matters most for that?

Agent> Severity mean is the dominant driver. Under a +/-25% shock:
       1. severity_mean       impact range $169M  (≈55% swing)
       2. severity_cv         impact range  $94M
       3. frequency_mean      impact range  $71M
       4. frequency_dispersion impact range $22M
       Severity assumptions warrant the most validation effort.

[1 tool calls, 2 LLM calls, 4.1s]
```

## Limitations & future work

- The model is intentionally simplified: a single hazard process, no cross-event correlation across years, no reinsurance treaty layer above the primary insurance contract. Production casualty cat models add aggregate caps, clash provisions, and reinsurance hierarchies.
- Severity allocation across policies is proportional to exposure share. A real model would calibrate damage functions to industry segment.
- The eval suite is starter-grade. A production agent-eval would include adversarial cases, judge-LLM regression tracking, and longitudinal stability tests.

## License

MIT
