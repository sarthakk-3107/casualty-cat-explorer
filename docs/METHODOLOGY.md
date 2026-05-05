# Methodology

This document explains the modeling choices in the casualty cat model and the design decisions in the agent layer. It's written for a technical reader who wants to understand *why* the project does what it does, not just *what* it does.

## Why casualty catastrophe modeling?

Property cat modeling — hurricanes, earthquakes, severe convective storms — has a 35-year industry track record. Every major reinsurer uses cat models to price risk and set capital. The methodology is well-defined: hazard, exposure, vulnerability, financial.

**Casualty** cat modeling is the harder, fuzzier sibling. Instead of physical damage from a measurable hazard (wind speed, ground acceleration), it deals with liability cascades:

- Mass torts — opioid litigation, talc/asbestos, PFAS, Roundup
- Systemic product defects propagating across many insureds
- Latent exposures surfacing years after policies were written
- Emerging risks — AI liability, cyber aggregation, climate litigation

The hazard is fundamentally legal/social, not physical. There's no analogue to wind speed. So casualty cat models lean heavily on:

1. Scenario construction (what's the loss-generating mechanism?)
2. Probabilistic reasoning under deep parameter uncertainty
3. Sensitivity to assumptions, since assumptions dominate the output

This project models a stylized casualty scenario — think a PFAS-style emerging mass tort affecting a portfolio of chemicals/pharma/consumer-products policies — and exposes the model through an agent that lets users explore exactly those three dimensions: scenario, probability, sensitivity.

## Modeling choices

### Frequency: Negative Binomial, not Poisson

The Negative Binomial distribution is the natural generalization of Poisson that admits **overdispersion** (variance > mean). For casualty events, this matters because:

- Mass tort events tend to cluster — a successful plaintiff verdict in one case spawns copycat litigation
- Class certifications, regulatory triggers, and discovery findings produce correlated waves of claims
- Pure Poisson (variance = mean) underestimates tail clustering

We parameterize by `(mean, dispersion)` rather than the standard `(n, p)` because dispersion has an intuitive interpretation: dispersion = 1 → Poisson; dispersion > 1 → clustering. This makes assumption-setting tractable for domain experts who think in terms of "how clumpy are events?"

### Severity: Lognormal

Lognormal is the workhorse distribution for liability losses. It is:

- Strictly positive (losses can't go negative)
- Heavy-tailed in log space (thin compared to Pareto, but heavier than Normal)
- Closed-form for moments and quantiles
- Parameterizable from `(mean, CV)`, both of which are domain-friendly

For deeper tails (catastrophic single-event losses), a Generalized Pareto fitted to threshold exceedances would be more accurate. We treat that as future work.

### Financial module: per-policy contract application

Each event's ground-up loss is allocated across the portfolio in proportion to exposure share. Per-policy deductibles and limits are then applied:

```
allocated_i = ground_up * share_i
insured_i  = min(max(allocated_i − deductible_i, 0), limit_i)
total      = sum(insured_i)
```

This is implemented in vectorized form via NumPy broadcasting, applying contract terms to all events × all policies in a single `(n_events, n_policies)` matrix operation.

The proportional-allocation assumption is a simplification — real casualty events affect different insureds non-uniformly based on exposure to the proximate cause. A production model would condition allocation on industry segment and event class. We document this as a known limitation.

### Output metrics: AAL, EP curve, PML, TVaR

- **AAL** — the expected annual loss; the actuarially fair premium baseline
- **EP curve** — full loss distribution summarized at standard return periods (2y → 1000y)
- **PML at return period** — the loss exceeded with probability `1/return_period` in any year. Equivalent to a quantile of the annual loss distribution. Drives capital and reinsurance.
- **TVaR at return period** — expected loss conditional on exceeding the corresponding PML threshold. More robust than PML for capital adequacy because it captures the average severity of tail losses, not just the threshold

### Bootstrap confidence intervals on tail metrics

A 250-year PML estimated from 30,000 simulated years is sampling roughly 120 events in the tail. Monte Carlo error is non-negligible. We compute non-parametric bootstrap CIs on tail quantiles:

```python
for b in range(n_bootstrap):
    resample = rng.choice(annual_losses, size=n, replace=True)
    boot_estimates[b] = np.quantile(resample, q)
ci = np.quantile(boot_estimates, [0.025, 0.975])
```

This is computationally cheap (the sims are already done) and gives the user honest bounds. A user reading "the 250-year PML is $305M ± $30M (95% CI)" understands the answer differently than a user reading "$305M" — and the difference is exactly what an agentic interface should surface.

### Sensitivity: one-at-a-time and tornado

OAT sensitivity varies one parameter at a time. It misses interactions, but it's interpretable and matches how risk committees actually reason about assumptions.

Tornado analysis applies a uniform shock (e.g. ±25%) to every parameter, ranks by impact range on a chosen metric, and presents the result as the canonical horizontal-bar "tornado" view. This is the standard sensitivity output in actuarial and reinsurance practice.

Future work: Sobol indices for global sensitivity with interaction effects.

## Agent design choices

### Direct OpenAI function calling vs. an agent framework

We use `chat.completions.create(...)` with `tools=[...]` directly rather than wrapping in LangChain or LangGraph. Reasoning:

1. The agent has 6 tools and one workflow shape (multi-turn tool dispatch with a max-iter guardrail). Framework abstractions add weight without buying anything here.
2. Direct function calling is transparent — the trace is just messages, tool calls, and tool results. Easy to debug and easy to evaluate.
3. Anyone reading the code can tell exactly what's happening at each step.

Where a framework helps is in multi-agent orchestration, complex routing, or persisted state across sessions — none of which apply.

### Schema-defined tool registry

Each tool registers an OpenAI-compatible JSON Schema declaratively:

```python
Tool(
    name="compute_pml_with_ci",
    description="...",
    parameters_schema={"type": "object", "properties": {...}, "required": [...]},
    implementation=lambda return_period: _pml_with_ci(ctx, return_period),
)
```

This pattern mirrors the Model Context Protocol (MCP) approach: tools are first-class registered entities with declarative schemas, making them composable, inspectable, and mechanically easy to add to. Adding a new tool means adding one entry to `build_registry()`.

### Stateful context with caching

The `ModelContext` holds the current scenario and caches the last simulation result, keyed on the configuration tuple. Tool calls that don't change the scenario re-use the cached result; `update_assumption` invalidates the cache automatically. This keeps follow-up questions snappy.

### Trace as a first-class output

Every LLM call, tool invocation, and tool result is captured as a `TraceEvent`:

```python
TraceEvent(timestamp, kind, payload)
```

Where `kind` is one of `llm_call`, `tool_call`, `tool_result`, `final_response`. The trace enables:

- Debugging: see exactly what the agent did, in order, with timing
- Evaluation: deterministic checks on which tools were called for which queries
- Token accounting: input/output tokens recorded per LLM call
- Latency profiling: elapsed time per tool and per LLM call

### Guardrails

- `max_iterations` caps the tool-use loop; if exceeded, the agent returns a clean failure message with the trace intact
- Malformed tool-call arguments (invalid JSON, wrong types) are returned to the model as error tool-results rather than crashing the agent — the model can recover
- The system prompt explicitly instructs the agent to refuse unsupported capabilities rather than hallucinate (e.g., "no, I can't fit distributions to external data")
- The eval suite tests that refusal pattern with a `forbidden_tools` check

## What this project does *not* do

Being explicit about scope:

- **No real industry data.** The portfolio and parameters are synthetic. Connecting to real insurance data would require data partnerships and confidentiality controls.
- **No reinsurance treaty modeling.** Real cat models layer reinsurance — quota-share, surplus, excess-of-loss, stop-loss — on top of the primary insurance. We model only the primary contract.
- **No clash or aggregate provisions.** A single event can hit multiple policies; we account for that, but we don't model aggregate caps or clash provisions across events.
- **No event-set construction.** Real cat models build a stochastic catalog of physical or scenario events, each with parameters that drive damage. We work directly in loss space.
- **No reproducible production deployment.** This is a portfolio project, not a production system. Containerization, observability, and IaC could be added but aren't here.

These omissions are deliberate. The goal is to demonstrate the methodology and the agent design at a depth that's interview-defensible, not to ship a Verisk competitor.
