"""Day 4: collects the results of tests/test_layers.py and prints a report I can read.

Same idea as report.py from Day 3: the tests call record_*() while they run, and conftest.py
prints terminal_lines() after pytest finishes.
"""
import math

TOL = 1e-6                 # the checklist says: match PyTorch within about 1e-6

TORCH = []                 # one row per component compared with PyTorch (forward + gradients)
NUMERIC = []               # one row per numerical-gradient check (does not need PyTorch)
INIT_LOSS = []             # starting loss of fresh networks, compared with ln(10)
SIGNAL = []                # how big the signal is after many layers for different initializations
MUTATIONS = []             # deliberately broken code, and whether my tests noticed
OUTCOMES = []              # (test id, outcome, seconds), filled by conftest.py


def record_torch(component, forward_err, grad_errs):
    # grad_errs is a dict like {"x": 1e-16, "W": 2e-16}; I keep both the details and the worst one.
    worst_grad = max(grad_errs.values()) if grad_errs else 0.0
    worst = max(forward_err, worst_grad)
    TORCH.append(dict(component=component, forward=forward_err, grads=grad_errs,
                      worst_grad=worst_grad, worst=worst, ok=worst < TOL))


def record_numeric(name, worst, ok):
    NUMERIC.append(dict(name=name, worst=worst, ok=ok))


def record_init_loss(label, mean, spread, n_seeds):
    INIT_LOSS.append(dict(label=label, mean=mean, spread=spread, n=n_seeds))


def record_signal(label, rms, note):
    SIGNAL.append(dict(label=label, rms=rms, note=note))


def record_mutation(bug, description, caught, how):
    MUTATIONS.append(dict(bug=bug, description=description, caught=caught, how=how))


def _fmt(x):
    return "inf" if x == float("inf") else ("0" if x == 0 else f"{x:.1e}")


def _cut(s, n):
    return s if len(s) <= n else s[: n - 1] + "…"


def _margin(worst):
    if worst <= 0:
        return "inf"
    return f"{TOL / worst:.0e}x"


def summary():
    all_t = [t["worst"] for t in TORCH]
    return dict(
        torch_n=len(TORCH), torch_ok=sum(t["ok"] for t in TORCH),
        torch_worst=max(all_t) if all_t else 0.0,
        num_n=len(NUMERIC), num_ok=sum(n["ok"] for n in NUMERIC),
        num_worst=max((n["worst"] for n in NUMERIC), default=0.0),
        caught=sum(m["caught"] for m in MUTATIONS), bugs=len(MUTATIONS),
        passed=sum(o[1] == "passed" for o in OUTCOMES), tests=len(OUTCOMES))


def terminal_lines():
    L, s = [], summary()
    bar = "=" * 104
    L += [bar, " DAY 4 REPORT - LAYERS AND LOSSES (written by me, so I can read what each number proves)", bar, ""]

    # ---------------- A. PyTorch comparison ----------------
    if TORCH:
        L += [" A. MY LAYERS AND LOSSES vs PYTORCH (float64, the checklist wants about 1e-6)",
              "    What I check: I give my code and PyTorch the same numbers, then compare the output (forward)",
              "    and every gradient (backward). The error is max|mine - torch| / max(1, |mine|, |torch|).",
              f"    {'component':<40}{'forward':>10}{'worst grad':>12}{'margin':>10}   status",
              "    " + "-" * 84]
        for t in TORCH:
            L.append(f"    {_cut(t['component'], 39):<40}{_fmt(t['forward']):>10}{_fmt(t['worst_grad']):>12}"
                     f"{_margin(t['worst']):>10}   {'PASS' if t['ok'] else 'FAIL  <<<<<'}")
        L += ["",
              "    Gradient errors one by one (so I can see exactly which gradient is the least accurate):"]
        for t in TORCH:
            if t["grads"]:
                parts = "  ".join(f"{k}={_fmt(v)}" for k, v in t["grads"].items())
                L.append(f"      {_cut(t['component'], 39):<40}{_cut(parts, 60)}")
        L += ["",
              f"    -> {s['torch_ok']}/{s['torch_n']} components match PyTorch, worst error {_fmt(s['torch_worst'])}",
              "       (float64 rounding is about 1e-16, so errors near 1e-15 mean my code does the SAME math as PyTorch)", ""]
    else:
        L += [" A. PyTorch comparison: skipped (torch is not installed in this environment)", ""]

    # ---------------- B. numerical gradient check ----------------
    L += [" B. NUMERICAL GRADIENT CHECK (does not use PyTorch)",
          "    What I check: I nudge every weight by +-1e-6, recompute the loss, and compare the slope with my backward().",
          f"    {'check':<44}{'worst error':>12}{'margin':>10}   status",
          "    " + "-" * 76]
    for n in NUMERIC:
        L.append(f"    {_cut(n['name'], 43):<44}{_fmt(n['worst']):>12}{_margin(n['worst']):>10}   {'PASS' if n['ok'] else 'FAIL  <<<<<'}")
    L += ["", f"    -> {s['num_ok']}/{s['num_n']} checks pass, worst error {_fmt(s['num_worst'])}", ""]

    # ---------------- C. initial loss ----------------
    if INIT_LOSS:
        L += [" C. THE STARTING LOSS OF A FRESH 10-CLASS NETWORK (the checklist wants about ln(10) = 2.3026)",
              "    What I check: untrained network, random data, random labels. Average over many seeds.",
              f"    {'network':<52}{'mean loss':>10}{'spread':>9}{'distance from ln10':>20}",
              "    " + "-" * 91]
        for r in INIT_LOSS:
            L.append(f"    {_cut(r['label'], 51):<52}{r['mean']:>10.4f}{r['spread']:>9.3f}{abs(r['mean'] - math.log(10)):>20.4f}")
        L += ["",
              "    How I read this: with plain He init the last layer makes big logits, so the loss starts well above 2.30.",
              "    With a small last layer (gain 0.01) all the logits start near 0, every class gets about 10%, and the loss is ln(10).", ""]

    # ---------------- D. signal ----------------
    if SIGNAL:
        L += [" D. SIGNAL SIZE AFTER 10 Linear+ReLU LAYERS (input has RMS 1.0, so 1.0 means 'healthy')",
              f"    {'initialization':<44}{'output RMS':>12}   what happened",
              "    " + "-" * 82]
        for r in SIGNAL:
            L.append(f"    {_cut(r['label'], 43):<44}{r['rms']:>12.4g}   {r['note']}")
        L.append("")

    # ---------------- E. bug injection ----------------
    if MUTATIONS:
        L += [" E. BUG INJECTION (I break my code on purpose; a good test suite must notice)",
              f"    {'bug':<34}{'what I broke':<40}{'caught':>8}   noticed by",
              "    " + "-" * 100]
        for m in MUTATIONS:
            L.append(f"    {_cut(m['bug'], 33):<34}{_cut(m['description'], 39):<40}{'YES' if m['caught'] else 'NO <<<<<':>8}   {m['how']}")
        L += [f"    -> caught {s['caught']}/{s['bugs']} injected bugs", ""]

    ok = (s["passed"] == s["tests"] and s["torch_ok"] == s["torch_n"] and s["num_ok"] == s["num_n"]
          and s["caught"] == s["bugs"])
    L += [bar,
          f" VERDICT: {'ALL GOOD' if ok else 'PROBLEMS FOUND'} - {s['passed']}/{s['tests']} tests passed, "
          f"{s['torch_ok']}/{s['torch_n']} PyTorch matches, {s['num_ok']}/{s['num_n']} numerical checks, "
          f"{s['caught']}/{s['bugs']} bugs caught.", bar]
    return L


def write_markdown(path):
    s = summary()
    out = ["# Day 4 - layers and losses report", "",
           f"- PyTorch matches: **{s['torch_ok']}/{s['torch_n']}**, worst error {_fmt(s['torch_worst'])}",
           f"- numerical gradient checks: **{s['num_ok']}/{s['num_n']}**, worst error {_fmt(s['num_worst'])}",
           f"- injected bugs caught: **{s['caught']}/{s['bugs']}**", "",
           "## Comparison with PyTorch", "", "| component | forward | worst gradient | status |", "|---|---|---|---|"]
    for t in TORCH:
        out.append(f"| {t['component']} | {_fmt(t['forward'])} | {_fmt(t['worst_grad'])} | {'PASS' if t['ok'] else 'FAIL'} |")
    out += ["", "## Starting loss of a fresh 10-class network (ln 10 = 2.3026)", "", "| network | mean loss | spread |", "|---|---|---|"]
    for r in INIT_LOSS:
        out.append(f"| {r['label']} | {r['mean']:.4f} | {r['spread']:.3f} |")
    out += ["", "## Bug injection", "", "| bug | caught | noticed by |", "|---|---|---|"]
    for m in MUTATIONS:
        out.append(f"| {m['bug']} | {'yes' if m['caught'] else 'NO'} | {m['how']} |")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
