"""Collects results during the run and prints/writes the analysis report."""
import math
from collections import defaultdict

CHECKS = []      # one dict per gradient check
MUTATIONS = []   # one dict per injected bug
OUTCOMES = []    # (nodeid, outcome, seconds) per test, filled by conftest


def record_check(category, name, shapes, stats, worst, ok, secs):
    CHECKS.append(dict(category=category, name=name, shapes=shapes,
                       stats=stats, worst=worst, ok=ok, secs=secs))


def record_mutation(bug, description, caught, worst):
    MUTATIONS.append(dict(bug=bug, description=description,
                          caught=caught, worst=worst))


def margin(worst, tol=1e-6):
    """How many times smaller than the tolerance the error is."""
    return tol / worst if worst > 0 else float("inf")


def _fmt(x):
    if x == float("inf"):
        return "inf"
    return f"{x:.1e}"


def _digits(worst):
    return max(0.0, -math.log10(worst)) if worst > 0 else 16.0


def _cut(s, n):
    return s if len(s) <= n else s[: n - 1] + "…"


def _by_category():
    d = defaultdict(list)
    for c in CHECKS:
        d[c["category"]].append(c)
    return d


def _by_test_function():
    d = defaultdict(lambda: [0, 0, 0.0])
    for nodeid, outcome, secs in OUTCOMES:
        fn = nodeid.split("::")[1].split("[")[0] if "::" in nodeid else nodeid
        d[fn][0] += 1
        d[fn][1] += outcome == "passed"
        d[fn][2] += secs
    return d


def _median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def _summary():
    worsts = [c["worst"] for c in CHECKS]
    return dict(
        n=len(CHECKS), passed=sum(c["ok"] for c in CHECKS),
        worst=max(worsts), median=_median(worsts),
        digits=_digits(max(worsts)),
        caught=sum(m["caught"] for m in MUTATIONS), bugs=len(MUTATIONS),
        outcomes_pass=sum(o[1] == "passed" for o in OUTCOMES),
        outcomes=len(OUTCOMES),
    )


def terminal_lines():
    L = []
    s = _summary()
    bar = "=" * 100
    L += [bar, " TENSOR ENGINE - GRADIENT ANALYSIS REPORT", bar, ""]

    L += [" A. PER-OPERATION GRADIENT CHECKS (autograd vs numerical)",
          f"    {'category':<13}{'check':<30}{'input shapes':<22}{'scaled err':>11}{'margin':>10}  status",
          "    " + "-" * 94]
    for c in CHECKS:
        m = margin(c["worst"])
        ms = "inf" if m == float("inf") else f"{m:.0e}x"
        L.append(f"    {c['category']:<13}{_cut(c['name'], 29):<30}{_cut(c['shapes'], 21):<22}"
                 f"{_fmt(c['worst']):>11}{ms:>10}  {'PASS' if c['ok'] else 'FAIL  <<<<<'}")
    L.append("")

    L += [" B. BY CATEGORY",
          f"    {'category':<14}{'checks':>7}{'passed':>8}{'worst err':>12}{'median err':>12}",
          "    " + "-" * 53]
    for cat, cs in _by_category().items():
        L.append(f"    {cat:<14}{len(cs):>7}{sum(c['ok'] for c in cs):>8}"
                 f"{_fmt(max(c['worst'] for c in cs)):>12}{_fmt(_median([c['worst'] for c in cs])):>12}")
    L.append("")

    worst_c = max(CHECKS, key=lambda c: c["worst"])
    L += [" C. OVERALL ACCURACY",
          f"    checks passed          : {s['passed']}/{s['n']}",
          f"    worst scaled error     : {_fmt(s['worst'])}   ({worst_c['name']} {worst_c['shapes']})",
          f"    median scaled error    : {_fmt(s['median'])}",
          f"    smallest safety margin : {margin(s['worst']):.0e}x below the tolerance",
          f"    correct digits (worst) : about {s['digits']:.0f}",
          ""]

    if MUTATIONS:
        L += [" D. BUG-INJECTION EFFECTIVENESS (do the tests catch deliberately broken code?)",
              f"    {'injected bug':<28}{'what was broken':<46}{'caught':>8}{'worst err':>12}",
              "    " + "-" * 94]
        for m in MUTATIONS:
            L.append(f"    {_cut(m['bug'], 27):<28}{_cut(m['description'], 45):<46}"
                     f"{'YES' if m['caught'] else 'NO  <<<<<':>8}{_fmt(m['worst']):>12}")
        L += [f"    caught {s['caught']}/{s['bugs']} injected bugs", ""]

    L += [" E. BY TEST FUNCTION",
          f"    {'test function':<52}{'tests':>6}{'passed':>8}{'seconds':>9}",
          "    " + "-" * 75]
    rows = sorted(_by_test_function().items(), key=lambda kv: -kv[1][2])
    for fn, (n, p, secs) in sorted(_by_test_function().items()):
        L.append(f"    {_cut(fn, 51):<52}{n:>6}{p:>8}{secs:>9.2f}")
    L.append("    slowest: " + ", ".join(f"{k} ({v[2]:.2f}s)" for k, v in rows[:3]))
    L.append("")

    all_ok = (s["passed"] == s["n"] and s["outcomes_pass"] == s["outcomes"]
              and (not MUTATIONS or s["caught"] == s["bugs"]))
    L += [bar,
          f" VERDICT: {'ALL GOOD' if all_ok else 'PROBLEMS FOUND'} - "
          f"{s['outcomes_pass']}/{s['outcomes']} tests passed, "
          f"{s['passed']}/{s['n']} gradient checks within tolerance, "
          f"{s['caught']}/{s['bugs']} injected bugs caught.",
          bar]
    return L


def write_markdown(path):
    s = _summary()
    out = ["# Day 3 - Tensor gradient report", "",
           f"- gradient checks passed: **{s['passed']}/{s['n']}**",
           f"- worst scaled error: **{_fmt(s['worst'])}** (about {s['digits']:.0f} correct digits)",
           f"- injected bugs caught: **{s['caught']}/{s['bugs']}**", "",
           "## Per-operation checks", "",
           "| category | check | shapes | scaled error | status |", "|---|---|---|---|---|"]
    for c in CHECKS:
        out.append(f"| {c['category']} | {c['name']} | {c['shapes']} | {_fmt(c['worst'])} | {'PASS' if c['ok'] else 'FAIL'} |")
    out += ["", "## Bug injection", "", "| bug | description | caught |", "|---|---|---|"]
    for m in MUTATIONS:
        out.append(f"| {m['bug']} | {m['description']} | {'yes' if m['caught'] else 'NO'} |")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
