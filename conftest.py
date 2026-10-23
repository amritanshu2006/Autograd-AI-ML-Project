import os
from tests import report as day3_report


def pytest_runtest_logreport(report):
    if report.when == "call":
        day3_report.OUTCOMES.append((report.nodeid, report.outcome, report.duration))


def pytest_terminal_summary(terminalreporter):
    if not day3_report.CHECKS:
        return
    tr = terminalreporter
    tr.write_line("")
    for line in day3_report.terminal_lines():
        tr.write_line(line)
    # only write the file after a full run, so partial runs don't overwrite it
    if day3_report.CHECKS and day3_report.MUTATIONS:
        os.makedirs("notebooks_or_plots", exist_ok=True)
        day3_report.write_markdown("notebooks_or_plots/day3_gradient_report.md")
        tr.write_line("report saved to notebooks_or_plots/day3_gradient_report.md")