import os
from tests import report as day3_report            # the Day 3 report (gradient checks of the Tensor engine)
from tests import layer_report as day4_report      # the Day 4 report (layers and losses)


def pytest_runtest_logreport(report):
    # pytest calls this for every test phase; I only keep the "call" phase = the test itself.
    if report.when == "call":
        item = (report.nodeid, report.outcome, report.duration)
        day3_report.OUTCOMES.append(item)
        if "test_layers.py" in report.nodeid:
            day4_report.OUTCOMES.append(item)


def pytest_terminal_summary(terminalreporter):
    tr = terminalreporter
    # Day 3 report: only when its tests ran.
    if day3_report.CHECKS:
        tr.write_line("")
        for line in day3_report.terminal_lines():
            tr.write_line(line)
        if day3_report.MUTATIONS:        # only after a full run, so a partial run never overwrites the file
            os.makedirs("notebooks_or_plots", exist_ok=True)
            day3_report.write_markdown("notebooks_or_plots/day3_gradient_report.md")
            tr.write_line("report saved to notebooks_or_plots/day3_gradient_report.md")
    # Day 4 report: only when the layer tests ran.
    if day4_report.NUMERIC:
        tr.write_line("")
        for line in day4_report.terminal_lines():
            tr.write_line(line)
        if day4_report.MUTATIONS:
            os.makedirs("notebooks_or_plots", exist_ok=True)
            day4_report.write_markdown("notebooks_or_plots/day4_layers_report.md")
            tr.write_line("report saved to notebooks_or_plots/day4_layers_report.md")