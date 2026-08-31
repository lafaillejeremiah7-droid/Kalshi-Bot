"""XAU/USD multi-agent signal company."""

# Dashboard telemetry is fail-open and observes the existing runtime logger. It
# never participates in strategy selection or signal authorization.
from .dashboard import install_dashboard_logging

install_dashboard_logging()
