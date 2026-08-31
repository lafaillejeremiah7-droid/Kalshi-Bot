"""XAU/USD multi-agent signal company."""

# Dashboard telemetry is fail-open and observes the existing runtime logger. It
# never participates in strategy selection or signal authorization.
from . import dashboard as _dashboard
from .dashboard_invention import install_invention_dashboard

install_invention_dashboard()
_dashboard.install_dashboard_logging()
