"""
Hypothesis generator that creates 100+ market behavior hypotheses.

This module programmatically constructs Hypothesis objects across multiple
categories, each with a signal function, economic rationale, and clear
documentation of data limitations.

Data Limitations:
    All hypotheses operate on OHLCV data only. Signals that attempt to proxy
    order flow, microstructure, or intraday behavior are limited by the
    granularity of daily bar data and exchange-reported volume.
"""

from __future__ import annotations

from quant_research.hypotheses.catalog import Hypothesis, HypothesisCategory
from quant_research.hypotheses import signals


ORDER_FLOW_LIMITATION = (
    "Inferred from OHLCV price-volume relationships. This is NOT true order "
    "flow data (Level II, Time & Sales, order book). These signals proxy order "
    "flow behavior but cannot capture bid-ask dynamics, queue position, or "
    "hidden liquidity."
)


class HypothesisGenerator:
    """Generates a catalog of 100+ testable market behavior hypotheses.

    Each hypothesis includes:
    - A signal computation function
    - Economic rationale explaining why the edge might persist
    - Clear documentation of data limitations

    Examples
    --------
    >>> gen = HypothesisGenerator()
    >>> hypotheses = gen.generate_all()
    >>> len(hypotheses) >= 100
    True
    """

    def generate_all(self) -> list[Hypothesis]:
        """Generate all hypotheses across all categories.

        Returns
        -------
        list[Hypothesis]
            List of 100+ Hypothesis objects organized by category.
        """
        hypotheses: list[Hypothesis] = []
        hypotheses.extend(self._momentum_hypotheses())
        hypotheses.extend(self._mean_reversion_hypotheses())
        hypotheses.extend(self._volatility_hypotheses())
        hypotheses.extend(self._gap_hypotheses())
        hypotheses.extend(self._session_effects_hypotheses())
        hypotheses.extend(self._order_flow_hypotheses())
        hypotheses.extend(self._regime_hypotheses())
        hypotheses.extend(self._microstructure_hypotheses())
        return hypotheses


    def _momentum_hypotheses(self) -> list[Hypothesis]:
        """Generate momentum category hypotheses."""
        return [
            Hypothesis(
                id="MOM_001",
                category=HypothesisCategory.MOMENTUM,
                name="1-Day Momentum",
                description="Yesterday positive return predicts today positive.",
                economic_rationale="Short-term autocorrelation from gradual information diffusion and herding behavior among retail traders.",
                data_requirements=["Close"],
                signal_function=signals.momentum_1d,
                expected_direction=1,
                data_limitations="Daily close-to-close returns only; does not capture intraday momentum patterns.",
            ),
            Hypothesis(
                id="MOM_002",
                category=HypothesisCategory.MOMENTUM,
                name="5-Day Momentum",
                description="5-day positive return predicts continued upside.",
                economic_rationale="Weekly momentum reflects institutional order flow that takes multiple days to execute, creating persistent price pressure.",
                data_requirements=["Close"],
                signal_function=signals.momentum_5d,
                expected_direction=1,
                data_limitations="Daily data only; cannot distinguish between continuous buying and gap-driven returns.",
            ),
            Hypothesis(
                id="MOM_003",
                category=HypothesisCategory.MOMENTUM,
                name="10-Day Momentum",
                description="10-day return predicts next-period direction.",
                economic_rationale="Two-week momentum captures trend-following behavior and delayed reaction to earnings/news.",
                data_requirements=["Close"],
                signal_function=signals.momentum_10d,
                expected_direction=1,
                data_limitations="Daily OHLCV only.",
            ),
            Hypothesis(
                id="MOM_004",
                category=HypothesisCategory.MOMENTUM,
                name="20-Day Momentum",
                description="Monthly return continuation.",
                economic_rationale="Monthly momentum persists due to fund flows, performance chasing, and anchoring bias among investors.",
                data_requirements=["Close"],
                signal_function=signals.momentum_20d,
                expected_direction=1,
                data_limitations="Daily OHLCV only.",
            ),
            Hypothesis(
                id="MOM_005",
                category=HypothesisCategory.MOMENTUM,
                name="60-Day Momentum",
                description="Quarterly momentum continuation.",
                economic_rationale="Quarterly momentum driven by earnings revisions cycle, analyst recommendation changes, and institutional rebalancing schedules.",
                data_requirements=["Close"],
                signal_function=signals.momentum_60d,
                expected_direction=1,
                data_limitations="Daily OHLCV only.",
            ),
            Hypothesis(
                id="MOM_006",
                category=HypothesisCategory.MOMENTUM,
                name="120-Day Momentum",
                description="6-month momentum continuation.",
                economic_rationale="Medium-term momentum from behavioral underreaction to fundamental changes and slow-moving institutional allocation shifts.",
                data_requirements=["Close"],
                signal_function=signals.momentum_120d,
                expected_direction=1,
                data_limitations="Daily OHLCV only.",
            ),
            Hypothesis(
                id="MOM_007",
                category=HypothesisCategory.MOMENTUM,
                name="252-Day Momentum",
                description="Annual momentum (12-month return).",
                economic_rationale="Classic Jegadeesh-Titman momentum factor: winners continue winning due to slow information diffusion and disposition effect.",
                data_requirements=["Close"],
                signal_function=signals.momentum_252d,
                expected_direction=1,
                data_limitations="Requires 252 days of history; single-asset momentum lacks cross-sectional component.",
            ),
            Hypothesis(
                id="MOM_008",
                category=HypothesisCategory.MOMENTUM,
                name="Dual Momentum",
                description="Combine absolute and relative momentum for stronger signal.",
                economic_rationale="Dual momentum filters out false signals by requiring both absolute return positivity and short-term confirmation, reducing whipsaw.",
                data_requirements=["Close"],
                signal_function=signals.dual_momentum,
                expected_direction=1,
                data_limitations="Daily OHLCV only; no cross-asset relative momentum available.",
            ),
            Hypothesis(
                id="MOM_009",
                category=HypothesisCategory.MOMENTUM,
                name="Vol-Scaled Momentum",
                description="Momentum signal scaled by inverse volatility.",
                economic_rationale="Risk-parity approach to momentum: scale position by inverse vol to normalize risk contribution, historically improving risk-adjusted returns.",
                data_requirements=["Close"],
                signal_function=signals.momentum_vol_scaled,
                expected_direction=1,
                data_limitations="Volatility estimated from daily returns only.",
            ),
            Hypothesis(
                id="MOM_010",
                category=HypothesisCategory.MOMENTUM,
                name="Rate of Change",
                description="10-day rate of change as momentum indicator.",
                economic_rationale="ROC captures acceleration in price, identifying strengthening or weakening trends before moving averages.",
                data_requirements=["Close"],
                signal_function=signals.rate_of_change,
                expected_direction=1,
                data_limitations="Daily OHLCV only.",
            ),
            Hypothesis(
                id="MOM_011",
                category=HypothesisCategory.MOMENTUM,
                name="Momentum Reversal After Extremes",
                description="Extreme momentum (>2 std) tends to reverse.",
                economic_rationale="Extreme short-term moves overshoot fair value due to panic/euphoria, triggering mean reversion as rational actors provide liquidity.",
                data_requirements=["Close"],
                signal_function=signals.momentum_reversal_extreme,
                expected_direction=-1,
                data_limitations="Daily OHLCV only.",
            ),
            Hypothesis(
                id="MOM_012",
                category=HypothesisCategory.MOMENTUM,
                name="Momentum Acceleration",
                description="Second derivative of momentum (momentum of momentum).",
                economic_rationale="Acceleration signals trend initiation or exhaustion; increasing momentum suggests new information still being priced.",
                data_requirements=["Close"],
                signal_function=signals.momentum_acceleration,
                expected_direction=1,
                data_limitations="Daily OHLCV only.",
            ),
            Hypothesis(
                id="MOM_013",
                category=HypothesisCategory.MOMENTUM,
                name="Trend Following MA Cross",
                description="Price vs 50-day MA crossover.",
                economic_rationale="MA crossovers capture regime changes; price above long-term MA indicates positive trend supported by institutional positioning.",
                data_requirements=["Close"],
                signal_function=signals.trend_following_ma_cross,
                expected_direction=1,
                data_limitations="Daily OHLCV only; lagging indicator by nature.",
            ),
            Hypothesis(
                id="MOM_014",
                category=HypothesisCategory.MOMENTUM,
                name="Dual MA Crossover",
                description="20-day MA vs 50-day MA crossover.",
                economic_rationale="Dual MA reduces whipsaw vs single MA; captures intermediate-term trend changes used by systematic CTA strategies.",
                data_requirements=["Close"],
                signal_function=signals.dual_ma_crossover,
                expected_direction=1,
                data_limitations="Daily OHLCV only; lagging indicator.",
            ),
            Hypothesis(
                id="MOM_015",
                category=HypothesisCategory.MOMENTUM,
                name="Momentum Crash Vol Filter",
                description="Long momentum in low vol, reduce in high vol (crash protection).",
                economic_rationale="Momentum strategies crash in high-vol regime shifts; vol filter preserves gains by reducing exposure when crash risk is elevated.",
                data_requirements=["Close"],
                signal_function=signals.momentum_crash_vol_filter,
                expected_direction=1,
                data_limitations="Volatility estimated from daily returns only; cannot detect intraday vol spikes.",
            ),
        ]


    def _mean_reversion_hypotheses(self) -> list[Hypothesis]:
        """Generate mean reversion category hypotheses."""
        return [
            Hypothesis(
                id="MR_001",
                category=HypothesisCategory.MEAN_REVERSION,
                name="RSI Mean Reversion",
                description="Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought).",
                economic_rationale="RSI extremes indicate exhaustion of directional pressure; liquidity providers step in at extremes, causing price to revert.",
                data_requirements=["Close"],
                signal_function=signals.rsi_mean_reversion,
                expected_direction=0,
                data_limitations="RSI computed from daily closes only; intraday RSI may differ significantly.",
            ),
            Hypothesis(
                id="MR_002",
                category=HypothesisCategory.MEAN_REVERSION,
                name="Bollinger Band Reversion",
                description="Buy at lower band, sell at upper band.",
                economic_rationale="Bollinger Bands capture 2-std moves; prices beyond bands are statistically extreme and tend to revert as mean-reversion traders provide liquidity.",
                data_requirements=["Close"],
                signal_function=signals.bollinger_mean_reversion,
                expected_direction=0,
                data_limitations="Daily OHLCV only; bands based on 20-day statistics.",
            ),
            Hypothesis(
                id="MR_003",
                category=HypothesisCategory.MEAN_REVERSION,
                name="Gap Fill Reversion",
                description="Overnight gaps tend to fill during the session.",
                economic_rationale="Overnight gaps often reflect low-liquidity price discovery; regular session liquidity corrects prices toward previous close.",
                data_requirements=["Open", "Close"],
                signal_function=signals.gap_fill_reversion,
                expected_direction=0,
                data_limitations="Gap measured from daily Open vs prior Close only; cannot observe actual fill timing.",
            ),
            Hypothesis(
                id="MR_004",
                category=HypothesisCategory.MEAN_REVERSION,
                name="Consecutive Move Reversion",
                description="3+ consecutive same-direction days tend to reverse.",
                economic_rationale="Extended unidirectional moves exhaust directional traders; contrarian capital and profit-taking create reversal pressure.",
                data_requirements=["Close"],
                signal_function=signals.consecutive_move_reversion,
                expected_direction=0,
                data_limitations="Daily OHLCV only.",
            ),
            Hypothesis(
                id="MR_005",
                category=HypothesisCategory.MEAN_REVERSION,
                name="Volatility Mean Reversion",
                description="High volatility reverts to mean over time.",
                economic_rationale="Volatility clustering eventually dissipates as uncertainty resolves; vol sellers profit from mean reversion of implied/realized vol.",
                data_requirements=["Close"],
                signal_function=signals.volatility_mean_reversion,
                expected_direction=1,
                data_limitations="Realized vol from daily returns only.",
            ),
            Hypothesis(
                id="MR_006",
                category=HypothesisCategory.MEAN_REVERSION,
                name="Volume-Weighted Reversion",
                description="High-volume extreme moves revert faster.",
                economic_rationale="High volume at extremes signals capitulation/euphoria; smart money provides liquidity at these points.",
                data_requirements=["Close", "Volume"],
                signal_function=signals.volume_weighted_reversion,
                expected_direction=0,
                data_limitations="Volume is exchange-reported daily total only.",
            ),
            Hypothesis(
                id="MR_007",
                category=HypothesisCategory.MEAN_REVERSION,
                name="Distance from MA Reversion",
                description="Price far from 50-day MA reverts toward it.",
                economic_rationale="Moving averages act as dynamic equilibrium; institutional rebalancing and value buyers create gravitational pull toward MA.",
                data_requirements=["Close"],
                signal_function=signals.distance_from_ma_reversion,
                expected_direction=0,
                data_limitations="Daily OHLCV only.",
            ),
            Hypothesis(
                id="MR_008",
                category=HypothesisCategory.MEAN_REVERSION,
                name="Z-Score Returns Reversion",
                description="Extreme z-score of 5-day returns reverts.",
                economic_rationale="Statistical mean reversion: extreme z-scores are by definition rare and tend to normalize as the generating process is stationary.",
                data_requirements=["Close"],
                signal_function=signals.zscore_returns_reversion,
                expected_direction=0,
                data_limitations="Daily OHLCV only; assumes stationarity of return distribution.",
            ),
            Hypothesis(
                id="MR_009",
                category=HypothesisCategory.MEAN_REVERSION,
                name="Hurst Exponent Regime",
                description="Low Hurst exponent (<0.5) indicates mean-reverting regime.",
                economic_rationale="Hurst exponent measures persistence; sub-0.5 values indicate anti-persistent (mean-reverting) dynamics where contrarian strategies profit.",
                data_requirements=["Close"],
                signal_function=signals.hurst_regime_reversion,
                expected_direction=0,
                data_limitations="Hurst estimation is noisy with limited daily data; proxy method used.",
            ),
            Hypothesis(
                id="MR_010",
                category=HypothesisCategory.MEAN_REVERSION,
                name="Put-Call Proxy Reversion",
                description="High fear proxy (vol spike) is contrarian bullish.",
                economic_rationale="Extreme fear (proxied by short-term vol spike) historically marks bottoms as panic selling exhausts; the market climbs a wall of worry.",
                data_requirements=["Close"],
                signal_function=signals.put_call_proxy_reversion,
                expected_direction=1,
                data_limitations="No actual put-call ratio available; using vol ratio as fear proxy.",
            ),
            Hypothesis(
                id="MR_011",
                category=HypothesisCategory.MEAN_REVERSION,
                name="RSI Divergence Reversion",
                description="Price new low but RSI higher signals bullish divergence.",
                economic_rationale="RSI divergence shows weakening selling pressure despite new price lows; early sign of trend exhaustion and reversal.",
                data_requirements=["Close"],
                signal_function=signals.rsi_divergence_reversion,
                expected_direction=0,
                data_limitations="Daily RSI only; divergences can persist longer than expected.",
            ),
            Hypothesis(
                id="MR_012",
                category=HypothesisCategory.MEAN_REVERSION,
                name="Overextension Reversion",
                description="Price far from 200-day MA reverts.",
                economic_rationale="Extreme deviation from 200-day MA is unsustainable; fundamental value acts as anchor and institutions rebalance at extremes.",
                data_requirements=["Close"],
                signal_function=signals.overextension_reversion,
                expected_direction=0,
                data_limitations="Daily OHLCV only; 200-day MA requires significant history.",
            ),
            Hypothesis(
                id="MR_013",
                category=HypothesisCategory.MEAN_REVERSION,
                name="MACD Reversion",
                description="MACD histogram at extremes reverts.",
                economic_rationale="Extreme MACD histogram readings indicate overextended short-term vs long-term trend divergence that normalizes.",
                data_requirements=["Close"],
                signal_function=signals.macd_reversion,
                expected_direction=0,
                data_limitations="Daily OHLCV only.",
            ),
            Hypothesis(
                id="MR_014",
                category=HypothesisCategory.MEAN_REVERSION,
                name="Keltner Channel Reversion",
                description="Price outside Keltner channel reverts to mean.",
                economic_rationale="Keltner channels use ATR-based bands; moves beyond 2x ATR from EMA are statistically extreme and attract mean-reversion capital.",
                data_requirements=["High", "Low", "Close"],
                signal_function=signals.keltner_reversion,
                expected_direction=0,
                data_limitations="Daily OHLCV only.",
            ),
        ]


    def _volatility_hypotheses(self) -> list[Hypothesis]:
        """Generate volatility category hypotheses."""
        return [
            Hypothesis(id="VOL_001", category=HypothesisCategory.VOLATILITY, name="Vol Expansion Breakout", description="Trade in direction of volatility expansion.", economic_rationale="Vol expansion after compression signals new information entering the market; initial direction tends to persist.", data_requirements=["Close"], signal_function=signals.vol_expansion_breakout, expected_direction=1, data_limitations="Daily OHLCV only; cannot detect intraday vol shifts."),
            Hypothesis(id="VOL_002", category=HypothesisCategory.VOLATILITY, name="Vol Compression Squeeze", description="Low volatility precedes directional breakout.", economic_rationale="Bollinger Band squeeze reflects equilibrium between buyers/sellers; breakout from compression tends to be explosive as stops trigger.", data_requirements=["Close"], signal_function=signals.vol_compression_squeeze, expected_direction=1, data_limitations="Daily OHLCV only."),
            Hypothesis(id="VOL_003", category=HypothesisCategory.VOLATILITY, name="Vol Clustering", description="High vol follows high vol; trade in current direction.", economic_rationale="Volatility clusters due to information cascades and feedback loops between vol and positioning (e.g., delta hedging).", data_requirements=["Close"], signal_function=signals.vol_clustering, expected_direction=1, data_limitations="Daily OHLCV only."),
            Hypothesis(id="VOL_004", category=HypothesisCategory.VOLATILITY, name="VIX Analog", description="Extreme realized vol as fear proxy is contrarian bullish.", economic_rationale="Extreme realized vol corresponds to panic; historically, buying during high-vol fear periods yields above-average returns.", data_requirements=["Close"], signal_function=signals.vix_analog_signal, expected_direction=1, data_limitations="No actual VIX available; using realized vol percentile as proxy."),
            Hypothesis(id="VOL_005", category=HypothesisCategory.VOLATILITY, name="Vol Term Structure Proxy", description="Short-term vs long-term vol ratio signals regime.", economic_rationale="Inverted vol term structure (short > long) indicates market stress; historically, stress episodes resolve with recovery.", data_requirements=["Close"], signal_function=signals.vol_term_structure_proxy, expected_direction=1, data_limitations="No actual VIX term structure; using realized vol at different windows."),
            Hypothesis(id="VOL_006", category=HypothesisCategory.VOLATILITY, name="Vol Regime Persistence", description="Stay with current vol regime strategy.", economic_rationale="Vol regimes are sticky; trend-following works in high-vol and mean-reversion in low-vol due to different market microstructure dynamics.", data_requirements=["Close"], signal_function=signals.vol_regime_persistence, expected_direction=0, data_limitations="Daily OHLCV only."),
            Hypothesis(id="VOL_007", category=HypothesisCategory.VOLATILITY, name="GARCH Conditional Vol", description="GARCH-based conditional volatility for regime detection.", economic_rationale="GARCH captures time-varying vol more accurately than simple rolling windows; better regime detection improves signal quality.", data_requirements=["Close"], signal_function=signals.garch_conditional_vol, expected_direction=0, data_limitations="GARCH approximation from daily data; simplified single-asset model."),
            Hypothesis(id="VOL_008", category=HypothesisCategory.VOLATILITY, name="Vol-of-Vol", description="High vol-of-vol indicates regime uncertainty.", economic_rationale="Vol-of-vol spikes precede regime changes; elevated uncertainty in vol itself signals transition periods with trading opportunities.", data_requirements=["Close"], signal_function=signals.vol_of_vol_signal, expected_direction=1, data_limitations="Daily OHLCV only."),
            Hypothesis(id="VOL_009", category=HypothesisCategory.VOLATILITY, name="Range Expansion", description="Unusually wide daily range signals continuation.", economic_rationale="Wide-range days indicate strong conviction and new information; the initial direction of wide-range days tends to persist.", data_requirements=["High", "Low", "Close"], signal_function=signals.range_expansion_signal, expected_direction=1, data_limitations="Daily OHLCV only."),
            Hypothesis(id="VOL_010", category=HypothesisCategory.VOLATILITY, name="Vol Smile Proxy", description="Asymmetric wick patterns suggest directional vol bias.", economic_rationale="Persistent wick asymmetry reveals directional pressure not captured by close-to-close returns; upper wicks signal distribution.", data_requirements=["Open", "High", "Low", "Close"], signal_function=signals.vol_smile_proxy, expected_direction=0, data_limitations="Proxy for vol smile from OHLCV only; no options data."),
            Hypothesis(id="VOL_011", category=HypothesisCategory.VOLATILITY, name="ATR Breakout", description="Price beyond 2x ATR from previous close.", economic_rationale="Moves exceeding 2x ATR represent significant dislocations; these often continue as stop-losses cascade and momentum traders enter.", data_requirements=["High", "Low", "Close"], signal_function=signals.atr_breakout, expected_direction=1, data_limitations="Daily OHLCV only."),
            Hypothesis(id="VOL_012", category=HypothesisCategory.VOLATILITY, name="Realized vs Implied Proxy", description="Current vol vs historical average divergence.", economic_rationale="When current vol is below historical average, it is cheap; vol tends to expand from compressed levels creating directional opportunity.", data_requirements=["Close"], signal_function=signals.realized_vs_implied_proxy, expected_direction=0, data_limitations="No actual implied vol; using realized vol ratio as proxy."),
            Hypothesis(id="VOL_013", category=HypothesisCategory.VOLATILITY, name="Vol Mean Reversion 20/60", description="20-day vs 60-day vol ratio mean reverts.", economic_rationale="Short-term vol deviations from longer-term levels are transient; vol selling strategies exploit this mean reversion tendency.", data_requirements=["Close"], signal_function=signals.vol_mean_reversion_20_60, expected_direction=0, data_limitations="Daily OHLCV only."),
            Hypothesis(id="VOL_014", category=HypothesisCategory.VOLATILITY, name="Parkinson vs Close Vol", description="Parkinson (range-based) vs close-to-close vol divergence.", economic_rationale="When Parkinson vol exceeds close vol, intraday moves are being masked by close prices; this hidden volatility often precedes directional moves.", data_requirements=["High", "Low", "Close"], signal_function=signals.parkinson_vs_close_vol, expected_direction=0, data_limitations="Daily OHLCV only; Parkinson estimator assumes continuous trading."),
        ]

    def _gap_hypotheses(self) -> list[Hypothesis]:
        """Generate gap category hypotheses."""
        return [
            Hypothesis(id="GAP_001", category=HypothesisCategory.GAPS, name="Large Gap Reversion", description="Large overnight gaps (>1.5 std) tend to fill.", economic_rationale="Large gaps reflect overnight news overreaction in thin liquidity; regular session participants correct the overreaction.", data_requirements=["Open", "Close"], signal_function=signals.gap_reversion_large, expected_direction=0, data_limitations="Gap from daily Open vs prior Close; cannot observe extended-hours trading."),
            Hypothesis(id="GAP_002", category=HypothesisCategory.GAPS, name="Gap Fill by Size", description="Smaller gaps have higher fill probability.", economic_rationale="Small gaps often reflect noise rather than information; regular session price discovery fills these non-informational gaps.", data_requirements=["Open", "Close"], signal_function=signals.gap_fill_probability_size, expected_direction=0, data_limitations="Daily OHLCV only; fill timing cannot be observed."),
            Hypothesis(id="GAP_003", category=HypothesisCategory.GAPS, name="Gap Fill Direction Bias", description="Down gaps fill more reliably than up gaps.", economic_rationale="Markets have long-term upward bias; down gaps against the trend fill more reliably as buy-the-dip behavior dominates.", data_requirements=["Open", "Close"], signal_function=signals.gap_fill_by_direction, expected_direction=1, data_limitations="Daily OHLCV only."),
            Hypothesis(id="GAP_004", category=HypothesisCategory.GAPS, name="Gap and Go", description="Trade in gap direction when momentum confirms.", economic_rationale="When a gap is confirmed by same-direction intraday move, it signals genuine new information rather than noise.", data_requirements=["Open", "Close"], signal_function=signals.gap_and_go, expected_direction=1, data_limitations="Daily OHLCV only; intraday confirmation proxied by close vs open."),
            Hypothesis(id="GAP_005", category=HypothesisCategory.GAPS, name="Unfilled Gap Support", description="Unfilled gaps act as support/resistance.", economic_rationale="Unfilled gaps represent price levels where consensus shifted; these levels act as psychological support/resistance.", data_requirements=["Open", "Close"], signal_function=signals.unfilled_gap_support, expected_direction=1, data_limitations="Daily OHLCV only; cannot track precise intraday gap fill."),
            Hypothesis(id="GAP_006", category=HypothesisCategory.GAPS, name="Monday Gap Effect", description="Weekend gaps tend to fill during Monday session.", economic_rationale="Weekend news creates gaps in thin Sunday futures; Monday regular session corrects the overreaction.", data_requirements=["Open", "Close"], signal_function=signals.monday_gap_effect, expected_direction=0, data_limitations="Daily OHLCV only."),
            Hypothesis(id="GAP_007", category=HypothesisCategory.GAPS, name="Holiday Gap Effect", description="Gaps after multi-day breaks tend to fill.", economic_rationale="Extended closures create uncertainty premium in opening prices; regular session price discovery resolves the premium.", data_requirements=["Open", "Close"], signal_function=signals.holiday_gap_effect, expected_direction=0, data_limitations="Daily OHLCV only."),
            Hypothesis(id="GAP_008", category=HypothesisCategory.GAPS, name="Gap Volume Confirmation", description="Gaps with high volume persist; low volume gaps fill.", economic_rationale="High volume confirms institutional participation and genuine information; low volume gaps are noise-driven.", data_requirements=["Open", "Close", "Volume"], signal_function=signals.gap_volume_confirmation, expected_direction=0, data_limitations="Volume is daily total only; cannot observe volume during gap formation."),
            Hypothesis(id="GAP_009", category=HypothesisCategory.GAPS, name="Gap Streak", description="Consecutive same-direction gaps signal exhaustion.", economic_rationale="Multiple gaps in one direction exhaust short-term directional capital; contrarian pressure builds.", data_requirements=["Open", "Close"], signal_function=signals.gap_streak, expected_direction=0, data_limitations="Daily OHLCV only."),
            Hypothesis(id="GAP_010", category=HypothesisCategory.GAPS, name="Gap Size Relative to ATR", description="Gap size relative to ATR determines fill vs continuation.", economic_rationale="Gaps exceeding 1 ATR represent significant moves relative to recent volatility and tend to continue; sub-ATR gaps fill.", data_requirements=["Open", "High", "Low", "Close"], signal_function=signals.gap_size_relative, expected_direction=0, data_limitations="Daily OHLCV only."),
        ]


    def _session_effects_hypotheses(self) -> list[Hypothesis]:
        """Generate session effects category hypotheses."""
        return [
            Hypothesis(id="SE_001", category=HypothesisCategory.SESSION_EFFECTS, name="Monday Weakness", description="Mondays historically show negative bias.", economic_rationale="Weekend uncertainty and margin calls create Monday selling pressure; retail traders accumulate weekend worry.", data_requirements=["Close"], signal_function=signals.day_of_week_monday, expected_direction=-1, data_limitations="Calendar-based; effect has weakened in recent decades."),
            Hypothesis(id="SE_002", category=HypothesisCategory.SESSION_EFFECTS, name="Wednesday Reversal", description="Wednesday tends to reverse Monday-Tuesday trend.", economic_rationale="Mid-week rebalancing by short-term traders who positioned Monday/Tuesday creates reversal pressure.", data_requirements=["Close"], signal_function=signals.day_of_week_wednesday, expected_direction=0, data_limitations="Daily OHLCV only."),
            Hypothesis(id="SE_003", category=HypothesisCategory.SESSION_EFFECTS, name="Friday Effect", description="Risk reduction before weekend creates slight negative Friday bias.", economic_rationale="Traders reduce risk before weekend uncertainty; portfolio managers flatten positions ahead of non-trading period.", data_requirements=["Close"], signal_function=signals.day_of_week_friday, expected_direction=-1, data_limitations="Calendar-based; effect varies by market regime."),
            Hypothesis(id="SE_004", category=HypothesisCategory.SESSION_EFFECTS, name="January Effect", description="January historically shows above-average returns.", economic_rationale="Tax-loss selling in December creates depressed prices; January sees reinvestment of proceeds and new year allocations.", data_requirements=["Close"], signal_function=signals.month_of_year_january, expected_direction=1, data_limitations="Calendar-based; primarily a small-cap effect."),
            Hypothesis(id="SE_005", category=HypothesisCategory.SESSION_EFFECTS, name="September Weakness", description="September historically worst month for equities.", economic_rationale="Post-summer return of institutional traders, mutual fund tax-loss selling, and fiscal year-end rebalancing create selling pressure.", data_requirements=["Close"], signal_function=signals.month_of_year_september, expected_direction=-1, data_limitations="Calendar-based; historical pattern may not persist."),
            Hypothesis(id="SE_006", category=HypothesisCategory.SESSION_EFFECTS, name="Turn of Month", description="Last 3 and first 3 trading days of month tend to be strong.", economic_rationale="Pension fund contributions, salary-based 401(k) inflows, and institutional rebalancing concentrate at month boundaries.", data_requirements=["Close"], signal_function=signals.turn_of_month, expected_direction=1, data_limitations="Calendar-based."),
            Hypothesis(id="SE_007", category=HypothesisCategory.SESSION_EFFECTS, name="Post-Holiday Bullishness", description="First trading day after multi-day market closure tends to be positive.", economic_rationale="Pent-up order flow, short covering after break, and optimism bias on return from holiday create positive drift on the first session back.", data_requirements=["Close"], signal_function=signals.pre_holiday_bullishness, expected_direction=1, data_limitations="Calendar-based; detects post-holiday session via backward-looking date gap. Function retains legacy name for API compatibility."),
            Hypothesis(id="SE_008", category=HypothesisCategory.SESSION_EFFECTS, name="Options Expiration Week", description="Third week of month (opex) has mean-reversion tendency.", economic_rationale="Options market maker gamma hedging creates pin risk; prices gravitate toward max pain strikes during opex week.", data_requirements=["Close"], signal_function=signals.options_expiration_week, expected_direction=0, data_limitations="No actual options data; opex week proxied from calendar."),
            Hypothesis(id="SE_009", category=HypothesisCategory.SESSION_EFFECTS, name="Quarter-End Rebalancing", description="Last 5 days of quarter tend to be bullish (window dressing).", economic_rationale="Fund managers buy winning stocks before quarter-end reports (window dressing); pension rebalancing creates buying pressure.", data_requirements=["Close"], signal_function=signals.quarter_end_rebalancing, expected_direction=1, data_limitations="Calendar-based."),
            Hypothesis(id="SE_010", category=HypothesisCategory.SESSION_EFFECTS, name="Sell in May", description="May-October historically weaker than November-April.", economic_rationale="Lower institutional participation during summer, reduced liquidity, and historical seasonal patterns create weaker summer returns.", data_requirements=["Close"], signal_function=signals.sell_in_may, expected_direction=-1, data_limitations="Calendar-based; effect inconsistent year-to-year."),
            Hypothesis(id="SE_011", category=HypothesisCategory.SESSION_EFFECTS, name="Santa Rally", description="Last 5 days of December + first 2 of January tend to be positive.", economic_rationale="Tax-loss selling completed, holiday optimism, low volume, and new year allocation anticipation create positive drift.", data_requirements=["Close"], signal_function=signals.santa_rally, expected_direction=1, data_limitations="Calendar-based; small sample size per year."),
            Hypothesis(id="SE_012", category=HypothesisCategory.SESSION_EFFECTS, name="First Hour Proxy", description="Opening strength/weakness from overnight gap predicts session.", economic_rationale="Strong openings reflect overnight institutional order accumulation that continues into regular session.", data_requirements=["Open", "Close"], signal_function=signals.first_hour_proxy, expected_direction=1, data_limitations="Daily OHLCV only; actual first-hour data not available."),
        ]

    def _order_flow_hypotheses(self) -> list[Hypothesis]:
        """Generate order flow proxy category hypotheses."""
        return [
            Hypothesis(id="OF_001", category=HypothesisCategory.ORDER_FLOW_PROXY, name="Volume Imbalance", description="Up-volume vs down-volume imbalance from close position in range.", economic_rationale="Close position within bar range approximates buying vs selling pressure; persistent imbalance reveals directional flow.", data_requirements=["High", "Low", "Close", "Volume"], signal_function=signals.volume_imbalance, expected_direction=1, data_limitations=ORDER_FLOW_LIMITATION),
            Hypothesis(id="OF_002", category=HypothesisCategory.ORDER_FLOW_PROXY, name="OBV Divergence", description="Price makes new high but OBV does not confirm.", economic_rationale="OBV divergence signals weakening participation in the trend; smart money reducing positions before retail notices.", data_requirements=["Close", "Volume"], signal_function=signals.obv_divergence, expected_direction=0, data_limitations=ORDER_FLOW_LIMITATION),
            Hypothesis(id="OF_003", category=HypothesisCategory.ORDER_FLOW_PROXY, name="Price-Volume Confirmation", description="Price up + volume up = bullish confirmation.", economic_rationale="Rising prices on rising volume confirms broad participation and conviction; the trend is supported by real capital.", data_requirements=["Close", "Volume"], signal_function=signals.price_volume_confirmation, expected_direction=1, data_limitations=ORDER_FLOW_LIMITATION),
            Hypothesis(id="OF_004", category=HypothesisCategory.ORDER_FLOW_PROXY, name="Price-Volume Divergence", description="Price up but volume declining = weak rally.", economic_rationale="Rising prices on declining volume reveals lack of conviction; the move is driven by low participation and likely to fail.", data_requirements=["Close", "Volume"], signal_function=signals.price_volume_divergence, expected_direction=-1, data_limitations=ORDER_FLOW_LIMITATION),
            Hypothesis(id="OF_005", category=HypothesisCategory.ORDER_FLOW_PROXY, name="Volume at Extremes", description="High volume at price extremes signals reversal.", economic_rationale="Climax volume at highs/lows indicates capitulation; all weak hands forced out, leaving only contrarian buyers/sellers.", data_requirements=["Close", "Volume"], signal_function=signals.volume_at_extremes, expected_direction=0, data_limitations=ORDER_FLOW_LIMITATION),
            Hypothesis(id="OF_006", category=HypothesisCategory.ORDER_FLOW_PROXY, name="Buying Pressure Proxy", description="(Close-Low)/(High-Low) * Volume estimates buying pressure.", economic_rationale="Close near the high of the bar indicates buyers controlled the session; sustained buying pressure drives price higher.", data_requirements=["High", "Low", "Close", "Volume"], signal_function=signals.buying_pressure_proxy, expected_direction=1, data_limitations=ORDER_FLOW_LIMITATION),
            Hypothesis(id="OF_007", category=HypothesisCategory.ORDER_FLOW_PROXY, name="Selling Pressure Proxy", description="(High-Close)/(High-Low) * Volume estimates selling pressure.", economic_rationale="Close near the low indicates sellers dominated; persistent selling pressure precedes further downside.", data_requirements=["High", "Low", "Close", "Volume"], signal_function=signals.selling_pressure_proxy, expected_direction=-1, data_limitations=ORDER_FLOW_LIMITATION),
            Hypothesis(id="OF_008", category=HypothesisCategory.ORDER_FLOW_PROXY, name="Chaikin Money Flow", description="20-period Chaikin Money Flow indicator.", economic_rationale="CMF measures accumulation/distribution over time; positive CMF indicates net buying pressure that supports price.", data_requirements=["High", "Low", "Close", "Volume"], signal_function=signals.chaikin_money_flow, expected_direction=1, data_limitations=ORDER_FLOW_LIMITATION),
            Hypothesis(id="OF_009", category=HypothesisCategory.ORDER_FLOW_PROXY, name="Accumulation/Distribution", description="A/D line trend direction.", economic_rationale="A/D line rising faster than price indicates stealth accumulation by informed players before price catches up.", data_requirements=["High", "Low", "Close", "Volume"], signal_function=signals.accumulation_distribution, expected_direction=1, data_limitations=ORDER_FLOW_LIMITATION),
            Hypothesis(id="OF_010", category=HypothesisCategory.ORDER_FLOW_PROXY, name="Money Flow Index", description="Volume-weighted RSI (MFI) extremes.", economic_rationale="MFI combines price and volume momentum; extremes indicate exhaustion of buying/selling pressure with volume confirmation.", data_requirements=["High", "Low", "Close", "Volume"], signal_function=signals.money_flow_index, expected_direction=0, data_limitations=ORDER_FLOW_LIMITATION),
            Hypothesis(id="OF_011", category=HypothesisCategory.ORDER_FLOW_PROXY, name="Volume-Weighted Momentum", description="Returns weighted by volume give more weight to high-conviction moves.", economic_rationale="Volume-weighting emphasizes high-participation moves; these have more informational content and predictive power.", data_requirements=["Close", "Volume"], signal_function=signals.volume_weighted_momentum, expected_direction=1, data_limitations=ORDER_FLOW_LIMITATION),
            Hypothesis(id="OF_012", category=HypothesisCategory.ORDER_FLOW_PROXY, name="Volume Breakout", description="Volume spike combined with directional price move.", economic_rationale="Volume breakouts signal institutional entry; large players cannot hide their participation and the move continues.", data_requirements=["Close", "Volume"], signal_function=signals.volume_breakout, expected_direction=1, data_limitations=ORDER_FLOW_LIMITATION),
            Hypothesis(id="OF_013", category=HypothesisCategory.ORDER_FLOW_PROXY, name="Volume Dry-Up", description="Very low volume precedes directional move.", economic_rationale="Volume dry-up indicates consolidation; participants are waiting for catalyst, and breakout from low-volume base is directional.", data_requirements=["Close", "Volume"], signal_function=signals.volume_dryup, expected_direction=0, data_limitations=ORDER_FLOW_LIMITATION),
            Hypothesis(id="OF_014", category=HypothesisCategory.ORDER_FLOW_PROXY, name="Up/Down Volume Ratio", description="Rolling ratio of up-day volume to down-day volume.", economic_rationale="Persistent up-volume dominance shows accumulation; institutional buying creates sustained upward pressure.", data_requirements=["Close", "Volume"], signal_function=signals.up_down_volume_ratio, expected_direction=1, data_limitations=ORDER_FLOW_LIMITATION),
            Hypothesis(id="OF_015", category=HypothesisCategory.ORDER_FLOW_PROXY, name="VWAP Deviation", description="Price far from rolling VWAP proxy signals reversion.", economic_rationale="VWAP represents average execution price; institutional algorithms target VWAP, creating gravitational pull back toward it.", data_requirements=["High", "Low", "Close", "Volume"], signal_function=signals.vwap_deviation, expected_direction=0, data_limitations=ORDER_FLOW_LIMITATION + " VWAP approximated from daily typical price, not tick-level."),
            Hypothesis(id="OF_016", category=HypothesisCategory.ORDER_FLOW_PROXY, name="Volume Climax", description="Extreme volume at price extreme signals reversal.", economic_rationale="Volume climax represents forced liquidation or panic buying; all marginal sellers/buyers exhausted at that point.", data_requirements=["Close", "Volume"], signal_function=signals.volume_climax, expected_direction=0, data_limitations=ORDER_FLOW_LIMITATION),
        ]


    def _regime_hypotheses(self) -> list[Hypothesis]:
        """Generate regime/market structure category hypotheses."""
        return [
            Hypothesis(id="REG_001", category=HypothesisCategory.REGIME, name="ADX Trend Detection", description="Trade in trend direction when ADX > 25.", economic_rationale="Strong trends (ADX>25) persist due to institutional order flow and behavioral momentum; trend-following captures these moves.", data_requirements=["High", "Low", "Close"], signal_function=signals.adx_trend_detection, expected_direction=1, data_limitations="ADX from daily bars only; may lag intraday regime shifts."),
            Hypothesis(id="REG_002", category=HypothesisCategory.REGIME, name="Volatility Regime", description="Low vol = trend follow; high vol = mean revert.", economic_rationale="Market microstructure differs by vol regime; low vol supports trending (informed flow dominates), high vol supports reversion (noise dominates).", data_requirements=["Close"], signal_function=signals.volatility_regime_signal, expected_direction=0, data_limitations="Daily OHLCV only."),
            Hypothesis(id="REG_003", category=HypothesisCategory.REGIME, name="Correlation Regime Proxy", description="Return-volatility correlation signals market regime.", economic_rationale="Negative return-vol correlation (leverage effect) is normal; positive correlation signals regime stress and different dynamics.", data_requirements=["Close"], signal_function=signals.correlation_regime_proxy, expected_direction=0, data_limitations="Single-asset correlation proxy; no cross-asset data."),
            Hypothesis(id="REG_004", category=HypothesisCategory.REGIME, name="Drawdown Recovery", description="Buy when recovering from drawdown.", economic_rationale="Recovery from drawdown indicates selling exhaustion and new buyer entry; early recovery often accelerates as confidence returns.", data_requirements=["Close"], signal_function=signals.drawdown_recovery, expected_direction=1, data_limitations="Daily OHLCV only."),
            Hypothesis(id="REG_005", category=HypothesisCategory.REGIME, name="Momentum Dispersion", description="Alignment of momentum across multiple timeframes.", economic_rationale="When all timeframe momentums align, the trend is strong and broad-based; disagreement signals potential reversal.", data_requirements=["Close"], signal_function=signals.momentum_dispersion, expected_direction=1, data_limitations="Daily OHLCV only; single-asset breadth proxy."),
            Hypothesis(id="REG_006", category=HypothesisCategory.REGIME, name="Trend Strength (Efficiency)", description="Ratio of net move to total path length.", economic_rationale="High efficiency ratio means price moved directly (strong trend); low ratio means choppy/range-bound (mean reversion works).", data_requirements=["Close"], signal_function=signals.trend_strength, expected_direction=1, data_limitations="Daily OHLCV only."),
            Hypothesis(id="REG_007", category=HypothesisCategory.REGIME, name="Support/Resistance Levels", description="Trade based on proximity to 20-day high/low.", economic_rationale="Recent highs/lows concentrate stop orders and limit orders; prices near these levels experience increased activity.", data_requirements=["High", "Low", "Close"], signal_function=signals.support_resistance_levels, expected_direction=0, data_limitations="Daily OHLCV only; cannot observe actual order book levels."),
            Hypothesis(id="REG_008", category=HypothesisCategory.REGIME, name="Range Contraction/Expansion", description="Range contraction precedes expansion in recent direction.", economic_rationale="Contracting ranges indicate equilibrium; breakout from contraction tends to be in the direction of the preceding micro-trend.", data_requirements=["High", "Low", "Close"], signal_function=signals.range_contraction_expansion, expected_direction=0, data_limitations="Daily OHLCV only."),
            Hypothesis(id="REG_009", category=HypothesisCategory.REGIME, name="New High/Low Proximity", description="Near 252-day high has momentum; near low has weakness.", economic_rationale="New highs attract momentum buyers and media attention; new lows trigger stop-losses and margin calls creating further selling.", data_requirements=["Close"], signal_function=signals.new_high_low_proximity, expected_direction=1, data_limitations="Daily OHLCV only."),
            Hypothesis(id="REG_010", category=HypothesisCategory.REGIME, name="MA Ribbon", description="Multiple MA alignment (10/20/50/100) indicates trend strength.", economic_rationale="Full MA alignment shows trend at all timeframes; this broad consensus among different-horizon traders supports continuation.", data_requirements=["Close"], signal_function=signals.ma_ribbon, expected_direction=1, data_limitations="Daily OHLCV only; lagging by nature."),
        ]

    def _microstructure_hypotheses(self) -> list[Hypothesis]:
        """Generate microstructure proxy category hypotheses."""
        return [
            Hypothesis(id="MS_001", category=HypothesisCategory.MICROSTRUCTURE_PROXY, name="Bar Range Analysis", description="Wide vs narrow range bars signal continuation vs breakout.", economic_rationale="Wide-range bars show strong conviction; narrow bars show indecision. Breakout from narrow range tends to be directional.", data_requirements=["Open", "High", "Low", "Close"], signal_function=signals.bar_range_analysis, expected_direction=0, data_limitations="Daily bars only; intraday microstructure not observable."),
            Hypothesis(id="MS_002", category=HypothesisCategory.MICROSTRUCTURE_PROXY, name="Doji Pattern", description="Doji (tiny body) after trend signals indecision/reversal.", economic_rationale="Doji after extended move shows equilibrium between buyers and sellers; often marks exhaustion point of the trend.", data_requirements=["Open", "High", "Low", "Close"], signal_function=signals.doji_pattern, expected_direction=0, data_limitations="Daily OHLCV only; candle patterns from daily bars have lower reliability than intraday."),
            Hypothesis(id="MS_003", category=HypothesisCategory.MICROSTRUCTURE_PROXY, name="Hammer Pattern", description="Long lower wick (hammer) signals bullish reversal.", economic_rationale="Hammer shows sellers pushed price down but buyers overwhelmed them by close; strong buying interest at lower levels.", data_requirements=["Open", "High", "Low", "Close"], signal_function=signals.hammer_pattern, expected_direction=0, data_limitations="Daily OHLCV only."),
            Hypothesis(id="MS_004", category=HypothesisCategory.MICROSTRUCTURE_PROXY, name="Engulfing Pattern", description="Current bar body engulfs previous bar signals reversal.", economic_rationale="Engulfing pattern shows dramatic shift in sentiment; the new direction has enough force to completely overwhelm prior bar.", data_requirements=["Open", "High", "Low", "Close"], signal_function=signals.engulfing_pattern, expected_direction=0, data_limitations="Daily OHLCV only."),
            Hypothesis(id="MS_005", category=HypothesisCategory.MICROSTRUCTURE_PROXY, name="Consecutive Direction", description="Streak of same-direction closes signals momentum or exhaustion.", economic_rationale="Short streaks (4-6) show momentum; very long streaks (7+) show potential exhaustion as directional capital depletes.", data_requirements=["Close"], signal_function=signals.consecutive_direction, expected_direction=0, data_limitations="Daily OHLCV only."),
            Hypothesis(id="MS_006", category=HypothesisCategory.MICROSTRUCTURE_PROXY, name="Opening Range Proxy", description="Open-to-high/low ratio signals session direction.", economic_rationale="Where price goes relative to open in the session reveals order flow direction; persistent pattern indicates directional flow.", data_requirements=["Open", "High", "Low"], signal_function=signals.opening_range_proxy, expected_direction=0, data_limitations="Daily bars only; actual opening range breakout requires intraday data."),
            Hypothesis(id="MS_007", category=HypothesisCategory.MICROSTRUCTURE_PROXY, name="Close Position (IBS)", description="Internal Bar Strength: close near low = buy, near high = sell.", economic_rationale="IBS is one of the most robust short-term mean reversion signals; close near low indicates selling exhaustion within the session.", data_requirements=["High", "Low", "Close"], signal_function=signals.close_position_in_range, expected_direction=0, data_limitations="Daily OHLCV only."),
            Hypothesis(id="MS_008", category=HypothesisCategory.MICROSTRUCTURE_PROXY, name="True Range Ratio", description="Today TR vs ATR shows unusual move.", economic_rationale="True range significantly exceeding ATR indicates event-driven move; direction of the move tends to have follow-through.", data_requirements=["High", "Low", "Close"], signal_function=signals.true_range_ratio, expected_direction=0, data_limitations="Daily OHLCV only."),
            Hypothesis(id="MS_009", category=HypothesisCategory.MICROSTRUCTURE_PROXY, name="Bar-to-Bar Acceleration", description="Momentum acceleration between consecutive bars.", economic_rationale="Increasing bar-to-bar momentum signals strengthening flow; new participants entering create acceleration that persists short-term.", data_requirements=["Close"], signal_function=signals.momentum_acceleration_bar, expected_direction=1, data_limitations="Daily OHLCV only."),
            Hypothesis(id="MS_010", category=HypothesisCategory.MICROSTRUCTURE_PROXY, name="Volume-Range Relationship", description="High volume + narrow range = accumulation.", economic_rationale="High volume in narrow range indicates large players absorbing supply/demand without moving price; precedes directional breakout.", data_requirements=["Open", "High", "Low", "Close", "Volume"], signal_function=signals.volume_range_relationship, expected_direction=0, data_limitations="Daily OHLCV only; true accumulation requires tick-level observation."),
            Hypothesis(id="MS_011", category=HypothesisCategory.MICROSTRUCTURE_PROXY, name="Price Rejection", description="Long wicks show rejection of price levels.", economic_rationale="Long wicks indicate strong supply/demand at those levels; price was rejected by limit orders or aggressive counter-flow.", data_requirements=["Open", "High", "Low", "Close"], signal_function=signals.price_rejection, expected_direction=0, data_limitations="Daily OHLCV only; actual rejection requires order book data."),
            Hypothesis(id="MS_012", category=HypothesisCategory.MICROSTRUCTURE_PROXY, name="Inside/Outside Bar", description="Inside bars signal consolidation; outside bars signal reversal.", economic_rationale="Inside bars show decreasing volatility (breakout pending); outside bars show volatility expansion with directional information.", data_requirements=["Open", "High", "Low", "Close"], signal_function=signals.inside_outside_bar, expected_direction=0, data_limitations="Daily OHLCV only."),
        ]

