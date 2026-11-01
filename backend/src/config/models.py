"""
Modelos ORM de SQLAlchemy que mapean las 4 tablas de PostgreSQL.
Estos modelos son los únicos que conocen la estructura física de la DB.
"""
from datetime import date, datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.config.database import Base


class OhlcvCandleORM(Base):
    """
    Velas OHLCV de Bitcoin desde Binance.
    Tabla principal: ohlcv_candles
    """
    __tablename__ = "ohlcv_candles"
    __table_args__ = (
        UniqueConstraint("symbol", "interval", "open_time", name="uq_candle_symbol_interval_time"),
        CheckConstraint(
            "taker_buy_base_volume IS NULL OR "
            "(taker_buy_base_volume >= 0 AND taker_buy_base_volume <= volume)",
            name="ck_candle_taker_base_volume",
        ),
        CheckConstraint(
            "taker_buy_quote_volume IS NULL OR "
            "(taker_buy_quote_volume >= 0 AND quote_volume IS NOT NULL "
            "AND taker_buy_quote_volume <= quote_volume)",
            name="ck_candle_taker_quote_volume",
        ),
        CheckConstraint(
            "(taker_buy_base_volume IS NULL) = (taker_buy_quote_volume IS NULL)",
            name="ck_candle_taker_volume_pair",
        ),
        {"comment": "Velas OHLCV de Bitcoin desde Binance Public API"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    interval: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    open_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    close_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    high: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    low: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    volume: Mapped[float] = mapped_column(Numeric(30, 8), nullable=False)
    quote_volume: Mapped[float] = mapped_column(Numeric(30, 8), nullable=True)
    num_trades: Mapped[int | None] = mapped_column(Integer, nullable=True)
    taker_buy_base_volume: Mapped[float | None] = mapped_column(Numeric(30, 8))
    taker_buy_quote_volume: Mapped[float | None] = mapped_column(Numeric(30, 8))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relaciones
    features: Mapped[list["ComputedFeatureORM"]] = relationship(
        back_populates="candle", cascade="all, delete-orphan"
    )
    predictions: Mapped[list["PredictionLogORM"]] = relationship(
        back_populates="candle", cascade="all, delete-orphan"
    )


class ComputedFeatureORM(Base):
    """
    Features técnicas y macroeconómicas calculadas para cada vela.
    Tabla: computed_features
    """
    __tablename__ = "computed_features"
    __table_args__ = (
        UniqueConstraint("candle_id", name="uq_features_candle"),
        {"comment": "Features calculadas para el modelo ML"},
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    candle_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("ohlcv_candles.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # ── Indicadores técnicos de precio ─────────────────────
    rsi_14: Mapped[float | None] = mapped_column(Numeric(10, 4))
    macd: Mapped[float | None] = mapped_column(Numeric(20, 8))
    macd_signal: Mapped[float | None] = mapped_column(Numeric(20, 8))
    macd_histogram: Mapped[float | None] = mapped_column(Numeric(20, 8))
    bb_upper: Mapped[float | None] = mapped_column(Numeric(20, 8))
    bb_middle: Mapped[float | None] = mapped_column(Numeric(20, 8))
    bb_lower: Mapped[float | None] = mapped_column(Numeric(20, 8))
    bb_width: Mapped[float | None] = mapped_column(Numeric(10, 6))
    ema_9: Mapped[float | None] = mapped_column(Numeric(20, 8))
    ema_21: Mapped[float | None] = mapped_column(Numeric(20, 8))
    ema_50: Mapped[float | None] = mapped_column(Numeric(20, 8))
    sma_200: Mapped[float | None] = mapped_column(Numeric(20, 8))

    # ── Indicadores de volatilidad ──────────────────────────
    atr_14: Mapped[float | None] = mapped_column(Numeric(20, 8))
    hist_volatility_20: Mapped[float | None] = mapped_column(Numeric(10, 6))
    hl_range_pct: Mapped[float | None] = mapped_column(Numeric(10, 6))

    # ── Indicadores de volumen ──────────────────────────────
    obv: Mapped[float | None] = mapped_column(Numeric(30, 2))
    volume_sma_ratio: Mapped[float | None] = mapped_column(Numeric(10, 4))
    vwap_daily: Mapped[float | None] = mapped_column(Numeric(20, 8))
    accumulation_distribution: Mapped[float | None] = mapped_column(Numeric(30, 8))

    # ── Momentum ────────────────────────────────────────────
    roc_5: Mapped[float | None] = mapped_column(Numeric(10, 6))
    roc_10: Mapped[float | None] = mapped_column(Numeric(10, 6))
    williams_r: Mapped[float | None] = mapped_column(Numeric(10, 4))
    stoch_rsi: Mapped[float | None] = mapped_column(Numeric(10, 4))
    price_ema50_distance: Mapped[float | None] = mapped_column(Numeric(20, 10))
    ema21_slope_5: Mapped[float | None] = mapped_column(Numeric(20, 10))
    higher_highs_20: Mapped[int | None] = mapped_column(Integer)
    lower_lows_20: Mapped[int | None] = mapped_column(Integer)

    # ── Macroeconómicos ─────────────────────────────────────
    dxy_close: Mapped[float | None] = mapped_column(Numeric(10, 4))
    dxy_change_pct: Mapped[float | None] = mapped_column(Numeric(10, 6))
    sp500_close: Mapped[float | None] = mapped_column(Numeric(10, 2))
    sp500_change_pct: Mapped[float | None] = mapped_column(Numeric(10, 6))
    vix_close: Mapped[float | None] = mapped_column(Numeric(10, 4))
    gold_change_pct: Mapped[float | None] = mapped_column(Numeric(10, 6))
    btc_sp500_corr_20: Mapped[float | None] = mapped_column(Numeric(10, 6))

    # ── Sentimiento ─────────────────────────────────────────
    fear_greed_value: Mapped[int | None] = mapped_column(Integer)
    fear_greed_sma_7: Mapped[float | None] = mapped_column(Numeric(10, 4))
    fear_greed_momentum_7: Mapped[float | None] = mapped_column(Numeric(10, 4))

    # ── Features temporales ─────────────────────────────────
    hour_of_day: Mapped[int | None] = mapped_column(Integer)
    day_of_week: Mapped[int | None] = mapped_column(Integer)
    is_weekend: Mapped[bool | None] = mapped_column(Boolean)
    weekday_mon: Mapped[bool | None] = mapped_column(Boolean)
    weekday_tue: Mapped[bool | None] = mapped_column(Boolean)
    weekday_wed: Mapped[bool | None] = mapped_column(Boolean)
    weekday_thu: Mapped[bool | None] = mapped_column(Boolean)
    weekday_fri: Mapped[bool | None] = mapped_column(Boolean)
    weekday_sat: Mapped[bool | None] = mapped_column(Boolean)
    weekday_sun: Mapped[bool | None] = mapped_column(Boolean)

    # ── Target ──────────────────────────────────────────────
    # 1 = próxima vela sube (close[t+1] > close[t])
    # 0 = próxima vela baja
    # NULL = última vela (target desconocido)
    target_label: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relaciones
    candle: Mapped["OhlcvCandleORM"] = relationship(back_populates="features")


class ContextualFeatureORM(Base):
    """
    Features contextuales por vela (JSONB) calculadas por el ML worker.
    Complementan `computed_features` con señales derivadas en lote (contexto de precio).
    Tabla: contextual_features
    """
    __tablename__ = "contextual_features"
    __table_args__ = {"comment": "Features contextuales por vela calculadas por el ML worker"}

    candle_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("ohlcv_candles.id", ondelete="CASCADE"), primary_key=True
    )
    feature_values: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ModelRegistryORM(Base):
    """
    Registro de modelos entrenados y sus métricas de validación.
    Tabla: model_registry
    """
    __tablename__ = "model_registry"
    __table_args__ = {"comment": "Registro de versiones de modelos ML entrenados"}

    version: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)   # xgboost, lstm, ensemble
    hyperparameters: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    val_accuracy: Mapped[float | None] = mapped_column(Numeric(6, 4))
    val_f1: Mapped[float | None] = mapped_column(Numeric(6, 4))
    val_precision: Mapped[float | None] = mapped_column(Numeric(6, 4))
    val_recall: Mapped[float | None] = mapped_column(Numeric(6, 4))
    baseline_accuracy: Mapped[float | None] = mapped_column(Numeric(6, 4))
    artifact_path: Mapped[str | None] = mapped_column(String(500))
    scaler_path: Mapped[str | None] = mapped_column(String(500))
    feature_names: Mapped[list | None] = mapped_column(JSON)              # Orden exacto de features
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    notes: Mapped[str | None] = mapped_column(String(1000))

    # Relaciones
    predictions: Mapped[list["PredictionLogORM"]] = relationship(
        back_populates="model", cascade="all, delete-orphan"
    )


class PredictionLogORM(Base):
    """
    Registro histórico de predicciones emitidas y su resultado real.
    Permite calcular el accuracy histórico real del modelo.
    Tabla: predictions_log
    """
    __tablename__ = "predictions_log"
    __table_args__ = {"comment": "Historial de predicciones emitidas vs resultados reales"}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    candle_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("ohlcv_candles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_version: Mapped[int] = mapped_column(
        Integer, ForeignKey("model_registry.version", ondelete="CASCADE"), nullable=False
    )
    predicted_direction: Mapped[str] = mapped_column(String(4), nullable=False)  # "UP" | "DOWN"
    confidence_score: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False)
    raw_probability_up: Mapped[float | None] = mapped_column(Numeric(6, 4))

    # Resuelto cuando cierra la siguiente vela
    actual_direction: Mapped[str | None] = mapped_column(String(4), nullable=True)
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relaciones
    candle: Mapped["OhlcvCandleORM"] = relationship(back_populates="predictions")
    model: Mapped["ModelRegistryORM"] = relationship(back_populates="predictions")



class MacroDailyORM(Base):
    """
    Cierres diarios de series macro (yfinance).
    Tabla: macro_daily
    """
    __tablename__ = "macro_daily"
    __table_args__ = {
        "comment": "Cierres diarios de series macro (DXY, S&P500, VIX, Gold) desde yfinance"
    }

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    close: Mapped[float] = mapped_column(Numeric(20, 6), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class FearGreedDailyORM(Base):
    """
    Fear & Greed Index diario (alternative.me).
    Tabla: fear_greed_daily
    """
    __tablename__ = "fear_greed_daily"
    __table_args__ = (
        CheckConstraint("value BETWEEN 0 AND 100", name="ck_fear_greed_range"),
        {"comment": "Fear & Greed Index diario (alternative.me), publicado a las 00:00 UTC"},
    )

    obs_date: Mapped[date] = mapped_column(Date, primary_key=True)
    value: Mapped[int] = mapped_column(Integer, nullable=False)
    classification: Mapped[str | None] = mapped_column(String(30))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class FundingRateORM(Base):
    """Eventos de funding de futuros perpetuos."""

    __tablename__ = "funding_rates"
    __table_args__ = (
        CheckConstraint("mark_price IS NULL OR mark_price > 0", name="ck_funding_mark_price"),
        {"comment": "Eventos históricos de funding de futuros perpetuos de Binance"},
    )

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    funding_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    funding_rate: Mapped[float] = mapped_column(Numeric(20, 12), nullable=False)
    mark_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
