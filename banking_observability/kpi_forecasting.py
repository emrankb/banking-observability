"""Five-minute forecasts for utilization and channel success-rate readings."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .forecasting import file_sha256, runtime_versions

HORIZON = 5
DAY = 1440
TARGETS = (
    'cpu_utilization', 'memory_usage', 'db_connection_pool_usage',
    'neft_success_rate', 'imps_success_rate', 'rtgs_success_rate',
)
REQUIRED = ('timestamp', *TARGETS)
CANDIDATES = ('persistence', 'trailing_15', 'trailing_60', 'daily_seasonal', 'ridge', 'hgb')
FEATURE_CODE = Path(__file__)
LAGS = (0, 1, 5, 15, 60)
WINDOWS = (5, 15, 60)
HISTORY_ROWS = DAY - HORIZON + 1


def validate_history(frame: pd.DataFrame, *, minimum_rows: int = 1) -> pd.DataFrame:
    """Check the observed minute grid without filling or capping source readings."""
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate input columns.')
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    if len(frame) < minimum_rows:
        raise ValueError(f'Need at least {minimum_rows} rows of monitoring history.')
    data = frame.loc[:, REQUIRED].copy()
    try:
        data['timestamp'] = pd.to_datetime(data['timestamp'], errors='raise')
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid monitoring timestamp.') from exc
    times = data['timestamp']
    if not pd.api.types.is_datetime64_any_dtype(times.dtype) or times.isna().any() or times.dt.tz is not None:
        raise ValueError('Expected timezone-naive, non-null monitoring timestamps.')
    if times.duplicated().any() or not times.is_monotonic_increasing or not times.eq(times.dt.floor('min')).all():
        raise ValueError('Monitoring timestamps must be ordered, unique minute boundaries.')
    if not times.diff().iloc[1:].eq(pd.Timedelta(minutes=1)).all():
        raise ValueError('Missing or irregular monitoring minutes; no interpolation is performed.')
    try:
        data[list(TARGETS)] = data[list(TARGETS)].apply(pd.to_numeric, errors='raise').astype('float64')
    except (TypeError, ValueError) as exc:
        raise ValueError('Monitoring readings must be numeric.') from exc
    observed = data[list(TARGETS)].to_numpy()
    if not np.isfinite(observed).all() or (observed < 0).any() or (observed > 100).any():
        raise ValueError('Utilization and success readings must be finite and within 0–100.')
    return data.reset_index(drop=True)


def read_history(path: Path) -> pd.DataFrame:
    return validate_history(pd.read_csv(path))


def build_features(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in TARGETS:
        raise ValueError('Unknown forecast target: ' + target)
    data = validate_history(frame)
    series = data[target]
    target_time = pd.DatetimeIndex(data['timestamp'] + pd.Timedelta(minutes=HORIZON))
    minute = target_time.hour * 60 + target_time.minute
    features: dict[str, object] = {
        'clock_sin': np.sin(2 * np.pi * minute / DAY),
        'clock_cos': np.cos(2 * np.pi * minute / DAY),
        'clock_weekend': (target_time.dayofweek >= 5).astype(float),
    }
    for lag in LAGS:
        features[f'lag_{lag}'] = series.shift(lag).to_numpy()
    for window in WINDOWS:
        rolling = series.rolling(window, min_periods=window)
        features[f'mean_{window}'] = rolling.mean().to_numpy()
        features[f'std_{window}'] = rolling.std(ddof=0).to_numpy()
    features['change_5'] = (series - series.shift(5)).to_numpy()
    # At origin t, the same clock minute yesterday for y[t+5] is t-1435.
    features['daily_target'] = series.shift(DAY - HORIZON).to_numpy()
    return pd.DataFrame(features, index=pd.DatetimeIndex(data.timestamp, name='forecast_origin'))


def supervised(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    features = build_features(frame, target)
    label = pd.Series(frame[target].shift(-HORIZON).to_numpy(), index=features.index, name=target)
    valid = features.notna().all(axis=1) & label.notna()
    return features.loc[valid], label.loc[valid]


@dataclass
class KPICandidate:
    name: str
    estimator: object | None = None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.name not in CANDIDATES:
            raise ValueError('Unexpected model type.')
        if self.name == 'persistence':
            values = features['lag_0'].to_numpy(dtype=float)
        elif self.name == 'trailing_15':
            values = features['mean_15'].to_numpy(dtype=float)
        elif self.name == 'trailing_60':
            values = features['mean_60'].to_numpy(dtype=float)
        elif self.name == 'daily_seasonal':
            values = features['daily_target'].to_numpy(dtype=float)
        else:
            if self.estimator is None:
                raise ValueError('Learned candidate has no fitted estimator.')
            values = self.estimator.predict(features)
        # These source metrics are rates/percentages; predicted values must obey their bounds.
        return np.clip(np.asarray(values, dtype=float), 0.0, 100.0)


def fit_candidate(name: str, features: pd.DataFrame, label: pd.Series, *, seed: int) -> KPICandidate:
    if name not in CANDIDATES:
        raise ValueError('Unknown candidate: ' + name)
    if name == 'ridge':
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    elif name == 'hgb':
        estimator = HistGradientBoostingRegressor(max_iter=75, max_depth=3, min_samples_leaf=25,
                                                  learning_rate=0.06, early_stopping=False, random_state=seed)
    else:
        return KPICandidate(name)
    estimator.fit(features, label)
    return KPICandidate(name, estimator)


class KPIForecaster:
    """Load a trusted model with the feature and runtime versions used during fitting."""
    def __init__(self, path: Path):
        self.path = Path(path)
        self.metadata = json.loads((self.path / 'metadata.json').read_text())
        meta = self.metadata
        if meta.get('format_version') != 1 or meta.get('target') not in TARGETS or meta.get('horizon_minutes') != HORIZON:
            raise ValueError('Unsupported KPI forecast artifact metadata.')
        if meta.get('code_sha256') != file_sha256(FEATURE_CODE):
            raise ValueError('Forecast code differs from the saved artifact.')
        if meta.get('runtime_versions') != runtime_versions():
            raise ValueError('Package version mismatch; install the recorded requirements or refit.')
        filename = self.path / 'model.joblib'
        if file_sha256(filename) != meta.get('model_sha256'):
            raise ValueError('Model checksum mismatch; refusing to load the artifact.')
        self.model = joblib.load(filename)  # Local trusted artifacts only; joblib uses pickle.
        if not isinstance(self.model, KPICandidate) or self.model.name != meta['candidate']:
            raise ValueError('Saved candidate differs from the model metadata.')

    def predict(self, history: pd.DataFrame, *, as_of: str | pd.Timestamp | None = None) -> dict:
        if 'timestamp' not in history:
            raise ValueError('Missing monitoring timestamps.')
        dates = pd.to_datetime(history['timestamp'], errors='raise')
        if len(dates) == 0:
            raise ValueError('Empty monitoring history.')
        origin = pd.Timestamp(dates.iloc[-1]) if as_of is None else pd.Timestamp(as_of)
        if pd.isna(origin) or origin.tzinfo is not None or origin != origin.floor('min'):
            raise ValueError('Origin must be an observed timezone-naive minute.')
        if not dates.eq(origin).any():
            raise ValueError('Origin must match a source monitoring timestamp.')
        if origin <= pd.Timestamp(self.metadata['trained_target_through']):
            raise ValueError('Origin must be later than the training-target cutoff.')
        # Trim future rows before feature validation or construction.
        causal = validate_history(history.loc[dates.le(origin)], minimum_rows=HISTORY_ROWS).tail(HISTORY_ROWS)
        row = build_features(causal, self.metadata['target']).tail(1)
        if row.index[-1] != origin or row.isna().any().any() or list(row.columns) != self.metadata['feature_columns']:
            raise ValueError('Insufficient aligned feature history.')
        with threadpool_limits(limits=1):
            predicted = float(self.model.predict(row)[0])
        if not np.isfinite(predicted):
            raise ValueError('Non-finite forecast.')
        latest = float(causal[self.metadata['target']].iloc[-1])
        train_range = self.metadata['training_range']
        return {
            'forecast_origin': origin.isoformat(),
            'target_timestamp': (origin + pd.Timedelta(minutes=HORIZON)).isoformat(),
            'horizon_minutes': HORIZON,
            'target_column': self.metadata['target'],
            'predicted_value': predicted,
            'last_observed_value': latest,
            'unit': 'percent',
            'model_name': self.model.name,
            'history_rows_used': len(causal),
            'trained_target_through': self.metadata['trained_target_through'],
            'latest_outside_training_range': not train_range['min'] <= latest <= train_range['max'],
            'model_sha256': self.metadata['model_sha256'],
            'source_timezone': 'unspecified; timezone-naive timestamps',
        }
