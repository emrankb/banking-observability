"""Training-only selection and frozen test replay for six monitoring targets."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits

from .forecasting import file_sha256, runtime_versions
from .kpi_forecasting import (
    CANDIDATES, HORIZON, HISTORY_ROWS, TARGETS, KPICandidate, KPIForecaster, build_features,
    fit_candidate, read_history, supervised,
)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError('Need aligned nonempty targets and forecasts.')
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Cannot score non-finite values.')
    return {
        'n': len(actual),
        'mae_pp': float(mean_absolute_error(actual, predicted)),
        'rmse_pp': float(root_mean_squared_error(actual, predicted)),
        'bias_pp': float(np.mean(predicted - actual)),
    }


def folds(features: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    result = list(TimeSeriesSplit(n_splits=3, test_size=1200, gap=HORIZON).split(features))
    for train, validation in result:
        # A training label at origin t uses the observation at t+5.
        last_label = features.index[train[-1]] + pd.Timedelta(minutes=HORIZON)
        if last_label >= features.index[validation[0]]:
            raise AssertionError('Training label overlaps validation origins.')
    return result


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def train_all(data_dir: Path, model_dir: Path, report_dir: Path) -> dict:
    data_dir, model_dir, report_dir = map(Path, (data_dir, model_dir, report_dir))
    train_path = data_dir / 'infrastructure_metrics_train.csv'
    test_path = data_dir / 'infrastructure_metrics_test.csv'
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError('Missing input: ' + str(path))
    source_paths = [p.resolve() for p in (data_dir, train_path, test_path)]
    for destination in (model_dir, report_dir):
        resolved = destination.resolve()
        if destination.exists() or any(resolved == p or resolved in p.parents or p in resolved.parents for p in source_paths):
            raise ValueError('Use new output directories separate from input data.')
    if model_dir.resolve() == report_dir.resolve() or model_dir.resolve() in report_dir.resolve().parents or report_dir.resolve() in model_dir.resolve().parents:
        raise ValueError('Model and report directories must be separate.')

    train_hash = file_sha256(train_path)
    train = read_history(train_path)
    seed = int(train_hash[:8], 16)
    choices = {}
    fold_rows = []
    all_models = {}
    with threadpool_limits(limits=1):
        for target in TARGETS:
            x, y = supervised(train, target)
            cv = folds(x)
            for name in CANDIDATES:
                for fold_number, (i_train, i_valid) in enumerate(cv, 1):
                    fit = fit_candidate(name, x.iloc[i_train], y.iloc[i_train], seed=seed + fold_number)
                    predicted = fit.predict(x.iloc[i_valid])
                    fold_rows.append({
                        'target': target, 'candidate': name, 'fold': fold_number,
                        'train_origin_end': x.index[i_train[-1]].isoformat(),
                        'validation_origin_start': x.index[i_valid[0]].isoformat(),
                        'validation_origin_end': x.index[i_valid[-1]].isoformat(),
                        **metrics(y.iloc[i_valid].to_numpy(), predicted),
                    })
            chosen = pd.DataFrame([r for r in fold_rows if r['target'] == target]).groupby('candidate').agg(
                cv_mae_pp=('mae_pp', 'mean'), cv_rmse_pp=('rmse_pp', 'mean'), worst_fold_mae_pp=('mae_pp', 'max'),
            ).reset_index().sort_values(['cv_mae_pp', 'cv_rmse_pp', 'candidate'], kind='stable')
            winner = str(chosen.iloc[0]['candidate'])
            choices[target] = {'candidate': winner, 'cv_mae_pp': float(chosen.iloc[0]['cv_mae_pp']),
                               'training_origins': len(x), 'fit_origin_end': x.index[-1].isoformat(),
                               'trained_target_through': (x.index[-1] + pd.Timedelta(minutes=HORIZON)).isoformat()}
            all_models[target] = fit_candidate(winner, x, y, seed=seed)

    # Read test data only after all six selections are frozen.
    test = read_history(test_path)
    if test.timestamp.iloc[0] != train.timestamp.iloc[-1] + pd.Timedelta(minutes=1):
        raise ValueError('Test timestamps must immediately follow training timestamps.')
    combined = pd.concat([train, test], ignore_index=True)
    start = test.timestamp.iloc[0]
    summaries = {}
    test_rows = []
    test_predictions = []
    with threadpool_limits(limits=1):
        for target in TARGETS:
            x, y = supervised(combined, target)
            mask = x.index >= start
            x, y = x.loc[mask], y.loc[mask]
            if len(x) != len(test) - HORIZON:
                raise AssertionError('Test replay must cover every origin with an observed t+5 target.')
            for name in CANDIDATES:
                model = (all_models[target] if name == choices[target]['candidate']
                         else KPICandidate(name) if name not in ('ridge', 'hgb') else None)
                if model is None:
                    continue  # Nonselected learned candidates are not fitted on the test interval.
                predicted = model.predict(x)
                scores = metrics(y.to_numpy(), predicted)
                test_rows.append({'target': target, 'candidate': name, 'selected': name == choices[target]['candidate'], **scores})
                if name == choices[target]['candidate'] or name == 'persistence':
                    for origin, actual, estimate in zip(x.index, y.to_numpy(), predicted):
                        test_predictions.append({'target': target, 'candidate': name, 'origin': origin.isoformat(),
                                                 'target_timestamp': (origin + pd.Timedelta(minutes=HORIZON)).isoformat(),
                                                 'actual': float(actual), 'prediction': float(estimate)})
            selected = next(r for r in test_rows if r['target'] == target and r['selected'])
            baseline = next(r for r in test_rows if r['target'] == target and r['candidate'] == 'persistence')
            summaries[target] = {
                **choices[target], 'test_mae_pp': selected['mae_pp'], 'test_rmse_pp': selected['rmse_pp'],
                'persistence_test_mae_pp': baseline['mae_pp'],
                'test_mae_improvement_vs_persistence_pct': float(100 * (1 - selected['mae_pp'] / baseline['mae_pp'])) if baseline['mae_pp'] > 0 else None,
                'test_origins': len(x),
                'test_origin_start': x.index[0].isoformat(), 'test_origin_end': x.index[-1].isoformat(),
            }

    # Recreate the candidate fits from train for final test metrics, including nonselected learned comparators.
    # Keeping the selected model fixed ensures that this step never changes the served candidate.
    with threadpool_limits(limits=1):
        for target in TARGETS:
            xtrain, ytrain = supervised(train, target)
            xtest, ytest = supervised(combined, target)
            xtest, ytest = xtest.loc[xtest.index >= start], ytest.loc[xtest.index >= start]
            for name in ('ridge', 'hgb'):
                if name == choices[target]['candidate']:
                    continue
                fitted = fit_candidate(name, xtrain, ytrain, seed=seed)
                test_rows.append({'target': target, 'candidate': name, 'selected': False,
                                  **metrics(ytest.to_numpy(), fitted.predict(xtest))})

    model_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)
    from . import kpi_forecasting as code
    for target, fitted in all_models.items():
        directory = model_dir / target
        directory.mkdir()
        joblib.dump(fitted, directory / 'model.joblib', compress=3)
        metadata = {
            'format_version': 1, 'target': target, 'horizon_minutes': HORIZON,
            'candidate': fitted.name,
            'feature_columns': list(build_features(train.iloc[-HISTORY_ROWS:], target).columns),
            'code_sha256': file_sha256(Path(code.__file__)),
            'model_sha256': file_sha256(directory / 'model.joblib'),
            'runtime_versions': runtime_versions(),
            'training_source_sha256': train_hash,
            'trained_target_through': choices[target]['trained_target_through'],
            'training_range': {'min': float(train[target].min()), 'max': float(train[target].max())},
        }
        dump(directory / 'metadata.json', metadata)
        replay = KPIForecaster(directory).predict(combined, as_of=start)
        dump(report_dir / f'{target}_inference.json', replay)

    pd.DataFrame(fold_rows).to_csv(report_dir / 'cv_fold_metrics.csv', index=False)
    leaderboard = pd.DataFrame(fold_rows).groupby(['target', 'candidate']).agg(
        mean_cv_mae_pp=('mae_pp', 'mean'), mean_cv_rmse_pp=('rmse_pp', 'mean'),
    ).reset_index().sort_values(['target', 'mean_cv_mae_pp', 'mean_cv_rmse_pp', 'candidate'])
    leaderboard.to_csv(report_dir / 'cv_leaderboard.csv', index=False)
    pd.DataFrame(test_rows).sort_values(['target', 'mae_pp']).to_csv(report_dir / 'test_metrics.csv', index=False)
    pd.DataFrame(test_predictions).to_csv(report_dir / 'test_predictions.csv', index=False)
    summary = {
        'scope': 'five-minute-ahead point readings, not intervals or incident probabilities',
        'targets': summaries, 'cv_candidates_per_target': len(CANDIDATES), 'cv_folds': 3,
        'raw_train_rows': len(train), 'raw_test_rows': len(test),
        'source_sha256': {train_path.name: train_hash, test_path.name: file_sha256(test_path)},
        'feature_warmup_rows': HISTORY_ROWS - 1,
        'test_access': 'after selecting all targets on training-only expanding chronological folds',
        'origin_target_alignment': 'observed values at origin t; single point target t+5',
        'no_uncertainty_interval': True, 'no_incident_root_cause': True,
    }
    dump(report_dir / 'summary.json', summary)
    print('KPI FORECASTS: five-minute targets, MAE in percentage points')
    print(pd.DataFrame([{'target': target, **values} for target, values in summaries.items()])[
        ['target', 'candidate', 'cv_mae_pp', 'test_mae_pp', 'persistence_test_mae_pp',
         'test_mae_improvement_vs_persistence_pct']].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('Reports: ' + str(report_dir))
    print('Models: ' + str(model_dir))
    return summary
