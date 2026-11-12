#!/usr/bin/env python3
"""Check multi-target forecast targets, selection, saved models, and replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from banking_observability.forecasting import file_sha256
from banking_observability.kpi_forecasting import (
    CANDIDATES, HORIZON, HISTORY_ROWS, TARGETS, KPIForecaster,
    build_features, fit_candidate, read_history, supervised, validate_history,
)
from banking_observability.kpi_experiment import folds, metrics, train_all

DATA: Path
MODELS: Path
REPORTS: Path


class KPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = read_history(DATA / 'infrastructure_metrics_train.csv')
        cls.test = read_history(DATA / 'infrastructure_metrics_test.csv')
        cls.whole = pd.concat([cls.train, cls.test], ignore_index=True)
        cls.summary = json.loads((REPORTS / 'summary.json').read_text())
        cls.board = pd.read_csv(REPORTS / 'cv_leaderboard.csv')
        cls.fold_rows = pd.read_csv(REPORTS / 'cv_fold_metrics.csv')
        cls.test_metrics = pd.read_csv(REPORTS / 'test_metrics.csv')
        cls.test_predictions = pd.read_csv(REPORTS / 'test_predictions.csv')
        cls.orig = {p.name: file_sha256(p) for p in DATA.glob('*.csv')}

    def test_exact_six_source_targets(self):
        self.assertEqual(TARGETS, ('cpu_utilization','memory_usage','db_connection_pool_usage',
                                   'neft_success_rate','imps_success_rate','rtgs_success_rate'))
        self.assertEqual(len(self.summary['targets']), 6)
        for target in TARGETS:
            self.assertIn(target, self.train)

    def test_5_minute_target_value_not_average(self):
        x, y = supervised(self.train, 'neft_success_rate')
        for n in (0, 20, 2000, len(y)-1):
            timestamp = x.index[n] + pd.Timedelta(minutes=5)
            self.assertEqual(y.iloc[n], self.train.loc[self.train.timestamp.eq(timestamp), 'neft_success_rate'].iloc[0])

    def test_history_is_at_or_before_origin(self):
        for target in TARGETS:
            before = build_features(self.whole.iloc[:1950], target).iloc[-1]
            after = build_features(self.whole.iloc[:1990], target).iloc[1949]
            pd.testing.assert_series_equal(before, after, check_exact=False, rtol=0, atol=1e-10)

    def test_mutating_future_does_not_change_features(self):
        origin_index = 8300
        original = build_features(self.whole.iloc[:origin_index+1], 'cpu_utilization').iloc[-1]
        changed = self.whole.iloc[:origin_index+20].copy()
        changed.loc[origin_index+1:, 'cpu_utilization'] = 100
        other = build_features(changed, 'cpu_utilization').iloc[origin_index]
        pd.testing.assert_series_equal(original, other, check_exact=False, rtol=0, atol=1e-10)

    def test_rolling_windows_include_latest_observation(self):
        frame = build_features(self.train, 'memory_usage')
        n = 2500
        self.assertEqual(frame.iloc[n].lag_0, self.train['memory_usage'].iloc[n])
        for window in (5,15,60):
            expected = self.train['memory_usage'].iloc[n-window+1:n+1].mean()
            self.assertAlmostEqual(frame.iloc[n][f'mean_{window}'], expected, places=10)

    def test_daily_reference_targets_same_minute_yesterday(self):
        frame = build_features(self.train, 'cpu_utilization')
        n = 2500
        self.assertEqual(frame.iloc[n].daily_target, self.train.cpu_utilization.iloc[n+5-1440])

    def test_train_only_time_folds_purged_for_label_horizon(self):
        x,_ = supervised(self.train, TARGETS[0]); cv=folds(x)
        self.assertEqual(len(cv), 3)
        for tr,val in cv:
            self.assertLess(x.index[tr[-1]]+pd.Timedelta(minutes=HORIZON),x.index[val[0]])
            self.assertEqual(len(val),1200)

    def test_selected_candidate_is_training_cv_winner(self):
        for target in TARGETS:
            group = self.board.loc[self.board.target.eq(target)].sort_values(['mean_cv_mae_pp','mean_cv_rmse_pp','candidate'])
            self.assertEqual(group.candidate.iloc[0], self.summary['targets'][target]['candidate'])
            self.assertAlmostEqual(group.mean_cv_mae_pp.iloc[0],self.summary['targets'][target]['cv_mae_pp'],places=10)

    def test_cv_metrics_have_all_candidates_and_folds(self):
        self.assertEqual(len(self.fold_rows),len(TARGETS)*len(CANDIDATES)*3)
        for (target,candidate),group in self.fold_rows.groupby(['target','candidate']):
            self.assertEqual(set(group.fold),{1,2,3})
            self.assertEqual(len(group),3)

    def test_test_coverage_and_alignment(self):
        for target in TARGETS:
            one=self.test_predictions.loc[(self.test_predictions.target==target)&(self.test_predictions.candidate==self.summary['targets'][target]['candidate'])]
            self.assertEqual(len(one),len(self.test)-5)
            origins=pd.to_datetime(one.origin);targets=pd.to_datetime(one.target_timestamp)
            self.assertTrue((targets.reset_index(drop=True)-origins.reset_index(drop=True)).eq(pd.Timedelta(minutes=5)).all())
            self.assertEqual(origins.iloc[0],self.test.timestamp.iloc[0])
            self.assertEqual(targets.iloc[-1],self.test.timestamp.iloc[-1])

    def test_saved_test_actuals_are_original_rows(self):
        source=self.whole.set_index('timestamp')
        for target in TARGETS:
            group=self.test_predictions.loc[self.test_predictions.target.eq(target)]
            expected=source[target].reindex(pd.to_datetime(group.target_timestamp)).to_numpy()
            np.testing.assert_array_equal(group.actual.to_numpy(),expected)

    def test_selected_predictions_match_reloaded_models(self):
        with threadpool_limits(limits=1):
            for target in TARGETS:
                model=KPIForecaster(MODELS/target)
                group=self.test_predictions.loc[(self.test_predictions.target==target)&(self.test_predictions.candidate==model.model.name)]
                features=build_features(self.whole,target).loc[pd.DatetimeIndex(pd.to_datetime(group.origin))]
                np.testing.assert_allclose(model.model.predict(features),group.prediction,rtol=0,atol=1e-10)

    def test_reported_test_metrics_recompute(self):
        for _,row in self.test_metrics.iterrows():
            if row.candidate not in ('persistence',self.summary['targets'][row.target]['candidate']):
                continue
            group=self.test_predictions.loc[(self.test_predictions.target==row.target)&(self.test_predictions.candidate==row.candidate)]
            actual=group.actual.to_numpy();predicted=group.prediction.to_numpy()
            result=metrics(actual,predicted)
            for key in ('mae_pp','rmse_pp','bias_pp'):
                self.assertAlmostEqual(result[key],row[key],places=9)

    def test_predictions_are_bounded_and_finite(self):
        self.assertTrue(np.isfinite(self.test_predictions.prediction).all())
        self.assertTrue(self.test_predictions.prediction.between(0,100).all())
        self.assertTrue(self.test_predictions.actual.between(0,100).all())

    def test_selected_model_never_updates_from_test_labels(self):
        for target in TARGETS:
            model=KPIForecaster(MODELS/target)
            self.assertEqual(model.metadata['trained_target_through'],self.train.timestamp.iloc[-1].isoformat())
            self.assertEqual(model.metadata['training_source_sha256'],file_sha256(DATA/'infrastructure_metrics_train.csv'))
            self.assertEqual(self.summary['test_access'],'after selecting all targets on training-only expanding chronological folds')

    def test_saved_inference_is_causal_and_same_after_future_modifications(self):
        origin=self.test.timestamp.iloc[50]
        for target in TARGETS:
            model=KPIForecaster(MODELS/target)
            a=model.predict(self.whole,as_of=origin)
            changed=self.whole.copy()
            changed.loc[changed.timestamp.gt(origin),target]=100
            b=model.predict(changed,as_of=origin)
            self.assertEqual(a,b)
            self.assertEqual(a['horizon_minutes'],5)
            self.assertEqual(a['history_rows_used'],HISTORY_ROWS)

    def test_inference_does_not_return_future_actuals(self):
        model=KPIForecaster(MODELS/'cpu_utilization')
        response=model.predict(self.whole)
        self.assertNotIn('actual',response)
        self.assertEqual(pd.Timestamp(response['target_timestamp']),self.whole.timestamp.iloc[-1]+pd.Timedelta(minutes=5))

    def test_train_interval_backcasting_is_rejected(self):
        model=KPIForecaster(MODELS/'cpu_utilization')
        with self.assertRaisesRegex(ValueError,'training-target cutoff'):
            model.predict(self.train,as_of=self.train.timestamp.iloc[-1])

    def test_invalid_and_missing_inputs_rejected(self):
        for column,value in [('cpu_utilization',101),('neft_success_rate',-1),('rtgs_success_rate',np.nan)]:
            frame=self.train.iloc[:30].copy();frame.loc[2,column]=value
            with self.subTest(column=column),self.assertRaises(ValueError):validate_history(frame)
        with self.assertRaises(ValueError):validate_history(self.train.drop(columns='imps_success_rate'))
        with self.assertRaises(ValueError):build_features(self.train,'unknown')

    def test_timestamp_gaps_duplicates_and_timezones_rejected(self):
        frames=[]
        gap=self.train.iloc[:50].drop(index=3);frames.append(gap)
        dup=self.train.iloc[:50].copy();dup.loc[3,'timestamp']=dup.loc[2,'timestamp'];frames.append(dup)
        zone=self.train.iloc[:50].copy();zone['timestamp']=zone.timestamp.dt.tz_localize('UTC');frames.append(zone)
        for frame in frames:
            with self.assertRaises(ValueError):validate_history(frame)

    def test_corrupt_artifact_refused_before_joblib_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);src=MODELS/'cpu_utilization'
            shutil.copy(src/'metadata.json',dest/'metadata.json')
            (dest/'model.joblib').write_bytes((src/'model.joblib').read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):
                KPIForecaster(dest)

    def test_code_and_package_mismatches_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);src=MODELS/'cpu_utilization';shutil.copy(src/'model.joblib',dest/'model.joblib')
            original=json.loads((src/'metadata.json').read_text())
            for field in ('code_sha256','runtime_versions'):
                changed=dict(original)
                changed[field]='wrong' if field=='code_sha256' else {'numpy':'0.0'}
                (dest/'metadata.json').write_text(json.dumps(changed))
                with self.subTest(field=field),self.assertRaises(ValueError):KPIForecaster(dest)

    def test_real_cli_matches_direct_inference(self):
        target='neft_success_rate'; origin=self.test.timestamp.iloc[0]
        p=subprocess.run([sys.executable,str(ROOT/'scripts/predict_kpi.py'), '--target',target,
                          '--history',str(DATA/'infrastructure_metrics.csv'),
                          '--model-dir',str(MODELS),'--as-of',str(origin)],
                         capture_output=True,text=True,check=True,timeout=30)
        actual=json.loads(p.stdout)
        expected=KPIForecaster(MODELS/target).predict(self.whole,as_of=origin)
        self.assertEqual(actual,expected)

    def test_training_refuses_existing_output_paths(self):
        before={p.name:file_sha256(p) for p in (MODELS/'cpu_utilization').iterdir()}
        with self.assertRaisesRegex(ValueError,'new output directories'):
            train_all(DATA,MODELS,REPORTS)
        self.assertEqual(before,{p.name:file_sha256(p) for p in (MODELS/'cpu_utilization').iterdir()})

    def test_retraining_reproduces_model_selection_and_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir=Path(tmp)/'models'; report_dir=Path(tmp)/'reports'
            same=train_all(DATA,model_dir,report_dir)
            for target in TARGETS:
                self.assertEqual(same['targets'][target],self.summary['targets'][target])
                self.assertEqual(KPIForecaster(model_dir/target).predict(self.whole,as_of=self.test.timestamp.iloc[0]),
                                 KPIForecaster(MODELS/target).predict(self.whole,as_of=self.test.timestamp.iloc[0]))

    def test_other_target_values_cannot_change_features(self):
        prefix=self.whole.iloc[:2000].copy()
        baseline=build_features(prefix,'neft_success_rate')
        prefix['cpu_utilization']=100
        prefix['memory_usage']=100
        updated=build_features(prefix,'neft_success_rate')
        pd.testing.assert_frame_equal(baseline,updated)

    def test_test_data_changes_cannot_change_selected_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/'data';source.mkdir()
            shutil.copy(DATA/'infrastructure_metrics_train.csv',source/'infrastructure_metrics_train.csv')
            changed=pd.read_csv(DATA/'infrastructure_metrics_test.csv')
            for target in TARGETS:
                changed[target]=100.0
            changed.to_csv(source/'infrastructure_metrics_test.csv',index=False)
            result=train_all(source,root/'models',root/'reports')
            for target in TARGETS:
                self.assertEqual(result['targets'][target]['candidate'],
                                 self.summary['targets'][target]['candidate'])
                self.assertEqual(result['targets'][target]['cv_mae_pp'],
                                 self.summary['targets'][target]['cv_mae_pp'])

    def test_sources_unchanged(self):
        self.assertEqual(self.orig,{p.name:file_sha256(p) for p in DATA.glob('*.csv')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/raw')
    p.add_argument('--model-dir',type=Path,default=ROOT/'models/kpi_5min')
    p.add_argument('--report-dir',type=Path,default=ROOT/'reports/kpi_forecast')
    args,rest=p.parse_known_args()
    DATA,MODELS,REPORTS=args.data_dir,args.model_dir,args.report_dir
    unittest.main(argv=[sys.argv[0],*rest],verbosity=2)
