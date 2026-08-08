from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from code.langtang_pipeline import prepare_langtang_hybrid_dataset
from code.real_data_pipeline import RealDataConfig, RealMicrogridDataPipeline


def test_icimod_langtang_source_produces_leakage_safe_real_windows(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "data"
        / "raw"
        / "icimod_langtang_khola"
        / "data"
        / "LangtangKholaDischarge_Kyanjing.csv"
    )
    hybrid_path = tmp_path / "langtang_hybrid.csv"
    provenance_path = tmp_path / "provenance.json"

    hybrid, provenance = prepare_langtang_hybrid_dataset(
        source,
        hybrid_path,
        provenance_path,
        max_linear_gap_hours=6,
    )

    selected = provenance["selected_contiguous_block"]
    assert selected["rows"] == len(hybrid)
    assert selected["rows"] > 365 * 24
    assert selected["short_gap_imputed_flow_rows"] == 7
    assert provenance["source"]["doi"] == "10.26066/RDS.1972407"
    assert np.isfinite(hybrid["river_flow_m3s"]).all()
    assert hybrid["timestamp"].is_monotonic_increasing

    prepared = RealMicrogridDataPipeline(
        RealDataConfig(max_imputed_fraction=0.01)
    ).prepare(hybrid_path)
    assert prepared.feature_names == (
        "river_flow_m3s",
        "critical_load_kw",
        "noncritical_load_kw",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    )
    for values in (
        prepared.x_train,
        prepared.y_train,
        prepared.x_validation,
        prepared.y_validation,
        prepared.x_test,
        prepared.y_test,
    ):
        assert np.isfinite(values).all()

    train_end = prepared.split_boundaries["train_end"]
    validation_end = prepared.split_boundaries["validation_end"]
    train_boundary = (
        prepared.cleaned_frame.index[train_end].tz_convert("UTC").tz_localize(None)
    )
    validation_boundary = (
        prepared.cleaned_frame.index[validation_end]
        .tz_convert("UTC")
        .tz_localize(None)
    )
    assert prepared.train.target_end_timestamps[-1] < np.datetime64(train_boundary)
    assert prepared.validation.target_start_timestamps[0] >= np.datetime64(
        train_boundary
    )
    assert prepared.validation.target_end_timestamps[-1] < np.datetime64(
        validation_boundary
    )
    assert prepared.test.target_start_timestamps[0] >= np.datetime64(
        validation_boundary
    )
