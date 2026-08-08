# Langtang Khola Real-Flow Microgrid Digital Twin

This package accompanies the paper **“Edge-Computed Deep Learning for
Hydrological Forecasting and Load Balancing in an Isolated Run-of-River
Micro-Grid: A Langtang Khola Real-Flow Digital-Twin Study.”**

The hydrological source is the ICIMOD/Utrecht University hourly discharge
archive for Langtang Khola at Kyanjing (28.21081° N, 85.56169° E, 3,850 m).
The full file contains 62,388 hourly rows from 2012–2019. Because 43.86% of
the discharge grid is missing, the primary pipeline interpolates only internal
gaps of at most six hours and selects the longest remaining complete block:
15,732 hours from 17 October 2016 to 3 August 2018. River flow is measured;
the critical/non-critical loads and plant parameters are explicitly simulated.

Authors: Roshan Kumar Sharma and Ujjwal Prasad Dahal, Department of Electrical
Engineering, Tribhuvan University, Institute of Engineering, Purwanchal
Campus, Dharan-8, Nepal.

## Reproduce the primary experiment

From the project root on Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
.\.venv\Scripts\python.exe -m code.langtang_pipeline
.\.venv\Scripts\python.exe -m pytest -q
```

The run creates leakage-safe 72-to-24-hour chronological windows, trains the
3,344-parameter LSTM, applies selective quantization-aware fine-tuning, exports
a strict INT8 TensorFlow Lite model, evaluates persistence, and runs the
observed low-flow hybrid-control test.

Reference results:

- Forecast test: 2,265 origins and 54,360 origin-lead targets.
- INT8 MAE/RMSE: 0.5374/0.8442 m³/s; R²: 0.9686.
- Persistence MAE: 0.7797 m³/s; INT8 MAE improvement: 31.08%.
- INT8 flatbuffer: 8.96 KB; conservative model-plus-tensor screen: 14.95 KB.
- Driest 96-hour test window: MPC critical reliability 100%; comparator
  reliability 72.92%; 26 comparator critical-shortfall hours avoided.

## Generic historical CSV interface

`code/real_data_pipeline.py` accepts hourly measured flow, critical load, and
non-critical load. Optional precipitation, air temperature, and snow-water
equivalent channels are included when present. It performs timezone
normalization, physical/stale-data checks, causal imputation, training-only
scaling, and disjoint chronological windowing.

```powershell
.\.venv\Scripts\python.exe -m code.real_data_pipeline historical.csv `
  --output results\real_microgrid_sequences.npz `
  --report results\real_microgrid_quality_report.json
```

## Package layout

- `code/`: ICIMOD ingestion, generic real-CSV preprocessing, QAT-LSTM,
  full-INT8 export, micro-hydro simulator, MPC, metrics, and plots.
- `tests/`: ten deterministic unit and integration tests.
- `data/raw/icimod_langtang_khola/`: unchanged source archive contents,
  metadata, station sheet, and data agreement.
- `data/langtang_khola_real_flow_hybrid_load.csv`: selected observed-flow block
  with gap provenance and simulated tiered loads.
- `data/langtang_forecast_test_all_leads.csv`: 54,360 held-out predictions.
- `data/langtang_forecast_leadwise_metrics.csv`: horizon-wise metrics.
- `data/langtang_observed_low_flow_stress.csv`: 96-hour control trace.
- `results/`: metrics JSON and exported INT8 TFLite model.
- `figures/`: nine publication figures; Figures 7–9 are the real-flow results.
- `paper/`: full Word/PDF manuscript, compact journal submission, and LaTeX.

## Deployment qualification

TensorFlow Model Optimization's default QAT registry does not cover Keras LSTM
layers. The implementation uses selective QAT for supported Dense layers,
then representative-data full-integer conversion of the complete graph. INT8
input/output tensors are verified. The 14.95 KB number is a screening estimate,
not a measured ESP32 tensor arena; firmware, operators, stacks, communication
buffers, and sensor drivers must be measured on the target board.

The archive’s included data agreement and attribution must remain with any
redistribution. The 28 m head, 72% efficiency, 120 kW rating, 3.35 m³/s bypass,
critical/non-critical demand, and resulting reliability are hypothetical
control-study assumptions—not measurements of a commissioned Langtang plant.
