<h1 align="center">Wind Power Predictor</h1>
<p align="center">A Jeju case study using weather forecasts and historical power output</p>

<div align="center">

![Jeju-inspired coastal wind farm with spinning turbines, breaking waves, and drifting clouds](assets/readme-banner.gif)

</div>

<p align="center">
  <a href="#how-to-run">How to run</a> &nbsp;|&nbsp;
  <a href="README.md" lang="ko">한국어</a> &nbsp;|&nbsp;
  <a href="assets/readme-banner.webp">Still image</a>
</p>

## How to run

### Installation

Use Python 3.10 or later. Install the package in editable mode from the repository root.

```sh
python -m pip install -e .
```

For DNN training on Windows CPU, install PyTorch separately (verified with `2.15.0+cpu`).

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Input data

Example source paths for Hankyung Area 1 (partial listing)

```text
dataset/rawdata/
├── wind/
│   └── 붙임1_한경풍력 발전실적 정보.xlsx
└── weather/
    └── hankyeong/
        └── 2014/
            └── weather_forecast_temp3h.csv
```

Workbook paths, sheet numbers, weather sites, and years. [Experiment settings](wind_power_predictor/configs/experiment.yaml)

### Commands

```sh
python -m wind_power_predictor.scripts.train_ml ./dataset/rawdata hk1 --history 0
python -m wind_power_predictor.scripts.train_dnn ./dataset/rawdata hk1 --history 0
```

Set data and model parameters before training. [Experiment settings](wind_power_predictor/configs/experiment.yaml)

| Option | Purpose |
| --- | --- |
| `--history` | Number of historical generation inputs (defaults to all configured values, h0 through h5) |
| `--output` | Output directory override (default `predictionresult/`) |
| `--feature-importance` | XGBoost feature importance experiment in `train_ml` |

### Regression tests

```sh
python -m unittest discover -s wind_power_predictor/tests -v
```

### Required libraries

Installation includes these main libraries. [Full dependency list](requirements.txt)

| Library | Version constraint | Purpose |
| --- | --- | --- |
| NumPy | `==1.26.4` | Array operations |
| pandas | `==2.2.1` | Data loading, preprocessing, and merging |
| openpyxl | `>=3.1,<4` | Reading wind generation Excel files |
| PyYAML | `>=6,<7` | Reading experiment settings |
| scikit-learn | `==1.4.1.post1` | Regression, scaling, and evaluation metrics |
| LightGBM | `==4.3.0` | LightGBM regression |
| XGBoost | `>=2,<4` | XGBoost regression and feature importance |
| Matplotlib | `>=3.8,<4` | Prediction plot exports |
| Plotly | `>=5,<8` | Interactive result visualization |

## Overview

- **Forecast inputs** Four-hour-ahead weather forecasts and historical generation at three-hour intervals
- **Study sites** Hankyung Area 1, Hankyung Area 2, and Seongsan wind farms in Jeju

```mermaid
flowchart LR
    A["Power output + forecasts"] --> B["Merge and build features"]
    B --> C["ML regressors / PyTorch DNN"]
    C --> D["Predict and evaluate"]
    classDef data fill:#163845,stroke:#4AB9B3,color:#FFFFFF;
    classDef model fill:#245D62,stroke:#6DD9C5,color:#FFFFFF;
    classDef result fill:#784225,stroke:#E6A76D,color:#FFFFFF;
    class A,B data;
    class C model;
    class D result;
```

## Project structure

| Path | Contents |
| --- | --- |
| [`wind_power_predictor/configs/experiment.yaml`](wind_power_predictor/configs/experiment.yaml) | Data, feature, model, and output settings |
| [`wind_power_predictor/models/`](wind_power_predictor/models) | Prediction models, including ML regressors and the PyTorch DNN |
| [`wind_power_predictor/datasets/`](wind_power_predictor/datasets) | Raw generation and weather readers |
| [`wind_power_predictor/preprocessing/`](wind_power_predictor/preprocessing) | Wind aggregation, forecast selection and time alignment |
| [`wind_power_predictor/features/`](wind_power_predictor/features) | Historical generation features, train/test split and scaling |
| [`wind_power_predictor/reporting/`](wind_power_predictor/reporting) | Prediction exports, feature importance, merged results, and plots |
| [`wind_power_predictor/experiment.py`](wind_power_predictor/experiment.py) | Training workflow for each plant and history setting |
| [`wind_power_predictor/scripts/`](wind_power_predictor/scripts) | ML and DNN training commands |
| [`wind_power_predictor/tests/`](wind_power_predictor/tests) | Regression tests included in the package |
| [`examples/`](examples) | Original research scripts outside the package |

## Original paper

Min-Woo Baek, Min Kyu Sim, and Jae-Yoon Jung (September 2020). *Wind Power Generation Prediction Based on Weather Forecast Data Using Deep Neural Networks*. [Paper PDF](http://www.icicelb.org/ellb/contents/2020/9/elb-11-09-07.pdf)

## Release history

- v2.0.0 release (unified package, YAML settings, PyTorch DNN, in-memory data merging)
- v1.0.0 release (March 19, 2024, Minwoo Baek)
