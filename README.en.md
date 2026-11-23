[한국어](README.md) | **English**

![Static Badge](https://img.shields.io/badge/python-3.10-blue)

# <center> Wind Power Prediction (WPP) Library </center>
### <center> Python Library of predicting wind Power. </center>

------------
# Update
- v2.0.0 release (unified package, YAML settings, PyTorch DNN, in-memory data merging)
- v1.0.0 2024-03-19 Release Minwoo Baek.
------------

# Abstract

- Wind power generation is one of the most important renewable energy sources.
Although predicting the amount of power generation is crucial for efficient operations, it is not easy because of fluctuating nature of wind speed.
This paper applies a deep neural network method to predicting wind power generation based on weather forecast data.
Wind power generation data were collected from a power plant located in Jeju, South Korea, and weather forecast data for the nearby weather stations were collected.

----

# Project structure

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

# How to run

## Installation

From the repository root, install the package with `python -m pip install -e .` to keep it linked to this checkout. For DNN training on Windows CPU, install PyTorch separately with `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` (verified with `2.15.0+cpu`).

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

## Input data

`raw_data` points to the directory containing the `wind/` workbooks and `weather/<site>/<year>/` forecast CSVs. The default layout follows the original scripts. Workbook paths, sheet numbers, weather sites, and years are defined in [`wind_power_predictor/configs/experiment.yaml`](wind_power_predictor/configs/experiment.yaml).

## Commands

```sh
python -m wind_power_predictor.scripts.train_ml ./dataset/rawdata hk1 --history 0
python -m wind_power_predictor.scripts.train_dnn ./dataset/rawdata hk1 --history 0
```

Edit [`wind_power_predictor/configs/experiment.yaml`](wind_power_predictor/configs/experiment.yaml) to change the experiment settings before running a command. `--history` selects one configured number of past power generation values. Omitting it runs all configured values, h0 through h5 by default. `--output` overrides the configured destination. Pass `--feature-importance` to `train_ml` to run the XGBoost feature importance experiment.

Run the regression tests with `python -m unittest discover -s wind_power_predictor/tests -v`.

------------

# Article

- Title : Wind power generation prediction based on weather forecast data using deep neural networks
- Author : Min-Woo Baek, Min Kyu Sim, Jae-Yoon Jung
- Date : 2020.09
- http://www.icicelb.org/ellb/contents/2020/9/elb-11-09-07.pdf
