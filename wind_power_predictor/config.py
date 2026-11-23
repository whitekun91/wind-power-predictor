from importlib.resources import files

import yaml


# 실행 디렉터리와 관계없이 패키지에 포함된 실험 설정을 읽습니다.
_CONFIG_PATH = files(__package__) / "configs" / "experiment.yaml"
with _CONFIG_PATH.open(encoding="utf-8") as config_file:
    SETTINGS = yaml.safe_load(config_file)

if not isinstance(SETTINGS, dict):
    raise ValueError(f"Expected a YAML mapping in {_CONFIG_PATH}")

PLANTS = tuple(SETTINGS["data"]["plants"])
DEFAULT_OUTPUT_PATH = SETTINGS["reporting"]["output_path"]
