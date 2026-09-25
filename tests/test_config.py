"""Run-configuration tests: a typo in a config must fail, never be ignored."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.schema import RunConfig, load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "src" / "config" / "default.yaml"


def test_default_config_file_loads():
    # Act
    config = load_config(DEFAULT_CONFIG)

    # Assert
    assert config.dataset == "mvtec_ad"
    assert config.k_values == (1, 2, 4, 8, 16, "full")
    assert config.n_draws >= 1


def test_unknown_key_is_rejected(tmp_path: Path):
    # Arrange: "methd" instead of "methods"
    path = tmp_path / "typo.yaml"
    path.write_text("methd: [padim]\n", encoding="utf-8")

    # Act / Assert
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(path)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        RunConfig(methods=("spade",))


def test_unknown_backbone_is_rejected():
    with pytest.raises(ValueError, match="unknown backbone"):
        RunConfig(backbone="resnet9000")


def test_unknown_support_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown support_mode"):
        RunConfig(support_mode="whatever")


def test_non_positive_draw_count_is_rejected():
    with pytest.raises(ValueError, match="n_draws"):
        RunConfig(n_draws=0)


def test_bad_k_value_is_rejected():
    with pytest.raises(ValueError, match="k values"):
        RunConfig(k_values=(0,))


def test_bad_fpr_budget_is_rejected():
    with pytest.raises(ValueError, match="fpr_budget"):
        RunConfig(fpr_budget=1.5)


def test_config_is_serialisable_for_the_result_file():
    # Act
    config = RunConfig()

    # Assert: every run writes this next to its metrics
    assert config.as_dict()["seed"] == config.seed


def test_non_mapping_config_is_rejected(tmp_path: Path):
    # Arrange
    path = tmp_path / "list.yaml"
    path.write_text("- padim\n", encoding="utf-8")

    # Act / Assert
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config(path)


@pytest.mark.parametrize("overrides, message", [
    ({"dataset": "imagenet"}, "unknown dataset"),
    ({"n_validation": 0}, "n_validation"),
    ({"augment": -1}, "augment"),
    ({"method_parameters": {"magic": {}}}, "unknown method"),
    ({"threshold_rule": "vibes"}, "threshold_rule"),
])
def test_the_task6_fields_are_validated(overrides, message):
    with pytest.raises(ValueError, match=message):
        RunConfig(**overrides)


def test_visa_and_per_method_parameters_are_accepted():
    # Act
    config = RunConfig(dataset="visa", methods=("winclip",),
                       method_parameters={"winclip": {"language_weight": 1.0}})

    # Assert
    assert config.method_parameters["winclip"]["language_weight"] == 1.0
