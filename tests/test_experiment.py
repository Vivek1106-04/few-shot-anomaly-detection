"""Experiment-runner tests (S9). The runner is checked against the miniature
dataset with a *fake* extractor, so the properties under test are the runner's own:
the cache, the loop structure, the provenance written next to every number, and the
latency accounting."""

from __future__ import annotations

import json

import numpy as np
import pytest

from config.schema import RunConfig
from data.mvtec import MVTecAD
from experiment import FeatureCache, RunRecord, run_single, run_sweep, summarise, write_results

CHANNELS = 8
GRID = 4


class CountingExtractor:
    """A stand-in for S2: deterministic features, and it counts what it was asked for."""

    def __init__(self) -> None:
        self.calls = 0
        self.images_seen = 0

    def extract(self, images: np.ndarray) -> np.ndarray:
        images = np.asarray(images, dtype=np.float32)
        if images.ndim == 3:
            images = images[None]
        self.calls += 1
        self.images_seen += len(images)
        # A feature that depends on the image content, so different images differ.
        summary = images.reshape(len(images), 3, -1).mean(axis=2)
        rng = np.random.default_rng(0)
        basis = rng.normal(size=(3, CHANNELS, GRID, GRID)).astype(np.float32)
        return np.einsum("nc,cdhw->ndhw", summary, basis)


@pytest.fixture
def dataset(mvtec_root) -> MVTecAD:
    return MVTecAD(mvtec_root)


@pytest.fixture
def config() -> RunConfig:
    return RunConfig(
        categories=("widget",),
        methods=("patchcore", "padim"),
        k_values=(1, 2),
        n_draws=2,
        seed=99,
    )


def toy_parameters(method: str) -> dict[str, object]:
    """Defaults tuned for a 28x28 grid are degenerate on the fixture's 4x4 one."""
    return ({"coreset_ratio": 1.0, "neighbourhood": 1} if method == "patchcore"
            else {"dim": CHANNELS})


# --- feature cache ----------------------------------------------------------


def test_the_cache_extracts_each_image_exactly_once(dataset):
    # Arrange
    extractor = CountingExtractor()
    cache = FeatureCache(dataset=dataset, extractor=extractor)
    samples = dataset.train_samples("widget")[:3]

    # Act
    first = cache.features(samples)
    second = cache.features(samples)

    # Assert
    assert extractor.images_seen == 3
    assert np.array_equal(first, second)


def test_the_cache_returns_features_in_the_order_requested(dataset):
    # Arrange
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())
    samples = dataset.test_samples("widget")[:4]

    # Act
    forwards = cache.features(samples)
    backwards = cache.features(list(reversed(samples)))

    # Assert
    assert np.array_equal(forwards, backwards[::-1])


def test_the_cache_times_only_the_images_it_actually_extracted(dataset):
    # Arrange
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())
    samples = dataset.train_samples("widget")[:2]

    # Act
    cache.features(samples)
    before = cache.ms_per_image
    cache.features(samples)  # served from the cache: must not count as a measurement

    # Assert
    assert cache.n_extracted == 2
    assert cache.ms_per_image == before


def test_an_empty_cache_reports_no_latency(dataset):
    assert FeatureCache(dataset=dataset, extractor=CountingExtractor()).ms_per_image == 0.0


def test_clearing_the_cache_frees_the_stored_features(dataset):
    # Arrange
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())
    cache.features(dataset.train_samples("widget")[:2])

    # Act
    cache.clear()

    # Assert
    assert cache.store == {}


# --- one run ----------------------------------------------------------------


def test_a_run_records_the_support_set_that_produced_it(dataset, config):
    # Arrange
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())

    # Act
    record = run_single(cache, "patchcore", "widget", 2, 0, config, toy_parameters("patchcore"))

    # Assert
    assert record.k == 2
    assert len(record.provenance["image_ids"]) == 2
    assert record.provenance["mode"] == "uniform"
    assert record.provenance["seed"] != config.seed  # the derived per-draw seed


def test_the_same_draw_twice_gives_the_same_numbers(dataset, config):
    # Arrange
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())

    # Act
    first = run_single(cache, "padim", "widget", 2, 1, config, toy_parameters("padim"))
    second = run_single(cache, "padim", "widget", 2, 1, config, toy_parameters("padim"))

    # Assert
    assert first.result.as_dict() == second.result.as_dict()


def test_different_draws_give_different_support_sets(dataset, config):
    # Arrange
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())

    # Act
    first = run_single(cache, "padim", "widget", 2, 0, config, toy_parameters("padim"))
    second = run_single(cache, "padim", "widget", 2, 1, config, toy_parameters("padim"))

    # Assert
    assert first.provenance["image_ids"] != second.provenance["image_ids"]


def test_latency_is_the_backbone_cost_plus_the_method_cost(dataset, config):
    # Arrange
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())

    # Act
    record = run_single(cache, "patchcore", "widget", 1, 0, config, toy_parameters("patchcore"))

    # Assert
    assert record.latency_ms == pytest.approx(
        record.extract_ms_per_image + record.score_ms_per_image
    )
    assert record.score_ms_per_image > 0.0


def test_a_record_serialises_its_metrics_and_its_method_parameters(dataset, config):
    # Arrange
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())

    # Act
    payload = run_single(
        cache, "patchcore", "widget", 1, 0, config, toy_parameters("patchcore")
    ).as_dict()

    # Assert
    assert payload["method"] == "patchcore"
    assert "I-AUROC" in payload["metrics"]
    assert payload["method_parameters"]["coreset_ratio"] == 1.0


# --- the sweep --------------------------------------------------------------


@pytest.fixture
def records(dataset, config, monkeypatch) -> list[RunRecord]:
    """The sweep, with both methods forced onto the fixture-sized parameters."""
    import experiment

    original = experiment.build_method
    monkeypatch.setattr(
        experiment,
        "build_method",
        lambda name, **kwargs: original(name, **{**toy_parameters(name), **kwargs}),
    )
    return run_sweep(dataset, CountingExtractor(), config, verbose=False)


def test_the_sweep_runs_every_method_k_and_draw(records):
    # Assert: 2 methods x 2 k values x 2 draws
    assert len(records) == 8
    assert {record.method for record in records} == {"patchcore", "padim"}
    assert {record.k for record in records} == {1, 2}


def test_the_sweep_scores_both_methods_on_identical_features(records):
    """The cache is shared, so a difference between methods is the method (Gap G1)."""
    # Arrange
    by_method = {
        (record.method, record.k, record.draw): record.provenance["image_ids"]
        for record in records
    }

    # Assert
    for k in (1, 2):
        for draw in (0, 1):
            assert by_method[("patchcore", k, draw)] == by_method[("padim", k, draw)]


def test_the_ceiling_run_is_drawn_only_once(dataset, monkeypatch):
    """k = full has a single deterministic support set; repeating it adds nothing."""
    # Arrange
    import experiment

    original = experiment.build_method
    monkeypatch.setattr(
        experiment,
        "build_method",
        lambda name, **kwargs: original(name, **{**toy_parameters(name), **kwargs}),
    )
    config = RunConfig(categories=("widget",), methods=("padim",), k_values=("full",), n_draws=3)

    # Act
    records = run_sweep(dataset, CountingExtractor(), config, verbose=False)

    # Assert
    assert len(records) == 1


def test_a_k_larger_than_the_train_pool_is_rejected(dataset):
    # Arrange
    config = RunConfig(categories=("widget",), methods=("padim",), k_values=(999,), n_draws=1)

    # Act / Assert
    with pytest.raises(ValueError, match="exceeds"):
        run_sweep(dataset, CountingExtractor(), config, verbose=False)


def test_the_summary_reports_mean_and_spread_per_method_category_and_k(records):
    # Act
    summary = summarise(records)

    # Assert
    entry = summary["patchcore"]["widget"]["2"]
    assert entry["n_draws"] == 2
    assert len(entry["I-AUROC"]) == 2  # (mean, std)
    assert entry["latency_ms"] > 0.0


def test_the_progress_table_names_the_method_category_and_k(dataset, config, capsys, monkeypatch):
    # Arrange
    import experiment

    original = experiment.build_method
    monkeypatch.setattr(
        experiment,
        "build_method",
        lambda name, **kwargs: original(name, **{**toy_parameters(name), **kwargs}),
    )
    one_run = RunConfig(categories=("widget",), methods=("padim",), k_values=(1,), n_draws=1)

    # Act
    run_sweep(dataset, CountingExtractor(), one_run, verbose=True)

    # Assert
    printed = capsys.readouterr().out
    assert "padim" in printed and "widget" in printed


def test_the_result_file_holds_the_config_the_summary_and_every_run(tmp_path, records, config):
    # Arrange
    path = tmp_path / "results.json"

    # Act
    write_results(path, config, records)
    payload = json.loads(path.read_text())

    # Assert
    assert payload["config"]["seed"] == config.seed
    assert len(payload["runs"]) == len(records)
    assert payload["summary"]["padim"]["widget"]["1"]["n_draws"] == 2


# --- the command line -------------------------------------------------------


def test_main_runs_the_real_backbone_end_to_end(mvtec_root, tmp_path, monkeypatch, capsys):
    """The one test that wires S1 -> S2 -> S3/S4 -> S5/S6 -> S10 together for real."""
    pytest.importorskip("torch")
    import sys

    import experiment

    # Arrange
    output = tmp_path / "results.json"
    monkeypatch.setattr(sys, "argv", [
        "experiment", "--data-root", str(mvtec_root), "--categories", "widget",
        "--methods", "patchcore", "padim", "--k", "1", "2", "--draws", "1",
        "--output", str(output),
    ])

    # Act
    experiment.main()

    # Assert
    payload = json.loads(output.read_text())
    assert {run["method"] for run in payload["runs"]} == {"patchcore", "padim"}
    assert payload["config"]["data_root"] == str(mvtec_root)
    assert "I-AUROC" in capsys.readouterr().out


def test_build_extractor_returns_the_backbone_named_by_the_config():
    pytest.importorskip("torch")
    from experiment import build_extractor
    from models.backbone import FrozenBackbone

    # Act
    extractor = build_extractor(RunConfig(backbone="wide_resnet50_2"))

    # Assert
    assert isinstance(extractor, FrozenBackbone)
    assert extractor.name == "wide_resnet50_2"


# --- Task 6: S7 decision, S11 stress modes, augmentation, memory, text ------


def test_a_run_fits_the_threshold_on_held_out_normals_and_reports_the_operating_point(
        dataset, config):
    # Arrange
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())

    # Act
    record = run_single(cache, "patchcore", "widget", 2, 0, config, toy_parameters("patchcore"))

    # Assert
    assert record.decision["n_validation"] == 10  # 12 train normals minus the 2 in support
    assert set(record.decision) >= {"threshold", "pixel_threshold", "meets_target"}
    assert 0.0 <= record.result.decision_recall <= 1.0
    assert 0.0 <= record.result.decision_fpr <= 1.0
    assert 0.0 <= record.result.image_auroc_robust <= 1.0


def test_the_ceiling_run_has_no_threshold_and_says_so(dataset, config):
    # Arrange
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())

    # Act
    record = run_single(cache, "padim", "widget", "full", 0, config, toy_parameters("padim"))

    # Assert
    assert record.decision is None
    assert np.isnan(record.result.decision_recall)
    assert record.stress == {}


def test_memory_is_measured_on_the_first_draw_only(dataset, config):
    # Arrange
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())

    # Act
    first = run_single(cache, "padim", "widget", 2, 0, config, toy_parameters("padim"))
    second = run_single(cache, "padim", "widget", 2, 1, config, toy_parameters("padim"))

    # Assert
    assert first.memory["reference_mb"] > 0.0
    assert first.memory["enrol_peak_mb"] >= 0.0 and "score_peak_mb" in first.memory
    assert second.memory is None


def test_a_contaminated_run_never_scores_its_contaminant_and_measures_recall_by_type(dataset):
    # Arrange
    config = RunConfig(categories=("widget",), methods=("patchcore",), k_values=(3,),
                       n_draws=1, support_mode="contaminated", n_contaminated=1)
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())

    # Act
    record = run_single(cache, "patchcore", "widget", 3, 0, config, toy_parameters("patchcore"))

    # Assert
    assert len(record.provenance["contaminated_ids"]) == 1
    assert record.provenance["mode"] == "contaminated"
    assert record.stress["n_same_type"] == 2  # 3 of that defect type, minus the contaminant
    assert 0.0 <= record.stress["same_type_recall"] <= 1.0


def test_a_coverage_gap_run_measures_false_alarms_on_the_withheld_group(dataset, monkeypatch):
    # Arrange: the fixture's 12 normals are one 20-id block, so group them in fives
    import stress

    original = stress.id_block_group
    monkeypatch.setattr(stress, "id_block_group", lambda sample: original(sample, 5))
    config = RunConfig(categories=("widget",), methods=("padim",), k_values=(2,), n_draws=1,
                       support_mode="coverage_gap")
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())

    # Act
    record = run_single(cache, "padim", "widget", 2, 0, config, toy_parameters("padim"))

    # Assert
    held_out = record.provenance["held_out_group"]
    assert record.provenance["mode"] == "coverage_gap"
    assert record.stress["held_out_group"] == held_out
    assert record.stress["n_held_out"] >= 1
    assert 0.0 <= record.stress["held_out_fpr"] <= 1.0
    in_support = {image_id.rsplit("/", 1)[1] for image_id in record.provenance["image_ids"]}
    assert all(f"block{int(stem) // 5}" != held_out for stem in in_support)


def test_augmented_support_adds_the_copies_to_the_reference(dataset, config):
    # Arrange
    extractor = CountingExtractor()
    cache = FeatureCache(dataset=dataset, extractor=extractor)
    samples = dataset.train_samples("widget")[:2]

    # Act
    plain = cache.support_features(samples, 0, seed=3)
    augmented = cache.support_features(samples, 3, seed=3)

    # Assert
    assert plain.shape[0] == 2
    assert augmented.shape[0] == 2 + 2 * 3
    assert np.array_equal(augmented[:2], plain)


def test_an_augmented_run_records_the_augmentation(dataset):
    # Arrange
    config = RunConfig(categories=("widget",), methods=("patchcore",), k_values=(1,),
                       n_draws=1, augment=2)
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())

    # Act
    record = run_single(cache, "patchcore", "widget", 1, 1, config, toy_parameters("patchcore"))

    # Assert
    assert record.method_parameters["augment"] == 2


class TextExtractor(CountingExtractor):
    """A fake CLIP: two trailing 'language' channels and prompt embeddings."""

    def __init__(self) -> None:
        super().__init__()
        self.text_calls = 0

    def text_embeddings(self, category: str) -> np.ndarray:
        self.text_calls += 1
        return np.eye(2, 2, dtype=np.float32)


def test_a_text_method_gets_the_category_prompts_once_per_category(dataset):
    # Arrange
    extractor = TextExtractor()
    config = RunConfig(categories=("widget",), methods=("winclip",), k_values=(1,), n_draws=2,
                       method_parameters={"winclip": {"language_dim": 2, "n_visual_blocks": 1}})
    cache = FeatureCache(dataset=dataset, extractor=extractor)

    # Act
    first = run_single(cache, "winclip", "widget", 1, 0, config)
    run_single(cache, "winclip", "widget", 1, 1, config)

    # Assert
    assert extractor.text_calls == 1
    assert first.method_parameters["language_dim"] == 2  # config parameters reached it


def test_a_text_method_on_an_image_only_backbone_fails_loudly(dataset, config):
    # Arrange
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())

    # Act / Assert
    with pytest.raises(ValueError, match="text-aligned backbone"):
        run_single(cache, "winclip", "widget", 1, 0, config)


def test_the_summary_averages_memory_and_stress_measurements(dataset):
    # Arrange
    config = RunConfig(categories=("widget",), methods=("patchcore",), k_values=(3,),
                       n_draws=2, support_mode="contaminated")
    cache = FeatureCache(dataset=dataset, extractor=CountingExtractor())
    records = [run_single(cache, "patchcore", "widget", 3, draw, config,
                          toy_parameters("patchcore")) for draw in range(2)]

    # Act
    entry = summarise(records)["patchcore"]["widget"]["3"]

    # Assert
    assert entry["memory"]["reference_mb"] > 0.0  # from draw 0 only
    assert "same_type_recall" in entry["stress"]
    assert "held_out_group" not in entry.get("stress", {})


def test_mean_of_ignores_missing_and_undefined_values():
    from experiment import _mean_of

    assert _mean_of([1.0, None, float("nan"), 3.0]) == 2.0
    assert np.isnan(_mean_of([None]))


def test_parameters_parse_from_the_command_line_form():
    from experiment import parse_parameters

    # Act
    parsed = parse_parameters(["winclip.language_weight=1.0", "patchcore.coreset_ratio=0.25"])

    # Assert
    assert parsed == {"winclip": {"language_weight": 1.0}, "patchcore": {"coreset_ratio": 0.25}}


def test_a_malformed_parameter_is_rejected():
    from experiment import parse_parameters

    with pytest.raises(ValueError, match="method.parameter=value"):
        parse_parameters(["winclip=1"])


def test_build_dataset_selects_the_benchmark_named_by_the_config(mvtec_root, visa_root):
    from data.visa import VisA
    from experiment import build_dataset

    assert isinstance(build_dataset(RunConfig(dataset="visa", data_root=str(visa_root))), VisA)
    assert type(build_dataset(RunConfig(data_root=str(mvtec_root)))) is MVTecAD


def test_build_extractor_returns_clip_when_asked():
    pytest.importorskip("open_clip")
    from experiment import build_extractor
    from models.clip_backbone import CLIPBackbone

    assert isinstance(build_extractor(RunConfig(backbone="clip_vit_b16")), CLIPBackbone)


def test_main_accepts_the_stress_augmentation_and_parameter_flags(
        mvtec_root, tmp_path, monkeypatch):
    """CLI wiring for the Task 6 flags, with a fake backbone so it stays fast."""
    import sys

    import experiment

    # Arrange
    output = tmp_path / "results.json"
    monkeypatch.setattr(experiment, "build_extractor", lambda config: CountingExtractor())
    monkeypatch.setattr(sys, "argv", [
        "experiment", "--data-root", str(mvtec_root), "--categories", "widget",
        "--methods", "patchcore", "--k", "3", "--draws", "1", "--mode", "contaminated",
        "--augment", "1", "--param", "patchcore.coreset_ratio=1.0", "patchcore.neighbourhood=1",
        "--dataset", "mvtec_ad", "--tag", "stress", "--threshold-rule", "conformal",
        "--output", str(output),
    ])

    # Act
    experiment.main()

    # Assert
    payload = json.loads(output.read_text())
    assert payload["config"]["support_mode"] == "contaminated"
    assert payload["config"]["augment"] == 1
    assert payload["config"]["threshold_rule"] == "conformal"
    assert payload["runs"][0]["method_parameters"]["coreset_ratio"] == 1.0
    assert payload["runs"][0]["method"] == "patchcore+stress"
