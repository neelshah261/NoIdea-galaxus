"""#33: the speed matrix — the grid, the cell runner, and the rule that an unrun cell stays empty.

Synthetic images in ``tmp_path`` only, no GPU and no network (AGENTS.md "Work → PR" 5). The GPU cells of the grid
are never run here: this file proves the machinery around them, with the CPU stand-in backend.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from serve_helpers import StubPredictor

from galaxus.data import images
from galaxus.serve import bench as serve_bench

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # `bench/` is a plan, not part of the installed package, so it is imported from the repository root.
    sys.path.insert(0, str(REPO_ROOT))

from bench import fake_backend, gpu_plan, speed_matrix


def named_gpu(price: float | None = None) -> speed_matrix.Hardware:
    """Hardware named the way a human names it on the command line, with or without a price."""
    source = "test fixture, not a real price" if price is not None else ""
    return speed_matrix.Hardware("test CPU, named by hand", price, source)


def fake_cell(area: int = images.IMAGE_SIZE, batch: int = 1) -> speed_matrix.Cell:
    return speed_matrix.Cell(area, batch, fake_backend.FAKE_BACKEND, fake_backend.NO_WEIGHTS, False)


# --- the grid -------------------------------------------------------------------------------------------------


def test_grid_covers_every_axis_of_the_issue():
    cells = speed_matrix.grid()
    assert {cell.image_area for cell in cells} == set(speed_matrix.IMAGE_AREAS) == {384, 512, 768}
    assert {cell.batch_size for cell in cells} == set(speed_matrix.BATCH_SIZES)
    assert {cell.backend for cell in cells} == set(speed_matrix.BACKENDS) == {"vllm", "local"}
    assert {cell.precision for cell in cells} == set(speed_matrix.PRECISIONS)
    assert {cell.prefix_caching for cell in cells} == {True, False}
    # The full product is every combination; the grid hides only the ones nobody could run.
    assert len(speed_matrix.grid(include_not_applicable=True)) == 3 * 3 * 2 * 4 * 2
    assert len(cells) == 81


def test_the_axes_come_from_the_other_lanes_defaults():
    # Nothing is hard-coded twice: the middle area and the middle batch size are the shared defaults.
    from galaxus.serve import runner as serve_runner

    assert speed_matrix.REFERENCE_AREA == images.IMAGE_SIZE
    assert speed_matrix.REFERENCE_BATCH == serve_runner.DEFAULT_BATCH_SIZE
    assert speed_matrix.BATCH_SIZES[1] == serve_runner.DEFAULT_BATCH_SIZE


def test_cell_ids_are_unique_and_round_trip():
    ids = [cell.cell_id for cell in speed_matrix.grid(include_not_applicable=True)]
    assert len(set(ids)) == len(ids)
    for cell_id in ids:
        assert speed_matrix.cell_from_id(cell_id).cell_id == cell_id
    with pytest.raises(speed_matrix.CellError, match="no cell"):
        speed_matrix.cell_from_id("a999-b999-vllm-bf16-cache_on")


def test_bad_axes_are_refused():
    with pytest.raises(ValueError, match="PATCH_SIZE"):
        speed_matrix.Cell(500, 1, "vllm", "bf16", True)
    with pytest.raises(ValueError, match="batch_size"):
        speed_matrix.Cell(512, 0, "vllm", "bf16", True)
    with pytest.raises(ValueError, match="backend"):
        speed_matrix.Cell(512, 1, "api", "bf16", True)
    with pytest.raises(ValueError, match="precision"):
        speed_matrix.Cell(512, 1, "vllm", "int8", True)


def test_only_what_the_repo_documents_counts_as_supported():
    assert speed_matrix.Cell(512, 16, "vllm", "bf16", True).support == speed_matrix.SUPPORTED
    assert speed_matrix.Cell(512, 16, "local", "bf16", False).support == speed_matrix.SUPPORTED
    for precision in ("fp8", "awq-int4", "gptq-int4"):
        cell = speed_matrix.Cell(512, 16, "vllm", precision, True)
        assert cell.support == speed_matrix.UNCONFIRMED
        assert "probe" in cell.support_reason


def test_impossible_combinations_are_not_applicable_and_say_why():
    no_cache = speed_matrix.Cell(512, 16, "local", "bf16", True)
    assert no_cache.support == speed_matrix.NOT_APPLICABLE and "vLLM feature" in no_cache.support_reason
    quantized_local = speed_matrix.Cell(512, 16, "local", "fp8", False)
    assert quantized_local.support == speed_matrix.NOT_APPLICABLE
    assert "dependency" in quantized_local.support_reason
    assert no_cache.cell_id not in {cell.cell_id for cell in speed_matrix.grid()}


def test_priority_puts_the_documented_main_line_first():
    first = [cell for cell in speed_matrix.grid() if cell.priority == "p0"]
    assert len(first) == 9
    assert all(cell.backend == "vllm" and cell.precision == "bf16" and cell.prefix_caching for cell in first)
    # An unconfirmed precision is only ablated at the reference point until the probe says more.
    assert speed_matrix.Cell(512, 16, "vllm", "fp8", True).priority == "p2"
    assert speed_matrix.Cell(768, 64, "vllm", "fp8", True).priority == "p3"


def test_the_fake_cells_are_separate_from_the_product_grid():
    fake_ids = {cell.cell_id for cell in speed_matrix.fake_grid()}
    assert len(fake_ids) == 9
    assert not fake_ids & {cell.cell_id for cell in speed_matrix.grid(include_not_applicable=True)}


# --- a cell that was not run stays empty ----------------------------------------------------------------------


def test_a_cell_that_was_not_run_carries_no_number():
    row = speed_matrix.pending_result(speed_matrix.Cell(512, 16, "vllm", "bf16", True))
    assert row.status == speed_matrix.PENDING
    for name in speed_matrix.NUMERIC_FIELDS:
        assert getattr(row, name) is None, name
    assert row.hardware is None
    assert row.note


@pytest.mark.parametrize("status", [speed_matrix.PENDING, speed_matrix.SKIPPED, speed_matrix.FAILED])
def test_a_plausible_number_cannot_be_put_into_an_unrun_cell(status):
    cell = speed_matrix.Cell(512, 16, "vllm", "bf16", True)
    row = asdict(speed_matrix.pending_result(cell))
    with pytest.raises(ValueError, match="never given a number"):
        speed_matrix.CellResult(**{**row, "status": status, "images_per_second": 25.0})
    with pytest.raises(ValueError, match="never given a number"):
        speed_matrix.CellResult(**{**row, "status": status, "hardware": "one L40S"})


def test_a_measured_cell_must_name_its_hardware_and_its_speed():
    cell = speed_matrix.Cell(512, 16, "vllm", "bf16", True)
    row = asdict(speed_matrix.pending_result(cell))
    with pytest.raises(ValueError, match="names its hardware"):
        speed_matrix.CellResult(**{**row, "status": speed_matrix.DONE})
    with pytest.raises(ValueError, match="without an hourly price"):
        speed_matrix.CellResult(
            **{
                **row,
                "status": speed_matrix.DONE,
                "hardware": "one L40S",
                "images": 10,
                "images_per_second": 5.0,
                "cost_per_100k": 1.23,
            }
        )


def test_skipped_and_failed_rows_keep_the_reason_and_stay_empty():
    cell = speed_matrix.Cell(768, 64, "vllm", "fp8", True)
    skipped = speed_matrix.skipped_result(cell, "the probe has not answered yet")
    failed = speed_matrix.failed_result(cell, "out of memory")
    assert skipped.status == speed_matrix.SKIPPED and "probe" in skipped.note
    assert failed.status == speed_matrix.FAILED and "memory" in failed.note
    assert all(getattr(row, name) is None for row in (skipped, failed) for name in speed_matrix.NUMERIC_FIELDS)


def test_the_hardware_is_named_and_a_price_needs_a_source():
    with pytest.raises(ValueError, match="name the hardware"):
        speed_matrix.Hardware("  ")
    with pytest.raises(ValueError, match="price_source"):
        speed_matrix.Hardware("one L40S", 1.8)
    with pytest.raises(ValueError, match="positive"):
        speed_matrix.Hardware("one L40S", 0, "a price page")


# --- running a cell -------------------------------------------------------------------------------------------


def test_run_cell_reuses_the_plan_md_cost_formula(items):
    cell = fake_cell(batch=1)
    predictor = fake_backend.FakeBackendPredictor(cell.image_area)
    row = speed_matrix.run_cell(cell, predictor, items, named_gpu(3.6), warmup_batches=1)

    assert row.status == speed_matrix.DONE and row.images == 2 and row.warmup_images == 1
    assert row.images_per_second > 0
    # The same figure the formula in PLAN.md "Speed" gives, up to the rounding galaxus.serve.bench applies.
    assert row.cost_per_100k == pytest.approx(serve_bench.cost_per_100k(3.6, row.images_per_second), abs=1e-3)
    assert row.hours_per_100k == pytest.approx(serve_bench.hours_per_100k(row.images_per_second), abs=1e-3)
    assert row.hardware == "test CPU, named by hand" and row.detected_device
    assert row.is_machinery_check, "a row with no model behind it never counts as a product number"


def test_without_a_price_the_cost_column_stays_empty(items):
    row = speed_matrix.run_cell(fake_cell(batch=1), fake_backend.FakeBackendPredictor(), items, named_gpu())
    assert row.images_per_second > 0
    assert row.cost_per_100k is None and row.dollars_per_hour is None
    assert row.hours_per_100k is not None, "hours per 100k needs no price"


def test_run_cell_refuses_a_predictor_that_contradicts_the_cell(items):
    cell = fake_cell(area=384, batch=1)
    wrong_area = fake_backend.FakeBackendPredictor(768)
    with pytest.raises(speed_matrix.CellError, match="image_area"):
        speed_matrix.run_cell(cell, wrong_area, items, named_gpu())


def test_run_cell_refuses_a_cell_nobody_could_run(items):
    cell = speed_matrix.Cell(512, 1, "local", "bf16", True)
    with pytest.raises(speed_matrix.CellError, match="not applicable"):
        speed_matrix.run_cell(cell, fake_backend.FakeBackendPredictor(), items, named_gpu())


def test_a_row_whose_axes_were_ignored_is_a_machinery_check(items):
    cell = speed_matrix.Cell(512, 1, "vllm", "bf16", True)
    row = speed_matrix.run_cell(cell, StubPredictor(), items, named_gpu(), axes_applied=False)
    assert row.status == speed_matrix.DONE and row.is_machinery_check
    assert "NOT applied" in row.note
    view = speed_matrix.matrix_view([row.to_record()])
    assert [line["cell_id"] for line in view.machinery] == [cell.cell_id]
    assert all(line["status"] == speed_matrix.PENDING for line in view.product)


def test_build_predictor_hands_the_axes_to_the_constructor():
    cell = fake_cell(area=768, batch=16)
    predictor, applied = speed_matrix.build_predictor("bench.fake_backend:FakeBackendPredictor", cell)
    assert applied and predictor.image_area == 768 and predictor.precision == cell.precision
    assert speed_matrix.check_predictor_matches_cell(cell, predictor) == [
        "image_area",
        "precision",
        "prefix_caching",
        "backend",
    ]


def test_build_predictor_says_which_signature_the_grid_needs():
    cell = speed_matrix.Cell(512, 16, "vllm", "bf16", True)
    with pytest.raises(speed_matrix.CellError) as error:
        speed_matrix.build_predictor("serve_helpers:StubPredictor", cell)
    assert "image_area" in str(error.value) and "TODO(model)" in str(error.value)
    # Opting out is allowed, but the row is then marked as not describing a configuration.
    predictor, applied = speed_matrix.build_predictor("serve_helpers:StubPredictor", cell, pass_cell_args=False)
    assert applied is False and callable(predictor.predict)


# --- the CPU stand-in backend ---------------------------------------------------------------------------------


def test_the_fake_backend_really_resizes_to_the_cell_area(items):
    small = fake_backend.FakeBackendPredictor(384)
    large = fake_backend.FakeBackendPredictor(768)
    small.predict(items)
    large.predict(items)
    assert small.vision_tokens_per_image < large.vision_tokens_per_image
    expected = images.vision_tokens(images.prepare_image(items[0].path, image_size=384))
    assert small.vision_tokens_per_image == pytest.approx(expected, abs=1.0)


def test_the_fake_backend_reads_class_names_and_never_spells_them(galaxus_data, items):
    predictor = fake_backend.FakeBackendPredictor()
    predictions = predictor.predict(items)
    assert predictor.labels_source == "galaxus.data.classes.allowed_classes()"
    from galaxus.data.classes import allowed_classes

    assert set(predictions[0].labels) == set(allowed_classes())
    source = Path(fake_backend.__file__).read_text(encoding="utf-8")
    for dimension in allowed_classes():
        assert dimension not in source, "a dimension or class name is never written into our code (AGENTS.md)"


def test_without_a_classes_source_the_fake_backend_flags_its_placeholder(monkeypatch, items):
    monkeypatch.delenv("GALAXUS_DATA", raising=False)
    monkeypatch.setattr("galaxus.data.classes.CONTRACTS_CLASSES", Path("/nonexistent/classes.json"))
    predictor = fake_backend.FakeBackendPredictor()
    predictions = predictor.predict(items)
    assert "placeholder" in predictor.labels_source and "timing only" in predictor.labels_source
    assert list(predictions[0].labels) == [fake_backend.PLACEHOLDER_DIMENSION]


def test_the_fake_backend_refuses_to_pretend_it_has_weights_or_a_prompt():
    with pytest.raises(ValueError, match="no weights"):
        fake_backend.FakeBackendPredictor(512, "bf16", False)
    with pytest.raises(ValueError, match="prefix cache"):
        fake_backend.FakeBackendPredictor(512, fake_backend.NO_WEIGHTS, True)


def test_synthetic_items_are_images_on_disk(tmp_path):
    built = fake_backend.synthetic_items(3, tmp_path / "synthetic")
    assert len(built) == 3
    assert all(Path(item.path).is_file() for item in built)
    assert len({item.image_id for item in built}) == 3
    with pytest.raises(ValueError, match="at least 1"):
        fake_backend.synthetic_items(0, tmp_path / "synthetic")


# --- saving, loading and the table ----------------------------------------------------------------------------


def test_results_are_saved_outside_the_repository_and_read_back(items, tmp_path):
    row = speed_matrix.run_cell(fake_cell(batch=1), fake_backend.FakeBackendPredictor(), items, named_gpu(1.5))
    path = speed_matrix.save([row], "unit", root=tmp_path)
    assert path == tmp_path / serve_bench.BENCH_DIR / "unit.jsonl"
    records = speed_matrix.load(path)
    assert len(records) == 1 and records[0]["cell_id"] == row.cell_id
    assert records[0]["is_machinery_check"] is True
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["cost_per_100k"] == row.cost_per_100k
    # A second run of the same cell replaces the first in the table.
    speed_matrix.save([row], "unit", root=tmp_path)
    assert len(speed_matrix.load(path)) == 2
    assert len(speed_matrix.latest_by_cell(speed_matrix.load(path))) == 1


def test_raw_results_never_land_in_git():
    with pytest.raises(ValueError, match="inside the repository"):
        speed_matrix.matrix_path("speed_matrix", root=REPO_ROOT / "bench")
    with pytest.raises(ValueError, match="inside the repository"):
        speed_matrix.matrix_path("speed_matrix", root=REPO_ROOT)


def test_a_label_cannot_escape_the_bench_directory(tmp_path):
    for bad in ("../escape", "a/b", ".hidden", ""):
        with pytest.raises(ValueError, match="label"):
            speed_matrix.matrix_path(bad, root=tmp_path)


def test_the_table_shows_pending_cells_as_empty(items, tmp_path):
    row = speed_matrix.run_cell(fake_cell(batch=1), fake_backend.FakeBackendPredictor(), items, named_gpu(1.5))
    table = speed_matrix.matrix_markdown([row.to_record()])
    assert "pending" in table and "Machinery checks" in table
    assert f"0 of {len(speed_matrix.grid())} product cells measured" in table
    assert table.isascii(), "the table is read in a Windows console as often as in a markdown viewer"
    pending_lines = [line for line in table.splitlines() if line.endswith("| pending |")]
    assert len(pending_lines) == len(speed_matrix.grid())
    for line in pending_lines:
        columns = [column.strip() for column in line.strip("| ").split("|")]
        measured = dict(zip(speed_matrix.HEADERS, columns))
        for column in ("img/s", "peak GB", "$/100k", "vis. tokens", "hardware"):
            assert measured[column] == "-", f"{column} of a pending cell must be empty, got {measured[column]!r}"


def test_load_rejects_a_file_that_is_not_a_matrix(tmp_path):
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a matrix record"):
        speed_matrix.load(broken)
    assert speed_matrix.load(tmp_path / "missing.jsonl") == []


# --- the commands a human runs --------------------------------------------------------------------------------


def test_the_plan_covers_every_cell_it_is_asked_for():
    cells = [cell for cell in speed_matrix.grid() if cell.priority in ("p0", "p1")]
    plan = gpu_plan.plan_markdown(cells)
    assert gpu_plan.cell_ids_in(plan) == {cell.cell_id for cell in cells}
    assert gpu_plan.VLLM_IMAGE in plan and gpu_plan.MODEL in plan
    for cell in cells:
        assert cell.cell_id in plan


def test_the_plan_never_runs_anything_itself_and_invents_no_number():
    plan = gpu_plan.plan_markdown()
    assert "agents never start a paid GPU job" in plan
    assert "TODO(owner)" in plan, "the price is the owner's, never ours"
    assert re.search(r"--dollars-per-hour [\d.]", plan) is None, "no price is written into a command"
    assert re.search(r"--hardware [\"']?[A-Za-z]", plan) is None, "the hardware name comes from nvidia-smi"
    assert "nvidia-smi" in plan and "$GPU_NAME" in plan
    assert plan.isascii()


def test_the_plan_gates_the_quantized_cells_behind_a_probe():
    unconfirmed = [cell for cell in speed_matrix.grid() if cell.support == speed_matrix.UNCONFIRMED]
    plan = gpu_plan.plan_markdown(unconfirmed[:2])
    assert "only after Step 2 says yes" in plan
    assert "quantization" in plan.lower()
    assert gpu_plan.cell_ids_in(plan) == {cell.cell_id for cell in unconfirmed[:2]}


def test_the_plan_names_what_is_still_missing():
    plan = gpu_plan.plan_markdown()
    assert "TODO(model)" in plan and "TODO(design)" in plan
    assert gpu_plan.PROPOSED_VLLM_PREDICTOR in plan and gpu_plan.PROPOSED_LOCAL_PREDICTOR in plan


# --- the command line -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["grid", "--format", "ids"],
        ["grid", "--format", "jsonl", "--priority", "p0"],
        ["grid", "--format", "md", "--backend", "local"],
        ["plan", "--priority", "p0"],
    ],
)
def test_the_read_only_subcommands_work(argv, capsys):
    assert speed_matrix.main(argv) == 0
    assert capsys.readouterr().out.strip()


def test_the_cpu_cells_run_from_the_command_line(tmp_path, capsys):
    assert (
        speed_matrix.main(
            ["fake", "--images", "4", "--images-dir", str(tmp_path / "pics"), "--root", str(tmp_path), "--label", "cli"]
        )
        == 0
    )
    records = speed_matrix.load(tmp_path / serve_bench.BENCH_DIR / "cli.jsonl")
    assert len(records) == len(speed_matrix.fake_grid())
    done = [record for record in records if record["status"] == speed_matrix.DONE]
    skipped = [record for record in records if record["status"] == speed_matrix.SKIPPED]
    # Four images are enough for batch size 1 and too few for the bigger batches: those rows stay empty.
    assert len(done) == 3 and len(skipped) == 6
    assert all(record["images_per_second"] > 0 and record["hardware"] for record in done)
    assert all(record["images_per_second"] is None for record in skipped)
    assert all(record["is_machinery_check"] for record in records)
    assert "too few" in skipped[0]["note"]


def test_the_report_subcommand_reads_what_was_saved(tmp_path, capsys):
    speed_matrix.main(
        ["fake", "--images", "4", "--images-dir", str(tmp_path / "pics"), "--root", str(tmp_path), "--label", "cli"]
    )
    capsys.readouterr()
    assert speed_matrix.main(["report", "--root", str(tmp_path), "--label", "cli"]) == 0
    out = capsys.readouterr().out
    assert "Machinery checks" in out and "pending" in out
    assert speed_matrix.main(["report", "--from", str(tmp_path / "nothing.jsonl")]) == 0
