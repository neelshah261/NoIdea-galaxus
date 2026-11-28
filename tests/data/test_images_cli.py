"""``python -m galaxus.data.images``: filling the 512 px cache for one part, and timing it (#180 task 1).

Why this CLI exists: both members of the final method read the data lane's 512 px cache, and the 5,292 final
images arrive uncached at 13:00. The rehearsal on a laptop ran on dev images that were already cached, so it never
paid that cost and could not measure it. ``--directory`` with ``--limit`` points the fill at a fresh folder, which
makes every image genuinely uncached and gives a real rate on the machine that will do the work.

Synthetic images written by the test, no competition data and no network.
"""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import write_image

from galaxus.data.images import HOSTS_FINAL_IMAGES, HOSTS_IMAGES_PER_SECOND, main


@pytest.fixture
def part(galaxus_data):
    """A ``train.csv`` of six readable images, laid out the way a decrypted part is."""
    images = galaxus_data / "images"
    images.mkdir()
    ids = [f"img_{index:05d}" for index in range(6)]
    for image_id in ids:
        write_image(images / f"{image_id}.png", "RGBA", (48, 36), color=(30, 120, 200, 255))
    pd.DataFrame({"image_id": ids, "image_path": [f"images/{i}.png" for i in ids]}).to_csv(
        galaxus_data / "train.csv", index=False
    )
    return galaxus_data, ids


def test_it_fills_the_cache_folder_it_is_given(part, tmp_path, capsys):
    folder = tmp_path / "fresh"
    assert main(["--part", "train", "--directory", str(folder), "--workers", "1"]) == 0
    assert len(list(folder.glob("*.jpg"))) == 6
    printed = capsys.readouterr().out
    assert "written 6" in printed
    assert "images per second" in printed


def test_the_rate_is_measured_only_on_the_images_it_actually_prepared(part, tmp_path, capsys):
    """A second run prepares nothing, so a rate would be a division by the work it skipped."""
    folder = tmp_path / "fresh"
    main(["--part", "train", "--directory", str(folder), "--workers", "1"])
    capsys.readouterr()

    assert main(["--part", "train", "--directory", str(folder), "--workers", "1"]) == 0
    printed = capsys.readouterr().out
    assert "already cached 6" in printed
    assert "nothing was prepared" in printed
    assert "images per second" not in printed


def test_limit_prepares_only_the_first_rows_so_a_sample_can_be_timed(part, tmp_path, capsys):
    folder = tmp_path / "fresh"
    assert main(["--part", "train", "--directory", str(folder), "--limit", "2", "--workers", "1"]) == 0
    assert len(list(folder.glob("*.jpg"))) == 2
    assert "2 of 6 rows" in capsys.readouterr().out


def test_it_reports_the_projection_against_the_hosts_own_figure(part, tmp_path, capsys):
    """The bar comes from challenge_description.md, not from a number of ours (never invent deadlines or rates)."""
    main(["--part", "train", "--directory", str(tmp_path / "fresh"), "--workers", "1"])
    printed = capsys.readouterr().out
    assert f"{HOSTS_FINAL_IMAGES:,}" in printed
    assert f"{HOSTS_IMAGES_PER_SECOND:g} images per second" in printed
    assert "final images take" in printed


def test_no_image_id_is_ever_printed(part, tmp_path, capsys):
    """Product rule 3: identifiers are masked or absent in anything shown."""
    _, ids = part
    main(["--part", "train", "--directory", str(tmp_path / "fresh"), "--workers", "2"])
    printed = capsys.readouterr().out
    for image_id in ids:
        assert image_id not in printed


def test_several_workers_prepare_the_same_set(part, tmp_path):
    folder = tmp_path / "threaded"
    assert main(["--part", "train", "--directory", str(folder), "--workers", "4"]) == 0
    assert len(list(folder.glob("*.jpg"))) == 6


@pytest.mark.parametrize("bad", [["--workers", "0"], ["--limit", "0"]])
def test_nonsense_counts_are_refused_rather_than_silently_adjusted(part, tmp_path, bad):
    with pytest.raises(SystemExit):
        main(["--part", "train", "--directory", str(tmp_path / "fresh"), *bad])


def test_a_missing_part_says_what_to_do_next_rather_than_raising_a_key_error(galaxus_data):
    """``--part final`` before the 13:00 decryption must name the next step, not fail obscurely."""
    with pytest.raises(FileNotFoundError, match="final.csv"):
        main(["--part", "final"])
