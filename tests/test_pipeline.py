"""Обработка папки: требования кейса."""
import json

import pytest

from cloudai.pipeline import DELETED, ERROR, SAVED, format_report, process_folder, save_report, should_delete
from conftest import flat_image


def test_threshold_rule_is_strictly_above():
    assert should_delete(75.1) is True
    assert should_delete(75.0) is False            # «выше 75 %»: ровно 75 % ещё оставляем
    assert should_delete(74.9) is False
    assert should_delete(50.0, threshold=40.0) is True


def test_cloudy_images_are_deleted_and_clear_are_saved(model, scene_file, tmp_path):
    incoming, saved = tmp_path / "incoming", tmp_path / "saved"
    scene_file("clear.jpg", 0.10, seed=1, folder=incoming)
    scene_file("half.jpg", 0.50, seed=2, folder=incoming)
    scene_file("cloudy.jpg", 0.95, seed=3, folder=incoming)

    report = process_folder(incoming, saved, model)

    by_name = {r.name: r for r in report.results}
    assert by_name["clear.jpg"].status == SAVED and by_name["half.jpg"].status == SAVED
    assert by_name["cloudy.jpg"].status == DELETED
    assert (saved / "clear.jpg").exists() and (saved / "half.jpg").exists()
    assert not (incoming / "cloudy.jpg").exists() and not (saved / "cloudy.jpg").exists()   # действительно стёрт
    assert list(incoming.iterdir()) == []                                                  # «входящие» пусты


def test_operator_message_has_three_required_facts(model, scene_file, tmp_path):
    """В сообщении есть: имя сохранённого файла, процент облачности, имя удалённого файла."""
    incoming = tmp_path / "incoming"
    scene_file("good_one.jpg", 0.20, seed=4, folder=incoming)
    scene_file("bad_one.jpg", 0.97, seed=5, folder=incoming)
    report = process_folder(incoming, tmp_path / "saved", model)
    text = format_report(report)
    saved_part, deleted_part = text.split("УДАЛЁННЫЕ ФАЙЛЫ")
    assert "good_one.jpg" in saved_part and "bad_one.jpg" in deleted_part
    for result in report.results:
        assert f"{result.cloud_percent:.1f}".replace(".", ",") + " %" in text
    assert "сохранено: 1" in text and "удалено: 1" in text


def test_decision_matches_the_number_shown_to_operator(model, scene_file, tmp_path):
    incoming = tmp_path / "incoming"
    for i, coverage in enumerate([0.70, 0.73, 0.75, 0.77, 0.80]):
        scene_file(f"edge_{i}.jpg", coverage, seed=20 + i, folder=incoming)
    report = process_folder(incoming, tmp_path / "saved", model)
    for r in report.results:
        assert (r.status == DELETED) == (r.cloud_percent > 75.0)


def test_trash_mode_moves_instead_of_deleting(model, scene_file, tmp_path):
    incoming, trash = tmp_path / "incoming", tmp_path / "trash"
    scene_file("cloudy.jpg", 0.95, seed=3, folder=incoming)
    report = process_folder(incoming, tmp_path / "saved", model, delete_mode="trash", trash_dir=trash)
    assert report.results[0].status == DELETED and report.results[0].moved_to_trash
    assert (trash / "cloudy.jpg").exists()
    assert "перенесён в корзину" in format_report(report)


def test_custom_threshold(model, scene_file, tmp_path):
    incoming = tmp_path / "incoming"
    scene_file("half.jpg", 0.50, seed=2, folder=incoming)
    report = process_folder(incoming, tmp_path / "saved", model, threshold=30.0)
    assert report.results[0].status == DELETED


def test_broken_file_is_never_deleted(model, tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    broken = incoming / "broken.jpg"
    broken.write_bytes(b"this is not a picture")
    report = process_folder(incoming, tmp_path / "saved", model)
    assert report.results[0].status == ERROR
    assert broken.exists()                                     # файл не тронут
    assert "broken.jpg" in format_report(report) and "НЕ ОБРАБОТАНЫ" in format_report(report)


def test_other_files_are_ignored(model, tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    (incoming / "notes.txt").write_text("заметки оператора", encoding="utf-8")
    (incoming / "folder.jpg").mkdir()                          # папка с «картиночным» именем — не снимок
    report = process_folder(incoming, tmp_path / "saved", model)
    assert report.results == []
    assert (incoming / "notes.txt").exists()
    assert "нет" in format_report(report)


def test_same_name_does_not_overwrite_saved_file(model, tmp_path):
    incoming, saved = tmp_path / "incoming", tmp_path / "saved"
    incoming.mkdir(); saved.mkdir()
    flat_image(saved / "scene.png", (10, 60, 10))
    old_bytes = (saved / "scene.png").read_bytes()
    flat_image(incoming / "scene.png", (20, 40, 90))
    report = process_folder(incoming, saved, model)
    assert report.results[0].saved_as == "scene_1.png"
    assert (saved / "scene.png").read_bytes() == old_bytes
    assert "scene_1.png (было: scene.png)" in format_report(report)


def test_formats_png_tiff_bmp_and_grayscale(model, tmp_path):
    import numpy as np
    from PIL import Image
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    flat_image(incoming / "a.png", (15, 70, 20))
    flat_image(incoming / "b.tif", (15, 70, 20))
    flat_image(incoming / "c.bmp", (240, 240, 245))
    Image.fromarray(np.full((120, 160), 20, dtype=np.uint8)).save(incoming / "d_gray.png")
    Image.fromarray(np.full((120, 160), 60000, dtype=np.uint16)).save(incoming / "e_16bit.tif")
    report = process_folder(incoming, tmp_path / "saved", model)
    status = {r.name: r.status for r in report.results}
    assert status == {"a.png": SAVED, "b.tif": SAVED, "c.bmp": DELETED, "d_gray.png": SAVED, "e_16bit.tif": DELETED}


def test_empty_and_missing_folder(model, tmp_path):
    report = process_folder(tmp_path / "nowhere", tmp_path / "saved", model)
    assert report.results == []
    assert "Обработано: 0" in format_report(report)


def test_wrong_settings_are_rejected(model, tmp_path):
    with pytest.raises(ValueError):
        process_folder(tmp_path, tmp_path / "saved", model, threshold=150)
    with pytest.raises(ValueError):
        process_folder(tmp_path, tmp_path / "saved", model, delete_mode="burn")
    with pytest.raises(ValueError):
        process_folder(tmp_path, tmp_path, model)              # одна и та же папка
    with pytest.raises(ValueError):
        process_folder(tmp_path, tmp_path / "saved", model, delete_mode="trash")   # нет папки корзины


def test_report_files(model, scene_file, tmp_path):
    incoming = tmp_path / "incoming"
    scene_file("one.jpg", 0.30, seed=7, folder=incoming)
    report = process_folder(incoming, tmp_path / "saved", model, previews_dir=tmp_path / "previews")
    paths = save_report(report, tmp_path / "reports", run_id="test")
    assert "one.jpg" in paths["txt"].read_text(encoding="utf-8")
    assert "one.jpg;сохранён;" in paths["csv"].read_text(encoding="utf-8-sig")
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["counts"] == {"total": 1, "saved": 1, "deleted": 0, "errors": 0}
    assert (tmp_path / "previews" / "0001_photo.jpg").exists() and (tmp_path / "previews" / "0001_clouds.jpg").exists()
