"""Признаки и поля «нет данных»."""
import numpy as np

from cloudai.features import FEATURE_NAMES, N_FEATURES, compute_features, find_nodata


def test_feature_table_shape_and_values():
    rgb = np.random.default_rng(0).random((60, 80, 3)).astype(np.float32)
    feats = compute_features(rgb)
    assert feats.shape == (60, 80, N_FEATURES)
    assert len(FEATURE_NAMES) == N_FEATURES
    assert np.isfinite(feats).all()


def test_white_pixel_is_bright_and_unsaturated():
    feats = compute_features(np.ones((30, 30, 3), dtype=np.float32))
    centre = dict(zip(FEATURE_NAMES, feats[15, 15]))
    assert centre["яркость (max)"] == 1.0
    assert centre["насыщенность"] < 0.01          # белый — не цветной
    assert centre["пестрота 15"] < 0.01           # ровное поле — гладкое


def test_green_pixel_is_saturated():
    rgb = np.zeros((30, 30, 3), dtype=np.float32)
    rgb[..., 1] = 0.5
    centre = dict(zip(FEATURE_NAMES, compute_features(rgb)[15, 15]))
    assert centre["насыщенность"] > 0.8


def test_black_frame_corner_is_nodata_but_dark_lake_is_not():
    rgb = np.full((100, 100, 3), 0.4, dtype=np.float32)
    rgb[:40, :40] = 0.0                            # чёрный угол касается края кадра
    rgb[60:80, 60:80] = 0.0                        # тёмное «озеро» внутри кадра
    nodata = find_nodata(rgb)
    assert nodata[:40, :40].all()
    assert not nodata[60:80, 60:80].any()
    assert not nodata[50, 90]


def test_transparent_pixels_are_nodata():
    rgb = np.full((50, 50, 3), 0.5, dtype=np.float32)
    alpha = np.ones((50, 50), dtype=np.float32)
    alpha[:, :10] = 0.0
    assert find_nodata(rgb, alpha)[:, :10].all()
