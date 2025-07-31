import numpy as np
import pytest

from lookandmove import PinholeCamera, Renderer, cut_template, match_template, ncc_map


def brute_force_ncc(img, tpl):
    h, w = tpl.shape
    t0 = tpl - tpl.mean()
    out = np.zeros((img.shape[0] - h + 1, img.shape[1] - w + 1))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            win = img[y : y + h, x : x + w]
            w0 = win - win.mean()
            out[y, x] = (w0 * t0).sum() / np.sqrt((w0**2).sum() * (t0**2).sum())
    return out


def test_fft_ncc_equals_brute_force():
    rng = np.random.default_rng(0)
    img = rng.random((23, 31))
    tpl = rng.random((5, 7))
    np.testing.assert_allclose(ncc_map(img, tpl), brute_force_ncc(img, tpl), atol=1e-10)


def test_exact_copy_scores_one():
    rng = np.random.default_rng(1)
    img = rng.random((40, 50))
    tpl = img[12:21, 30:39]
    m = match_template(img, tpl)
    assert m.score == pytest.approx(1.0)
    np.testing.assert_allclose(m.feature, [34.0, 16.0], atol=0.05)


@pytest.fixture(scope="module")
def renderer():
    return Renderer(PinholeCamera.looking_down([0.24, -0.06, 0.905], np.deg2rad(-58.3)))


@pytest.mark.parametrize("pixel", [(200.3, 150.7), (410.55, 300.25), (320.0, 240.0)])
def test_ball_is_found_with_sub_pixel_accuracy(renderer, pixel):
    cam = renderer.camera
    tcp = cam.backproject(pixel, cam.depth_of_plane(0.40))
    img = renderer.render(tcp, rng=np.random.default_rng(2))
    m = match_template(img, renderer.teach_template())
    assert np.linalg.norm(m.feature - np.array(pixel)) < 0.15
    assert m.score > 0.9


def test_roi_changes_the_reported_feature_but_not_the_image_position(renderer):
    cam = renderer.camera
    tcp = cam.backproject((300.0, 200.0), cam.depth_of_plane(0.40))
    img = renderer.render(tcp)
    tpl = renderer.teach_template()
    full = match_template(img, tpl)
    roi = match_template(img, tpl, roi=(220, 120, 200, 160))
    np.testing.assert_allclose(roi.feature_roi, full.feature - [220, 120], atol=1e-6)
    np.testing.assert_allclose(roi.feature, full.feature, atol=1e-6)


def test_cut_template_is_centred():
    img = np.arange(100.0).reshape(10, 10)
    tpl = cut_template(img, (4, 5), 3)
    assert tpl[1, 1] == img[5, 4]
