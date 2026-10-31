from PIL import Image
import pytest
from galaxylens.transforms import apply_corruption

def test_clean_corruption_preserves_pixels():
    image = Image.new("RGB", (32, 32), color=(80, 100, 120))
    result = apply_corruption(image, "clean", 0, seed=4)
    assert result.size == image.size
    assert result.getpixel((0, 0)) == image.getpixel((0, 0))

@pytest.mark.parametrize("kind", ["blur", "resolution", "noise", "brightness"])
@pytest.mark.parametrize("severity", [1, 2, 3])
def test_corruption_returns_same_size_rgb(kind, severity):
    image = Image.new("RGB", (64, 64), color=(80, 100, 120))
    result = apply_corruption(image, kind, severity, seed=123)
    assert result.mode == "RGB"
    assert result.size == image.size

def test_invalid_severity_rejected():
    with pytest.raises(ValueError):
        apply_corruption(Image.new("RGB", (16,16)), "blur", 4)


@pytest.mark.parametrize(
    "severity, expected_pixel",
    [(1, 80), (2, 60), (3, 40)]
)
def test_brightness_severity_uses_correct_darkening(
    severity, expected_pixel
):
    image = Image.new("RGB", (32, 32), color=(100, 100, 100))
    result = apply_corruption(image, "brightness", severity)
    assert result.getpixel((0, 0)) == (
        expected_pixel, expected_pixel, expected_pixel
    )

def test_clean_corruption_accepts_severity_zero():
    image = Image.new("RGB", (32, 32), color=(80, 100, 120))
    result = apply_corruption(image, "clean", 0, seed=4)
    assert result.size == image.size
    assert result.getpixel((0, 0)) == image.getpixel((0, 0))

