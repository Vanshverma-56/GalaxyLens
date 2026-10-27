from PIL import Image
import pytest
from galaxylens.transforms import apply_corruption

def test_clean_corruption_preserves_pixels():
    image = Image.new("RGB", (32, 32), color=(80, 100, 120))
    result = apply_corruption(image, "clean", 1, seed=4)
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
