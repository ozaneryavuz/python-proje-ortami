import os
import sys
import pytest

# Proje kökü src dizinini import edebilmek için sys.path'e ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import merhaba


def test_merhaba():
    assert merhaba() == "Merhaba Dünya"
