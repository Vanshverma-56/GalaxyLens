from galaxylens.labels import CLASS_NAMES, NUM_CLASSES, PROVISIONAL_SUPERCLASS_MAP

def test_original_label_schema_has_ten_classes():
    assert NUM_CLASSES == 10
    assert len(CLASS_NAMES) == 10
    assert list(PROVISIONAL_SUPERCLASS_MAP) == list(range(10))
