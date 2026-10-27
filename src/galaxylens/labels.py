"""Canonical Galaxy10 DECaLS labels.

The primary research task deliberately retains all ten source classes. The earlier
three-superclass idea is not silently applied because any grouping changes the
scientific target and needs an explicitly reviewed mapping.
"""
CLASS_NAMES = (
    "Disturbed Galaxies",
    "Merging Galaxies",
    "Round Smooth Galaxies",
    "In-between Round Smooth Galaxies",
    "Cigar Shaped Smooth Galaxies",
    "Barred Spiral Galaxies",
    "Unbarred Tight Spiral Galaxies",
    "Unbarred Loose Spiral Galaxies",
    "Edge-on Galaxies without Bulge",
    "Edge-on Galaxies with Bulge",
)

NUM_CLASSES = len(CLASS_NAMES)

# Proposed grouping retained for future comparison only; not used in training.
# This is a working hypothesis, not an astrophysical ground-truth taxonomy.
PROVISIONAL_SUPERCLASS_MAP = {
    0: 0, 1: 0,       # Disturbed / merging
    2: 1, 3: 1, 4: 1, # Smooth
    5: 2, 6: 2, 7: 2, 8: 2, 9: 2, # Disk-like classes
}
PROVISIONAL_SUPERCLASS_NAMES = ("Disturbed", "Smooth", "Disk")
