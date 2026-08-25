"""
features/taxonomy.py
----------------------
A small, explicit species -> taxonomic-group lookup used to compute
grouped Trektellen activity features (e.g. waterfowl_count_previous_7d).

Per project requirements: taxonomic-group features are only computed
when a species has a DOCUMENTED entry here. Anything not listed resolves
to "unknown" - it is never guessed from the name string. Extend this
table deliberately as real data introduces new species.

Keys are scientific names (the more stable identifier across common-name
translations between English and Dutch/other European common names used
in Trektellen exports).
"""

from __future__ import annotations

UNKNOWN_GROUP = "unknown"

# scientific_name -> (common_name, group)
# group is one of: waterfowl, gull, raptor, large_bird, songbird, other
SPECIES_TAXONOMY: dict[str, tuple[str, str]] = {
    "Branta canadensis": ("Canada Goose", "waterfowl"),
    "Anser anser": ("Greylag Goose", "waterfowl"),
    "Anas platyrhynchos": ("Mallard", "waterfowl"),
    "Cygnus olor": ("Mute Swan", "waterfowl"),
    "Larus argentatus": ("Herring Gull", "gull"),
    "Larus marinus": ("Great Black-backed Gull", "gull"),
    "Chroicocephalus ridibundus": ("Black-headed Gull", "gull"),
    "Buteo buteo": ("Common Buzzard", "raptor"),
    "Falco tinnunculus": ("Common Kestrel", "raptor"),
    "Accipiter nisus": ("Eurasian Sparrowhawk", "raptor"),
    "Haliaeetus leucocephalus": ("Bald Eagle", "raptor"),
    "Ardea herodias": ("Great Blue Heron", "large_bird"),
    "Ardea cinerea": ("Grey Heron", "large_bird"),
    "Phalacrocorax carbo": ("Great Cormorant", "large_bird"),
    "Corvus brachyrhynchos": ("American Crow", "songbird"),
    "Corvus corax": ("Common Raven", "songbird"),
    "Sturnus vulgaris": ("Common Starling", "songbird"),
    "Turdus migratorius": ("American Robin", "songbird"),
    "Hirundo rustica": ("Barn Swallow", "songbird"),
    "Columba livia": ("Rock Pigeon", "other"),
    "Charadrius hiaticula": ("Common Ringed Plover", "other"),
}

KNOWN_GROUPS: tuple[str, ...] = (
    "waterfowl",
    "gull",
    "raptor",
    "large_bird",
    "songbird",
    "other",
)


def lookup_group(scientific_name: str | None) -> str:
    """Returns the documented taxonomic group, or UNKNOWN_GROUP if the
    species is not in SPECIES_TAXONOMY. Never guesses from the string."""
    if not scientific_name:
        return UNKNOWN_GROUP
    entry = SPECIES_TAXONOMY.get(scientific_name.strip())
    return entry[1] if entry else UNKNOWN_GROUP


def lookup_common_name(scientific_name: str | None) -> str | None:
    if not scientific_name:
        return None
    entry = SPECIES_TAXONOMY.get(scientific_name.strip())
    return entry[0] if entry else None


def is_large_bird(scientific_name: str | None) -> bool:
    """large_bird group OR waterfowl/raptor, which are also large-bodied
    and relevant to strike-severity risk; documented explicitly rather
    than inferred."""
    return lookup_group(scientific_name) in {"large_bird", "waterfowl", "raptor"}
