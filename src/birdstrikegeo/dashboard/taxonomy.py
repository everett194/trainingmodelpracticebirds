"""
birdstrikegeo.dashboard.taxonomy
-------------------------------------
Classifies FAA Wildlife Strike Database SPECIES strings (the exact
common-name resolution FAA reports, e.g. "Canada goose", "Geese",
"Unknown bird - large") for BirdStrikeNet's dashboard.

NOT the same table as birdstrikegeo.features.taxonomy.SPECIES_TAXONOMY,
which is keyed by scientific name for Trektellen grouped-activity
features. This one is keyed by FAA's own common-name strings, because
that's the resolution strike reports actually carry, and per the
project brief: "keep FAA's reported resolution... never assign a
group-level report to a specific species." A report already filed as
"Geese" stays "Geese" in the default (ungrouped) view; the GROUP field
below is used only for the optional grouped view, where it rolls
"Canada goose" and "Geese" up into the same coarse bucket without ever
pretending one is the other.

Coverage: every species in this table was looked up by hand (common
North American bird/mammal taxonomy, not guessed from the string) for
the ~120 most-frequent species in national_species_risk.csv plus every
species actually observed in Eastern Shore MD strike reports. A SPECIES
value not in this table returns UNKNOWN_GROUP / is_mammal=False /
banding_detectable=None - never a guessed classification.
"""

from __future__ import annotations

from dataclasses import dataclass

UNKNOWN_GROUP = "unknown"

GROUP_LABELS: dict[str, str] = {
    "waterfowl": "Geese, ducks & swans",
    "gulls": "Gulls",
    "vultures": "Vultures",
    "raptors": "Raptors (hawks, falcons, owls, eagles)",
    "large_waterbirds": "Herons, egrets, cormorants, pelicans & cranes",
    "doves_pigeons": "Doves & pigeons",
    "songbirds_small": "Small songbirds & swifts/swallows",
    "shorebirds": "Shorebirds (plovers, sandpipers)",
    "gamebirds": "Upland gamebirds",
    "pelagic_seabirds": "Pelagic seabirds",
    "deer": "Deer",
    "mammals_other": "Other mammals",
    "unknown": "Unknown species",
}


@dataclass(frozen=True)
class TaxonInfo:
    group: str
    is_mammal: bool
    is_unknown: bool
    # True = mist-net banding could plausibly catch this taxon; False =
    # structurally cannot (large/soaring/waterbound/ground-dwelling);
    # None = can't be determined (species unresolved).
    banding_detectable: bool | None

    @property
    def group_label(self) -> str:
        return GROUP_LABELS.get(self.group, self.group)


def _entry(group: str, is_mammal: bool = False, is_unknown: bool = False,
           banding_detectable: bool | None = None) -> TaxonInfo:
    return TaxonInfo(group=group, is_mammal=is_mammal, is_unknown=is_unknown,
                      banding_detectable=banding_detectable)


# FAA SPECIES string (as reported, case preserved from the database) ->
# TaxonInfo. Group-level FAA categories ("Geese", "Gulls", "Hawks", ...)
# are classified exactly like specific species for group/mammal/banding
# purposes - the distinction that matters (never attributing a
# group-level report to one species) is enforced in query.py, not here.
_TAXONOMY: dict[str, TaxonInfo] = {
    # --- waterfowl ---
    "Canada goose": _entry("waterfowl", banding_detectable=False),
    "Geese": _entry("waterfowl", banding_detectable=False),
    "Ducks": _entry("waterfowl", banding_detectable=False),
    "Mallard": _entry("waterfowl", banding_detectable=False),
    "Snow goose": _entry("waterfowl", banding_detectable=False),
    "Ducks, geese, swans": _entry("waterfowl", banding_detectable=False),
    "Greater white-fronted goose": _entry("waterfowl", banding_detectable=False),
    "Northern pintail": _entry("waterfowl", banding_detectable=False),

    # --- gulls ---
    "Gulls": _entry("gulls", banding_detectable=False),
    "Ring-billed gull": _entry("gulls", banding_detectable=False),
    "Herring gull": _entry("gulls", banding_detectable=False),
    "Laughing gull": _entry("gulls", banding_detectable=False),
    "Glaucous-winged gull": _entry("gulls", banding_detectable=False),
    "California gull": _entry("gulls", banding_detectable=False),
    "White-headed gull complex": _entry("gulls", banding_detectable=False),
    "Franklin's gull": _entry("gulls", banding_detectable=False),

    # --- vultures ---
    "Turkey vulture": _entry("vultures", banding_detectable=False),
    "Black vulture": _entry("vultures", banding_detectable=False),
    "New World vultures": _entry("vultures", banding_detectable=False),

    # --- raptors ---
    "American kestrel": _entry("raptors", banding_detectable=False),
    "Red-tailed hawk": _entry("raptors", banding_detectable=False),
    "Hawks": _entry("raptors", banding_detectable=False),
    "Bald eagle": _entry("raptors", banding_detectable=False),
    "Osprey": _entry("raptors", banding_detectable=False),
    "Peregrine falcon": _entry("raptors", banding_detectable=False),
    "Owls": _entry("raptors", banding_detectable=False),
    "American barn owl": _entry("raptors", banding_detectable=False),
    "Great horned owl": _entry("raptors", banding_detectable=False),
    "Cooper's hawk": _entry("raptors", banding_detectable=False),
    "Merlin": _entry("raptors", banding_detectable=False),
    "Falcons, kestrels, falconets": _entry("raptors", banding_detectable=False),
    "Raptors: Hawks, eagles, vultures, kites, osprey, falcons, caracaras": _entry("raptors", banding_detectable=False),
    "Snowy owl": _entry("raptors", banding_detectable=False),
    "Short-eared owl": _entry("raptors", banding_detectable=False),
    "Swainson's hawk": _entry("raptors", banding_detectable=False),
    "Red-shouldered hawk": _entry("raptors", banding_detectable=False),
    "Northern harrier": _entry("raptors", banding_detectable=False),
    "Burrowing owl": _entry("raptors", banding_detectable=False),

    # --- large waterbirds (herons, egrets, cormorants, pelicans, cranes) ---
    "Western cattle egret": _entry("large_waterbirds", banding_detectable=False),
    "Great blue heron": _entry("large_waterbirds", banding_detectable=False),
    "Egrets": _entry("large_waterbirds", banding_detectable=False),
    "Great egret": _entry("large_waterbirds", banding_detectable=False),
    "Herons": _entry("large_waterbirds", banding_detectable=False),
    "Anhinga": _entry("large_waterbirds", banding_detectable=False),
    "Double-crested cormorant": _entry("large_waterbirds", banding_detectable=False),
    "Sandhill crane": _entry("large_waterbirds", banding_detectable=False),
    "Brown pelican": _entry("large_waterbirds", banding_detectable=False),
    "American white pelican": _entry("large_waterbirds", banding_detectable=False),

    # --- doves & pigeons (mist-net detectable) ---
    "Mourning dove": _entry("doves_pigeons", banding_detectable=True),
    "Rock pigeon": _entry("doves_pigeons", banding_detectable=True),
    "Doves": _entry("doves_pigeons", banding_detectable=True),
    "Zebra dove": _entry("doves_pigeons", banding_detectable=True),
    "Spotted dove": _entry("doves_pigeons", banding_detectable=True),

    # --- small songbirds / swifts / swallows (mist-net detectable) ---
    "European starling": _entry("songbirds_small", banding_detectable=True),
    "Sparrows": _entry("songbirds_small", banding_detectable=True),
    "Barn swallow": _entry("songbirds_small", banding_detectable=True),
    "Horned lark": _entry("songbirds_small", banding_detectable=True),
    "Blackbirds": _entry("songbirds_small", banding_detectable=True),
    "Eastern meadowlark": _entry("songbirds_small", banding_detectable=True),
    "Western meadowlark": _entry("songbirds_small", banding_detectable=True),
    "American crow": _entry("songbirds_small", banding_detectable=True),
    "Cliff swallow": _entry("songbirds_small", banding_detectable=True),
    "American robin": _entry("songbirds_small", banding_detectable=True),
    "Perching birds (y)": _entry("songbirds_small", banding_detectable=True),
    "Chimney swift": _entry("songbirds_small", banding_detectable=True),
    "Tree swallow": _entry("songbirds_small", banding_detectable=True),
    "Crows": _entry("songbirds_small", banding_detectable=True),
    "Bank swallow": _entry("songbirds_small", banding_detectable=True),
    "Cedar waxwing": _entry("songbirds_small", banding_detectable=True),
    "Savannah sparrow": _entry("songbirds_small", banding_detectable=True),
    "Snow bunting": _entry("songbirds_small", banding_detectable=True),
    "Meadowlarks": _entry("songbirds_small", banding_detectable=True),
    "Red-winged blackbird": _entry("songbirds_small", banding_detectable=True),
    "Purple martin": _entry("songbirds_small", banding_detectable=True),
    "White-throated swift": _entry("songbirds_small", banding_detectable=True),
    "Common nighthawk": _entry("songbirds_small", banding_detectable=True),
    "Western kingbird": _entry("songbirds_small", banding_detectable=True),
    "Swainson's thrush": _entry("songbirds_small", banding_detectable=True),
    "American pipit": _entry("songbirds_small", banding_detectable=True),
    "Yellow-rumped warbler": _entry("songbirds_small", banding_detectable=True),
    "House sparrow": _entry("songbirds_small", banding_detectable=True),
    "White-throated sparrow": _entry("songbirds_small", banding_detectable=True),
    "Finches, euphonias": _entry("songbirds_small", banding_detectable=True),
    "Brown-headed cowbird": _entry("songbirds_small", banding_detectable=True),
    "Scissor-tailed flycatcher": _entry("songbirds_small", banding_detectable=True),
    "Common raven": _entry("songbirds_small", banding_detectable=True),
    "Common grackle": _entry("songbirds_small", banding_detectable=True),
    "Common yellowthroat": _entry("songbirds_small", banding_detectable=True),
    "Lapland longspur": _entry("songbirds_small", banding_detectable=True),

    # --- shorebirds (ground/water foragers; not mist-net targets) ---
    "Killdeer": _entry("shorebirds", banding_detectable=False),
    "Sandpipers, curlews, phalaropes, allies": _entry("shorebirds", banding_detectable=False),
    "Pacific golden-plover": _entry("shorebirds", banding_detectable=False),
    "Plovers": _entry("shorebirds", banding_detectable=False),
    "Black-bellied plover": _entry("shorebirds", banding_detectable=False),
    "Upland sandpiper": _entry("shorebirds", banding_detectable=False),
    "American golden-plover": _entry("shorebirds", banding_detectable=False),
    "Least sandpiper": _entry("shorebirds", banding_detectable=False),

    # --- gamebirds (ground-dwelling) ---
    "Wild turkey": _entry("gamebirds", banding_detectable=False),
    "Ring-necked pheasant": _entry("gamebirds", banding_detectable=False),

    # --- pelagic seabirds ---
    "Bonin petrel": _entry("pelagic_seabirds", banding_detectable=False),
    "Laysan albatross": _entry("pelagic_seabirds", banding_detectable=False),

    # --- mammals: deer ---
    "White-tailed deer": _entry("deer", is_mammal=True, banding_detectable=False),
    "Mule deer": _entry("deer", is_mammal=True, banding_detectable=False),

    # --- mammals: other ---
    "Coyote": _entry("mammals_other", is_mammal=True, banding_detectable=False),
    "Microbats": _entry("mammals_other", is_mammal=True, banding_detectable=False),
    "Woodchuck": _entry("mammals_other", is_mammal=True, banding_detectable=False),
    "Red fox": _entry("mammals_other", is_mammal=True, banding_detectable=False),
    "Striped skunk": _entry("mammals_other", is_mammal=True, banding_detectable=False),
    "Domestic/feral dog": _entry("mammals_other", is_mammal=True, banding_detectable=False),
    "Black-tailed jackrabbit": _entry("mammals_other", is_mammal=True, banding_detectable=False),
    "Raccoon": _entry("mammals_other", is_mammal=True, banding_detectable=False),
    "Virginia opossum": _entry("mammals_other", is_mammal=True, banding_detectable=False),
    "Unknown terrestrial mammal": _entry("mammals_other", is_mammal=True, is_unknown=True),

    # --- unknown (species genuinely unresolved in the FAA report) ---
    "Unknown bird": _entry("unknown", is_unknown=True),
    "Unknown bird - small": _entry("unknown", is_unknown=True),
    "Unknown bird - medium": _entry("unknown", is_unknown=True),
    "Unknown bird - large": _entry("unknown", is_unknown=True),
    "Unknown bird or bat": _entry("unknown", is_unknown=True),
}

_FALLBACK = _entry(UNKNOWN_GROUP, is_mammal=False, is_unknown=False, banding_detectable=None)

# Case-insensitive view of _TAXONOMY: FAA SPECIES strings are mostly
# sentence-case ("Canada goose"), but Trektellen exports title-case
# common names ("Canada Goose") - both resolve to the same entry here
# rather than requiring every caller to normalize case itself.
_TAXONOMY_UPPER: dict[str, TaxonInfo] = {name.upper(): info for name, info in _TAXONOMY.items()}


def classify(species: str | None) -> TaxonInfo:
    """
    Returns the documented TaxonInfo for an FAA (or Trektellen common-
    name-compatible) species string, matched case-insensitively, or an
    "unknown group / not a mammal / banding-detectability unknown"
    fallback if the species isn't in the table - never a guessed
    classification. Note: this fallback's is_unknown is False even
    though the taxon's identity is uncertain to us - "is_unknown" means
    "FAA itself could not resolve this to a species," a different fact
    from "this classifier doesn't have an entry for it yet."
    """
    if not species:
        return _FALLBACK
    return _TAXONOMY_UPPER.get(species.strip().upper(), _FALLBACK)


def is_covered(species: str | None) -> bool:
    """True if `species` has a real (non-fallback) taxonomy entry."""
    return bool(species) and species.strip().upper() in _TAXONOMY_UPPER


# banding_detectable is consistent across every species sharing a group
# (see the table above) - precomputed here so a GROUP's detectability
# can be looked up directly, without depending on any one species
# actually having a matching Trektellen row. This is the fix for taxa
# mist nets structurally cannot catch (vultures, waterfowl, ...): they
# may have ZERO rows in a real Trektellen export, and "no row found"
# must never be confused with "not monitored."
_GROUP_BANDING_DETECTABLE: dict[str, bool | None] = {}
for _info in _TAXONOMY.values():
    _GROUP_BANDING_DETECTABLE.setdefault(_info.group, _info.banding_detectable)


def group_banding_detectable(group: str) -> bool | None:
    """banding_detectable for a whole taxonomy group, or None if the group isn't documented."""
    return _GROUP_BANDING_DETECTABLE.get(group)
