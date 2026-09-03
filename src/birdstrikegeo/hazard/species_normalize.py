"""
birdstrikegeo.hazard.species_normalize
------------------------------------------
Normalizes Trektellen species names for joining against FAA species
names. Trektellen names occasionally carry a parenthetical subspecies
tag ("Northern Flicker (Yellow-shafted)") or lump confusion species
behind a slash ("Alder/Willow Flycatcher"). Neither form will exact-match
FAA nomenclature, so both are expanded to plain species names here.

Not a full taxonomic resolver - a pragmatic heuristic sufficient for this
one-station export, documented as a known limitation rather than treated
as authoritative.
"""

from __future__ import annotations

import re

_PARENTHETICAL_SUFFIX = re.compile(r"\s*\([^)]*\)\s*$")


def expand_species_name(raw_name: str) -> list[str]:
    """
    Returns one or more plain species names for a raw Trektellen species
    label:
      - a trailing "(...)" subspecies-group tag is stripped
      - a "X/Y Suffix" pair (first part has no space of its own) expands
        to ["X Suffix", "Y Suffix"], borrowing the shared trailing word
        from the second part
      - a "Full Name / Full Name" pair (both parts already multi-word)
        expands to both names unchanged
    """
    name = _PARENTHETICAL_SUFFIX.sub("", raw_name).strip()

    if "/" not in name:
        return [name]

    first, second = (part.strip() for part in name.split("/", 1))
    if " " not in first and " " in second:
        suffix = second.split(" ", 1)[1]
        first = f"{first} {suffix}"

    return [first, second]
