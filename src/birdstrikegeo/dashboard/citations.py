"""
birdstrikegeo.dashboard.citations
--------------------------------------
Every source link the dashboard's action panel can cite, verified by
live web lookup during BirdStrikeNet's construction (not carried over
from the project brief's working titles - several revision letters and
exact URLs were corrected against what the FAA site actually serves
today). Re-verify periodically; advisory circulars do get superseded.

Part 139 note: the AC 150/5200-33/-32/-38 series sets MANDATORY
standards for Part 139-certificated airports (commercial service). Most
Eastern Shore MD GA airports are NOT Part 139 certificated, so every
action citing one of these is framed as voluntary best practice, not a
compliance requirement - see dashboard.actions' ACTION_RULES.
"""

from __future__ import annotations

CATEGORY_MONITORING = "Monitoring & reporting"
CATEGORY_ASSESSMENT = "Formal assessment / management"

CITATIONS: dict[str, dict] = {
    "ac_150_5200_33c": {
        "label": "FAA AC 150/5200-33C — Hazardous Wildlife Attractants on or near Airports",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/150-5200-33C.pdf",
        "verified": "Active, issued 2020-02-21 (supersedes 33B); confirmed via FAA's current-document listing.",
    },
    "ac_150_5200_32c": {
        "label": "FAA AC 150/5200-32C — Reporting Wildlife Aircraft Strikes",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/AC-150-5200-32C-Wildlife-Strikes.pdf",
        "verified": "Active, issued 2024-07-31 (supersedes 32B); confirmed via FAA's current-document listing.",
    },
    "ac_150_5200_38": {
        "label": "FAA AC 150/5200-38 — Protocol for the Conduct and Review of Wildlife Hazard Site "
                 "Visits, Wildlife Hazard Assessments, and Wildlife Hazard Management Plans",
        "url": "https://www.faa.gov/documentLibrary/media/Advisory_Circular/150-5200-38.pdf",
        "verified": "Active, issued 2018-08-20; no lettered revision found beyond the base document.",
    },
    "faa_wildlife_strike_database": {
        "label": "FAA Wildlife Hazard Mitigation / National Wildlife Strike Database — report a strike",
        "url": "https://www.faa.gov/airports/airport_safety/wildlife",
        "verified": "FAA's wildlife-hazard resource hub, links the current strike-reporting path; "
                     "reporting is voluntary (see AC 150/5200-32C).",
    },
    "usda_aphis_wildlife_services": {
        "label": "USDA APHIS Wildlife Services — Airport Wildlife Hazards program",
        "url": "https://www.aphis.usda.gov/national-wildlife-programs/airports",
        "verified": "Current APHIS page (most recently modified of the candidates found), describes the "
                     "on-request Qualified Airport Wildlife Biologist program partnered with FAA since 1989.",
    },
    "smithsonian_feather_lab": {
        "label": "Smithsonian Feather Identification Lab — how to submit bird-strike remains",
        "url": "https://www.faa.gov/airports/airport_safety/wildlife/smithsonian/how-to-collect-birdstrike-remains",
        "verified": "FAA-hosted instructions; service is free to US-registered aircraft owners/operators "
                     "(shipping not included). Phone: (202) 633-0801.",
    },
}
