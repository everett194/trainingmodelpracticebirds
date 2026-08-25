"""
BirdStrikeGeo
-------------
A geospatial machine-learning project for exploring reported wildlife
strikes to aircraft and bird-migration monitoring data.

Two distinct, non-interchangeable tasks are supported:

1. **Damage prediction** (`birdstrikegeo.models.neural_net`,
   `birdstrikegeo.training.train_damage`): given a *reported* wildlife
   strike, estimate the probability that it caused aircraft damage. This
   is conditional on a strike having already been reported - it is NOT a
   prediction of whether a strike will occur.

2. **Activity index** (`birdstrikegeo.models.activity_index`): a
   geospatial summary of relative wildlife activity near an airport
   during a time window, derived from nearby bird-migration monitoring
   observations (Trektellen). This is an index, not a calibrated
   probability, and is only as good as the monitoring coverage available.

See README.md, DATA_CARD.md, MODEL_CARD.md and GEOSPATIAL_METHODS.md at
the repository root for full documentation, and synthetic_demo/ for the
original, unrelated educational neural-network project this repository
started from.
"""

__version__ = "0.1.0-dev"
