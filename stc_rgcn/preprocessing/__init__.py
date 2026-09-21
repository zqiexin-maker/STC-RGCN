"""Feature pipeline that turns raw city data into the model's input files.

Run the five stages in order; each is also a console script:

=========================  ===========================  ========================
Stage                      Console script               Produces
=========================  ===========================  ========================
``poi_embedding``          ``stc-rgcn-poi-embedding``   POI Doc2Vec vectors per grid cell
``dataset_builder``        ``stc-rgcn-build-dataset``   entity dict + train/valid/test splits
``trajectory_synthesis``   ``stc-rgcn-synthesize``      constrained random-walk trajectories
``trajectory_embedding``   ``stc-rgcn-traj-embedding``  trajectory Word2Vec vectors per grid cell
``feature_fusion``         ``stc-rgcn-fuse-features``   the fused ``features.txt``
=========================  ===========================  ========================

Stages 1-3 need raw inputs (a POI shapefile, trip records, processed grid data)
that are not part of this repository; the behavioural constraint tables under
``data/constraints/`` and the resulting sample dataset are included.

The stages depend on ``gensim`` and ``geopandas``, which the ``preprocessing``
extra installs::

    pip install "stc-rgcn[preprocessing]"

Nothing is imported here, so importing the package stays cheap.
"""
