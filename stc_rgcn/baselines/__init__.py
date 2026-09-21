"""Comparison baselines.

Every baseline reads the same OD dataset through :mod:`stc_rgcn.baselines.common`
and writes the same 34-column prediction file, so a whole results directory can
be scored in one pass with ``stc-rgcn-eval score``.

=================  ===============================  =======================
Model              Console script                   Requires
=================  ===============================  =======================
Gravity, radiation ``stc-rgcn-baseline-physics``    numpy
Tree ensembles     ``stc-rgcn-baseline-trees``      xgboost / lightgbm / sklearn
SI-GCN             ``stc-rgcn-baseline-sigcn``      torch
GMEL               ``stc-rgcn-baseline-gmel``       torch, dgl
=================  ===============================  =======================

Install the optional dependencies with ``pip install "stc-rgcn[baselines]"``.

Models are deliberately not imported here: several pull in heavy optional
dependencies, and importing this package should stay cheap.

Attribution
-----------
``gmel`` is a single-file consolidation of the open-source `GMEL
<https://github.com/jackmiemie/GMEL>`_ implementation (Liu et al., AAAI 2020,
*Learning Geo-Contextual Embeddings for Commuting Flow Prediction*); the data
pipeline was replaced with project adapters and the GBRT fine-tuning stage
removed.  ``sigcn`` is a PyTorch re-implementation of SI-GCN (Yao et al., 2020,
*Spatial Origin-Destination Flow Imputation Using Graph Convolutional Neural
Networks*), whose `reference code <https://github.com/susurrant/flow-imputation>`_
is TensorFlow 1.x.  The gravity, radiation and tree-ensemble baselines were
written for this project.
"""
