# Phase 1 — ArcFace + MHE repulsion on classifier weight matrix
#
# Loss = L_arcface + lambda_repulsion * L_repulsion
#
# L_repulsion = mean of off-diagonal cosine similarities of ArcFaceHead weights.
# Penalizes class centroids that are too close on the hypersphere,
# encouraging uniform angular distribution.
#
# Config fields needed:
#   loss.lambda_repulsion: float  (e.g. 0.1)

raise NotImplementedError("ArcFace+Repulsion loss: Phase 1 extension")
