# Integration changes relative to upstream

The released model checkpoints are unchanged. Model attention tensors retain
batch/head dimensions when calling PyTorch SDPA on Windows CUDA; the included
verification script checks equivalence against the original mathematical form.
Inference offloads the autoencoder/diffusion model in turn to fit an 8GB GPU.
The native model's point sampling, PCA transform, regularized noise and chart
fields are retained. OBJ/PLY output preserves polygons, including failed layouts.

Native QuadTools source includes the following repairs used by the standard
preset: remove consecutive duplicate polygon corners, compact corresponding
source vertex indices with extracted vertices, and skip negative/unassigned
cluster indices before accessing cluster arrays. These change some failed
upstream outputs, including the benchmark table. They do not ensure good loops,
nonfolding geometry, or artist quality. The standard preset disables additional
experimental chart fitting, region growth, seed-radius and feature-corner modes.

The original hardcoded subdivision face budget can be disabled with maxfaces=0.
The node defaults to that value, zero preprocessing face budget and zero target
quads. Standard extraction flags otherwise match upstream. The three requested
subdivision passes are retained; no-count-cap does not imply infinite density.

Loop aggregation uses min(Fratio,Eratio), including spiraling, across native
diagnostics. A previous research version used the N-only score; that was not
the full paper metric. Four candidate generations are selected by quad fraction,
then full loop score. Geometric defects are reported separately. This selection
does not certify artist-ready topology.

Character cages, QuadWild backends and the earlier character-fitting research
are excluded. Only the learned SQuadGen node is registered by this package.
