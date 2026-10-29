# Measured benchmark — 2026-10-29

**Experimental quad conversion, not verified artist-quality topology.** One table
does not prove that all furniture works, that only furniture works, or that only
watertight inputs are supported by the model. Those claims are not made.

A fresh table was generated through native ComfyUI Pixal3D using the installed
INT8 model, Trellis2 shape VAE, DINOv3 and MoGe. 1024 shape resolution, shape seeds
42/43, structure seed 56, 25 sampling steps. Generation took 192.64 seconds.
Raw output: 3,382,750 triangles, non-manifold geometry and many small components.

Preparation: ComfyUI UDF remesh at 512, then SDF manifold remesh at 512, then
CGAL alpha wrapping via PyMeshLab (alpha 0.5%, offset 0.05% of bounding diagonal).
The intermediate SDF mesh still had 858 boundary edges. The wrapped result has
41,090 triangles, one component, zero boundary/non-manifold edges. Preparation
changes the surface and rounds some details; this is not lossless preservation.

SQuadGen: unchanged released weights, 8192 FP32 latents, regularized noise,
dopri5/50 integration points, latent smoothing enabled, edge 0.2%, standard
extraction rings8/a150/div0/t0.1/s3/r15. No polygon cap, count target, or post-run
target refinement. The model chooses layout complexity; no cap does not mean
the model will preserve every detail or emit an arbitrary number of faces.

| Seed | Faces | All quads | Full loop score | Folded quads [1] | Source p99 distance [2] | Pristine faces / quads |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 7,296 | yes | 0.300 | 1.25% | 0.257% | 108 / 38 |
| 1 | 5,792 | yes | 0.126 | 1.73% | 0.179% | 87 / 21 |
| 2 | 5,664 | yes | 0.060 | 2.17% | 0.966% | process failed (3221225477) |
| 3 | 4,864 | yes | 0.070 | 1.91% | 0.208% | 72 / 20 |

[1] Fold check compares the normals of the two triangles across a quad diagonal;
it flags problematic quads, and is not a complete self-intersection test.
[2] 60,000 independently sampled points per direction against the **prepared**
input, distance as percent of its bounding diagonal. The reverse direction and
normal errors are included in benchmark_results.json. This is a percentile,
not a bound on the maximum error and not a guarantee against detail loss.
The separate raw-Pixal3D comparison includes all preprocessing changes. Its
normal statistics are unreliable because the original has inconsistent winding
and overlapping surfaces; they must not be read as quad-output orientation errors.

Seed 0 has the highest full loop score among the all-quad results. A loop
score is min(Fratio,Eratio), including spiraling and self-intersection; it is
not a visual quality percentage. Folded and stretched faces remain, so no
sample is certified artist-ready. The selection is for review only.

The last column replays the **untouched upstream QuadTools extractor** on the
exact same neural fields. The packaged extractor includes index repairs and
consecutive duplicate-corner removal, as well as a disabled-by-default count
budget. These repairs can allow subdivision where upstream stops with a broken
polygon layout. Thus a repaired all-quad output must not be advertised as an
unmodified upstream success. Experimental chart fitting, feature-corner and
region-growth options are OFF in the standard preset.

Two earlier samples on the open SDF table failed: 110 polygons / 16 quads and
75 polygons / 13 quads. The other two open-input samples were cancelled after
these failures. All four wrapped-input samples above were completed.

The upstream fandisk control produced 7,008 quads and full loop score 0.384.
There, packaged and untouched extractors produced identical vertex/face arrays.
The control confirms a functioning pipeline, not universally clean layouts.

Validation: actual ComfyUI generation and four SQuadGen seeds; raw polygon and
edge checks; bidirectional surface sampling; full native loop diagnostics;
upstream saved-field comparisons; packaged saved-field replay; attention
equivalence tests; CPU-only node orchestration/default tests. Tested on Windows
x64/Python3.12/Torch2.8 CUDA12.8/RTX5050 Laptop8GB. Other installations untested.

Safe description for an upload: **Experimental ComfyUI integration of SQuadGen.
Tested on one prepared Pixal3D table and the upstream fandisk example. Generated
quad topology can contain folds, stretched faces and poor loops; manual cleanup
may be needed. Artist-ready topology and lossless detail are not guaranteed.**
