# SQuadGen for ComfyUI — experimental

Generates quad layouts with Microsoft's released SQuadGen model. This package
contains the learned-layout node only. The original model weights are required.
It does not include the separate character-cage or QuadWild research backends.

Tested on one prepared Pixal3D table and the upstream fandisk example.
Quad conversion succeeded after preparation and extractor repairs; folded and
stretched faces remain. Artist-ready topology is not verified. See BENCHMARK.md
for all four samples, untouched-upstream comparisons, and measured limitations.

## Installation

Place this folder in ComfyUI/custom_nodes/SQuadGen and install requirements.txt
with the Python used by ComfyUI. Existing Torch/torchvision are not replaced.
The upstream model also imports nvdiffrast and torch-cluster: install compatible
builds for your Torch/CUDA/Python environment following the upstream setup guide.
The test machine used torch-cluster 1.6.3+pt28cu128 and nvdiffrast already present
in its ComfyUI environment. A clean installation on other systems is untested.

Download sqvae_geomae.safetensors and sqdiffuse.safetensors from
https://huggingface.co/microsoft/SQuadGen into ckpts/, then restart ComfyUI.

## Use

Add SQuadGen • Experimental Quad Layout. Select a triangle mesh by absolute path.
The defaults use 8192 latents, FP32, regularized inference, four seeds, and the
standard upstream extraction settings. Latents are not a polygon-count target.
The default triangle budget, output target and face budget are all zero/off.
Results are saved as OBJ/PLY with actual polygon topology and JSON diagnostics.
GLB/viewer triangulation cannot be used to judge whether the stored mesh is quad.
Input textures, UVs and rigs are not transferred.

## Runtime and source

Windows x64 native tools and their corresponding source are included. To rebuild:

    cmake -S QuadTools -B QuadTools/build
    cmake --build QuadTools/build --config Release --target QuadExtraction QuadQuality

The model source retains the Windows CUDA attention shape fix; mathematical
equivalence is checked by verify_attention.py. Native changes include portable
build support, an optional disabled face budget, extraction index repairs and
experimental options which are disabled by the standard extraction preset.
SOURCE_VERSIONS.json records the upstream commits. Model weights are unchanged.

SQuadGen and QuadTools retain their upstream licenses. Vendored third-party
files in QuadTools/src/external retain the notices and licenses in those files.

## Optional preparation and checks

prepare_alpha_wrap.py INPUT OUTPUT.ply wraps a mesh into a closed surface;
it changes geometry and can remove detail. The tested table also received two
native remeshing passes before this step; wrapping alone is not a verified
replacement for that chain. Preparation is optional and never happens silently.

workflows/squadgen_api.json contains the tested API graph. Set mesh_path before
submitting it. workflows/squadgen_workflow.json opens in ComfyUI's canvas.
Set the mesh path and queue. Four generations can take around an hour on an 8GB
laptop GPU; adaptive integration time varies with the sample.

Run python tests/test_node.py for CPU-only orchestration checks, and
python verify_attention.py in the ComfyUI Python environment for attention
equivalence. The existing installed models are used; weights are not bundled.
