# Third-party notices

Mothbot Process's own source code is released under the MIT License (see `LICENSE`).
It depends on, and the packaged desktop app bundles, third-party software, models and
data that keep their own licenses. Those licenses are listed below. Each package's full
license text ships inside the package itself (in its `*.dist-info` folder, or in the
packaged app's internal folder).

## Important: AGPL-3.0 components

Object detection uses **Ultralytics** (`ultralytics`, `ultralytics-thop`), licensed under
the **GNU Affero General Public License v3.0**. The MIT License covers Mothbot's own code
only. When Mothbot Process is distributed together with Ultralytics, for example as the
packaged desktop app, the combined work must also respect AGPL-3.0, which requires the
corresponding source code to be available to recipients. This repository being public
satisfies that for Mothbot's part; Ultralytics' source is at
https://github.com/ultralytics/ultralytics. Ultralytics also sells an Enterprise License
for uses where AGPL-3.0 does not fit.

## Python packages

| Package | License |
|---|---|
| ultralytics, ultralytics-thop | AGPL-3.0 |
| torch, torchvision | BSD-3-Clause |
| onnxruntime | MIT |
| onnx | Apache-2.0 |
| opencv-python | Apache-2.0 |
| numpy | BSD-3-Clause (with bundled MIT, Zlib, 0BSD, CC0 parts) |
| scipy, scikit-learn | BSD-3-Clause |
| pandas | BSD-3-Clause |
| polars | MIT |
| pillow | MIT-CMU (HPND) |
| pi-heif | BSD-3-Clause code; bundles libheif (LGPL-3.0) |
| pystray | LGPL-3.0 |
| hdbscan | BSD-3-Clause |
| timm | Apache-2.0 |
| open-clip-torch | MIT |
| pybioclip | MIT |
| rembg | MIT |
| gradio, gradio-client | Apache-2.0 |
| piexif | MIT |
| pyyaml | MIT |
| pyparsing | MIT |
| python-dateutil | Apache-2.0 / BSD-3-Clause (dual) |
| pytz | MIT |
| tzdata | Apache-2.0 |
| regex | Apache-2.0 and CNRI-Python |
| tqdm | MPL-2.0 and MIT |
| babel | BSD-3-Clause |
| packaging | Apache-2.0 or BSD-2-Clause |
| typing-extensions | PSF-2.0 |
| pyinstaller (build only) | GPL-2.0 with the bootloader exception, which allows distributing the built app under any license |

The LGPL-3.0 libraries (pystray, libheif) are used unmodified and remain replaceable
within the packaged app's folder, as the LGPL requires.

## Bundled tools

| Tool | License |
|---|---|
| ExifTool (Phil Harvey), when bundled | Perl Artistic License or GPL, at the user's option |

## Models

| Model | Source | License |
|---|---|---|
| Mothbot Detect (`MBD-*`) and Anybug detectors in `trained_models/` | Digital Naturalism Laboratories, trained with Ultralytics | Released with Mothbot; see the AGPL-3.0 note above for Ultralytics |
| DINOv2 ViT-S/14 (`assets/dinov2_vits14_pretrain.pth`) | Meta AI Research, https://github.com/facebookresearch/dinov2 | Apache-2.0 |
| BioCLIP / BioCLIP 2 (downloaded from Hugging Face on first use) | Imageomics Institute, https://huggingface.co/imageomics | See the model cards (MIT at time of writing) |
| BiRefNet-lite, dynamic ONNX export (`birefnet-lite-dynamic`) | ZhengPeng7/BiRefNet; export by senty-au on Hugging Face | MIT |

## Data

| Data | License |
|---|---|
| GBIF species list in `specieslists/` (download 0024729-260928105237408, DOI 10.15468/dl.4j8kfb) | **CC BY-NC 4.0**: attribution required, non-commercial use only. Cite the DOI when publishing results identified against this list. |

GBIF downloads carry the most restrictive license among their source records. Species lists
you download yourself may carry a different license; check the download page on GBIF.
