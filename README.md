# WebSplatter: Enabling Cross-Device Efficient Gaussian Splatting in Web Browsers via WebGPU 🚀

This repository is for the paper, **"WebSplatter: Enabling Cross-Device Efficient Gaussian Splatting in Web Browsers via WebGPU"**.

## Live Demo 🌐

Try WebSplatter directly in your browser — no installation required:

- [**Interactive Demo (Van Gogh Room scene)**](https://anonymous.4open.science/w/webgs/?model_url=scenes%2Fvan_gogh_room%2Fvan_gogh_room.ply&camera_url=scenes%2Fvan_gogh_room%2Fcameras.json&clip_sh_degree=3&sort=none&renderer=gaussian&animation=0) — loads a pre-configured scene for immediate viewing
- [**Demo (Scene Selection)**](https://anonymous.4open.science/w/webgs/) — choose from multiple scenes or upload your own `.ply` / `.glb` model

## Model Download 💾

You can download the models from the following URL:
[https://drive.google.com/drive/folders/1WXCpR3kshQt2jmOtuCBsHKfzt1IMqey2](https://drive.google.com/drive/folders/1WXCpR3kshQt2jmOtuCBsHKfzt1IMqey2)

## Supported Formats 📁

- **PLY** — original INRIA 3DGS `.ply` files (loaded directly)
- **glTF/GLB** — with `KHR_gaussian_splatting` extension (uncompressed)
- **glTF/GLB + SPZ** — with `KHR_gaussian_splatting_compression_spz_2` extension (decompressed in-browser via WASM)

### Converting PLY to GLB

Use the included `ply2gltf.py` script to convert PLY files to glTF/GLB:

```bash
# Uncompressed GLB
python ply2gltf.py --input scene.ply --output scene.glb

# SPZ-compressed GLB (much smaller file size)
python ply2gltf.py --input scene.ply --output scene_spz.glb --compression=spz
```

SPZ compression requires the [spz](https://github.com/nianticlabs/spz) Python package.

## Model Preparation 🏗️

Organize your model files in the **`public`** folder as shown below. Both `.ply` and `.glb` formats are supported.

```
public
└── scenes
    ├── bicycle
    │   ├── bicycle_30000.cleaned.ply
    │   ├── bicycle_30000.ply
    │   └── cameras.json
    ├── bonsai
    │   ├── bonsai_30000.ply
    │   └── cameras.json
    ├── garden
    │   ├── cameras.json
    │   └── garden_30000.ply
    ├── train
    │   ├── cameras.json
    │   └── train_30000.ply
    ├── truck
    │   ├── cameras.json
    │   └── truck_30000.ply
    └── van_gogh_room
        ├── cameras.json
        ├── van_gogh_room.glb
        ├── van_gogh_room.ply
        └── van_gogh_room_spz.glb
```

-----

## Installation 📦

First, make sure you have [Node.js](https://nodejs.org/) installed on your system.

Then install the project dependencies using npm:

```bash
npm install
```

## Running the Project 🛠️

After installation, start the development server:

```bash
npm run dev
```

The application will be available in your web browser.
