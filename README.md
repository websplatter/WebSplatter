# WebSplatter: Efficient and Faithful In-Browser 3D Gaussian Splatting across Devices via WebGPU 🚀

[**🌐 Live Demo**](https://websplatter.github.io) · [**📄 arXiv**](https://arxiv.org/abs/2602.03207) · [**💻 Code**](https://github.com/yudshj/WebSplatter)

> Accepted to **ACM Multimedia 2026** (the 35th ACM International Conference on Multimedia).

WebSplatter is a WebGPU-native rendering framework for 3D Gaussian Splatting (3DGS) in the browser. Existing web viewers port CUDA-era pipelines whose synchronization and memory patterns do not generalize across GPU architectures, which causes rendering failures on non-NVIDIA and mobile devices. WebSplatter eliminates both failure modes with a **wait-free radix sort** and a **hardware-accelerated rasterization pipeline** with opacity-aware quad sizing.

Evaluated across **eleven devices spanning five GPU architectures** (NVIDIA, Apple, Intel, AMD, Qualcomm), WebSplatter renders every benchmark scene on every tested device without failure, delivering 1.1×–2.5× speedups over the surviving baselines at near-lossless fidelity to the native CUDA renderer (average ΔPSNR 0.07 dB).

## Live Demo 🌐

Try WebSplatter directly in your browser — no installation required:

**[https://websplatter.github.io](https://websplatter.github.io)**

The demo loads the *Van Gogh Room* scene (341K Gaussians) and runs on desktops, laptops, phones and handhelds alike. Click the canvas to focus it, use **WASD** to move, **Shift** to move faster, the **arrow keys** to look around, and **Q/E** to roll. Drag to rotate, right-drag to translate, and scroll to zoom. Camera controls are disabled while animation is enabled.

> Requires a browser with WebGPU enabled: Chrome 113+, Safari on iOS 26 / macOS 26, or Firefox 141+ (Windows only).

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

## Citation 📚

If you find WebSplatter useful in your research, please cite:

```bibtex
@inproceedings{han2026websplatter,
  title={WebSplatter: Efficient and Faithful In-Browser 3D Gaussian Splatting across Devices via WebGPU},
  author={Han, Yudong and Xu, Chao and Ye, Xiaodan and Bi, Weichen and Xu, Xuanhuai and Dong, Zilong and Ma, Yun},
  booktitle={Proceedings of the 35th ACM International Conference on Multimedia},
  year={2026},
  doi={10.1145/3767308.3835220}
}
```

## License 📄

MIT — see [LICENSE](LICENSE).
