# PrioSDK

PrioSDK is a general-purpose SDK for TurboWarp. Right now, its heart is a powerful GPU-powered 3D and 2D renderer, with more to come beyond rendering.

Build scenes from your own meshes, materials, and textures, then choose how to bring them to life: ray tracing, path tracing, or traditional rasterization. HDR lighting, volumetrics, physics, video capture, and stylized 2D/2.5D output give you room to experiment.

## Try it

Download an example below, open it in TurboWarp with **File → Load from your computer**, and approve **Run without sandbox** for the included extension. Click the green flag to begin. The projects carry their own extension and assets, so you can share the `.sb3` itself.

- [Cinematic room](examples/PrioSDK-Gen4-Photoreal-Room-Cinematic.sb3?raw=true) — a still scene that accumulates samples, then stops.
- [Cinematic room · experimental learned reconstruction](examples/PrioSDK-Gen4-Photoreal-Room-Cinematic-Experimental-Learned.sb3?raw=true) — a separate version with trained GPU denoising; optional 2× reconstruction is available through its enhancement block.
- [Realtime room](examples/PrioSDK-Gen4-Photoreal-Room-Realtime.sb3?raw=true) — explore the scene with mouse-drag looking and WASD movement.

For your own project, load [dist/priosdk-gen4.js](dist/priosdk-gen4.js?raw=true) as an unsandboxed custom extension. If you host it yourself, keep the physics module beside it. The examples' visible blocks are a good starting point for building something entirely different.

## A few things to know

PrioSDK needs a WebGPU-capable browser and GPU; it does not run in the standard Scratch editor. Cinematic rendering can take a long time, and speed depends on your device and scene. The examples show the current sample and target, then say Done when a still finishes. Experimental reconstruction runs packaged neural networks on your GPU, but cannot recover every missing detail or guarantee that every image improves.

If you have an older download, download the updated example and open it in a fresh TurboWarp tab. Startup now shows shader compilation and asset progress. Cinematic tiles appear during the unfinished first sample, instead of waiting behind an empty checkerboard; at full quality they can fill in very slowly. Failed setup keeps its first error visible. The **renderer diagnostic report** block provides a copyable runtime, adapter, and startup log; messages also appear in the browser console. Stop cancels pending renderer work without changing your quality settings.

The 'Realtime' example scene was rendered on and optimized for the NVIDIA GeForce RTX 3050 or higher.

This repository contains the ready-to-load builds and three room examples. For the libraries and assets behind them, see the [third-party notices](dist/THIRD-PARTY-NOTICES.md).
