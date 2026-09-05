# PrioSDK

PrioSDK is a general-purpose SDK for TurboWarp. Right now, its heart is a powerful GPU-powered 3D and 2D renderer, with more to come beyond rendering.

Build scenes from your own meshes, materials, and textures, then choose how to bring them to life: ray tracing, path tracing, or traditional rasterization. HDR lighting, volumetrics, physics, video capture, and stylized 2D/2.5D output give you room to experiment.

## Try it

Download an example below, open it in TurboWarp with **File → Load from your computer**, and approve **Run without sandbox** for the included extension. Click the green flag to begin. The projects carry their own extension and assets, so you can share the `.sb3` itself.

- [Cinematic room](examples/PrioSDK-Gen4-Photoreal-Room-Cinematic.sb3?raw=true) — a still scene that accumulates samples, then stops.
- [Cinematic room · experimental learned filtering](examples/PrioSDK-Gen4-Photoreal-Room-Cinematic-Experimental-Learned.sb3?raw=true) — a separate version with optional detail-preserving enhancement.
- [Realtime room](examples/PrioSDK-Gen4-Photoreal-Room-Realtime.sb3?raw=true) — explore the scene with mouse-drag looking and WASD movement.

For your own project, load [dist/priosdk-gen4.js](dist/priosdk-gen4.js?raw=true) as an unsandboxed custom extension. If you host it yourself, keep the physics module beside it. The examples' visible blocks are a good starting point for building something entirely different.

## A few things to know

PrioSDK needs a WebGPU-capable browser and GPU; it does not run in the standard Scratch editor. Cinematic rendering can take a long time, and speed depends on your device and scene. Experimental filtering is a tool to try, not a promise that every image improves.

This repository contains the ready-to-load builds and three room examples. For the libraries and assets behind them, see the [third-party notices](dist/THIRD-PARTY-NOTICES.md).
