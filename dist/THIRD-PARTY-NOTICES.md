# Third-party notices

This document records third-party code incorporated into or required by PrioSDK
Gen 4 v0.9.0's 154-opcode TurboWarp extension. It is provided for attribution and
does not change the licenses of the listed projects.

## Rapier 3D

PrioSDK's rigid-body subsystem uses
[`@dimforge/rapier3d-compat`](https://www.npmjs.com/package/@dimforge/rapier3d-compat),
the official JavaScript bindings and embedded WebAssembly compatibility build of
[Rapier](https://rapier.rs/). The lockfile currently resolves version 0.20.0.

Copyright 2020 Dimforge EURL.

Rapier is licensed under the
[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).
The dependency's complete license text is installed at
`node_modules/@dimforge/rapier3d-compat/LICENSE` by `npm ci` and is also available
in the [upstream Rapier repository](https://github.com/dimforge/rapier/blob/master/LICENSE).
Redistributions of the separately built PrioSDK physics module, or any project
that embeds it, contain Rapier code and WebAssembly and must preserve the
applicable Apache-2.0 license and notices. The lean core extension does not parse
or evaluate Rapier until physics is first requested.

`npm run build` writes the renderer/extension core to `dist/priosdk-gen4.js` and
the self-contained Rapier ESM/WASM payload to
`dist/priosdk-gen4-physics.mjs`. It copies the complete license text to
`dist/LICENSE-RAPIER-APACHE-2.0.txt` and this notice to
`dist/THIRD-PARTY-NOTICES.md`. The self-contained `.sb3` generator includes both
notices as archive records, embeds the core as the project's extension URL, and
stores the physics module in a SHA-256-verified hidden PNG carrier. Hosted
distributions must serve the physics module beside the core.

PrioSDK exposes a bounded subset of Rapier. An upstream Rapier capability is not
automatically a supported PrioSDK feature or block.

## AMD FidelityFX Super Resolution 1

The FSR 1 EASU/RCAS WGSL translation derives from AMD's MIT-licensed
[`ffx_fsr1.h`](https://github.com/GPUOpen-Effects/FidelityFX-FSR/blob/master/ffx-fsr/ffx_fsr1.h).
The complete retained notice is
[`NOTICE.fsr1.txt`](NOTICE.fsr1.txt) and is
preserved in generated-bundle legal comments.

The locally authored `fsr3-experimental-*` temporal reconstruction and
interpolation modes are not a port of AMD's FSR 3 SDK and make no FSR 3
conformance claim.

## Poly Haven room assets

The room examples retain CC0 1.0 Universal assets from
[Poly Haven](https://polyhaven.com/). CC0 does not require attribution, but the
source records are preserved for provenance:

- [Mid Century Lounge Chair](https://polyhaven.com/a/mid_century_lounge_chair),
  by Kuutti Siitonen. The retained official 1K glTF geometry is rematerialized by
  the room project; its original texture maps are not retained.
- [Pachira Aquatica 01](https://polyhaven.com/a/pachira_aquatica_01), with model
  and scan credits recorded in the asset notice.
- [Anthurium Botany 01](https://polyhaven.com/a/anthurium_botany_01), retained
  with selected cinematic/realtime geometry and its authored material maps; the
  full scan credits are preserved in its asset notice.
- [Throw Pillows 01](https://polyhaven.com/a/throw_pillows_01), whose retained
  geometry is rematerialized by the room project.
- [Desk Lamp Arm 01](https://polyhaven.com/a/desk_lamp_arm_01), by Kuutti
  Siitonen and Yann Kervran. The cinematic project retains its detailed mesh
  primitives and embeds the official 2K base-color/OpenGL-normal textures plus a
  lossless red-channel roughness image derived from the ARM green channel. The
  original ARM image is retained in the repository for provenance/future schema
  support but is not embedded as a redundant room texture.
- [Classic Laptop](https://polyhaven.com/a/classic_laptop), by Arrangemonk. The
  cinematic project retains its detailed mesh primitives and embeds the official
  2K base-color/OpenGL-normal textures with the same lossless roughness-channel
  extraction. Its original ARM image likewise remains a repository source rather
  than a redundant embedded room texture.
- [Signal Hill Sunrise](https://polyhaven.com/a/signal_hill_sunrise), retained as
  the official 4096x2048 Radiance HDR source used by both current room projects
  exclusively for physical lighting, reflections, diffuse irradiance, and
  captured-sun matching. It is not the camera-visible plate. The retained
  23,098,664-byte file has SHA-256
  `6e0d7eaa0af56d7516fbeacd0dcd7af472966ef2cae6c2aee6b90be99cbff650`.
- [Venice Sunset](https://polyhaven.com/a/venice_sunset), retained as the official
  4096x2048 Radiance HDR panorama used as the camera-only backplate in both room
  variants. The backplate is excluded from environment sampling, reflections,
  diffuse irradiance, and every light path. The retained 22,841,596-byte file has
  SHA-256
  `f243a060f84df1437705d2e50bbdbcce66aa55ddb2d913c18eb3f87873213b10`.

The project-authored `assets/environments/golden-hour-city-panorama.png` remains a
retained SDR asset and is not embedded or loaded by either current room archive.
Its provenance and SDR source range are recorded in
`assets/environments/golden-hour-city-panorama.json`. The optional general camera-
backplate API remains available to other projects.

The active retained Poly Haven scan-material set used by the rooms is:

- [Terlenka](https://polyhaven.com/a/terlenka);
- [Poly Wool Herringbone](https://polyhaven.com/a/poly_wool_herringbone);
- [Oak Wood Planks](https://polyhaven.com/a/oak_wood_planks);
- [Dark Wood](https://polyhaven.com/a/dark_wood);
- [Plastered Wall 04](https://polyhaven.com/a/plastered_wall_04).

Each active retained scan material includes its official base-color, OpenGL
tangent-normal, and scalar roughness image. Dirty Carpet remains in the source
provenance catalog but none of its maps are embedded, loaded, or bound by the
current rooms.

The visible area rug instead uses the aligned project-authored Photoreal Persian
Rug v2 set at 1254x1254 with complete four-sided border and one-to-one UVs:

- base color `photoreal-persian-rug-albedo-v2.png`, SHA-256
  `6c90a4372de193cf28311c7692e42b9dabee132ee531286e06f5441c90613203`;
- OpenGL positive-Y normal `photoreal-persian-rug-normal-gl-v2.png`, SHA-256
  `1de2fba6ffa0c93b7eeffbf1bc5d0da84aca2f0398d6ab07a2023d55f91a72e2`;
- linear-red roughness `photoreal-persian-rug-roughness-v2.png`, SHA-256
  `e1d3695c0f762f1f16025a3c0323e85b9b9c3b6532684411071cb7833339502a`.

Its project-authored manifest is
`assets/textures/photoreal-persian-rug-v2.json`; normal strength is intentionally
restrained to `0.1`, roughness-map strength is `0.32`, and the high base
roughness keeps the surface dry and fibrous instead of wet or smeared.

## Project-authored learned enhancement

The optional learned image-enhancement pass is wholly project-authored. It does
not incorporate or call a third-party model, model service, generative system,
or vendor upscaling/denoising SDK. Its fixed supervised 3x3 log-HDR coefficient
set has SHA-256
`03aaa872aa19b591f87f8c02ee883d3c58818faed2f3538527d04c2506247a6a`;
this identifier is provenance for the bundled coefficients, not a third-party
license notice. The `photoreal-detail`, `balanced`, and `artifact-cleanup`
quality presets all use that same coefficient set and change only authored gate
settings. The AMD FSR 1 notice above applies only to the separate FSR 1
EASU/RCAS implementation.

The complete per-asset records accompany the retained sources under
`assets/models`, `assets/models/photoreal-room-hero`, and `assets/textures`.
Their CC0 dedication is available at
[Creative Commons](https://creativecommons.org/publicdomain/zero/1.0/).
