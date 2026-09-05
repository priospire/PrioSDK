"use strict";
(() => {
  // src/renderer/scene/limits.ts
  var MAX_SCENE_TRIANGLES = 262144;
  var MAX_WIND_DISPLACEMENT = 0.35;
  var NO_TEXTURE_LAYER = 4294967295;
  var MAX_TEXTURE_LAYER = NO_TEXTURE_LAYER - 1;

  // src/renderer/scene/math.ts
  var DEGREES_TO_RADIANS = Math.PI / 180;
  var IDENTITY_TRANSFORM = Object.freeze({
    translation: Object.freeze([0, 0, 0]),
    rotation: Object.freeze([0, 0, 0]),
    scale: Object.freeze([1, 1, 1])
  });

  // src/renderer/scene/validation.ts
  var MAX_MESH_TRIANGLES = MAX_SCENE_TRIANGLES;
  var MAX_MESH_VERTICES = MAX_MESH_TRIANGLES * 3;

  // src/renderer/scene/bvh.ts
  var BVH_MAX_DEPTH = 48;
  var EMPTY_MIN = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  var EMPTY_MAX = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];

  // src/renderer/advanced-shaders.ts
  var INTERSECTION_EPSILON_METERS = 1e-5;
  var RAY_ORIGIN_BIAS_METERS = 1e-4;
  var RAY_TRACE_WORKGROUP_SIZE = 8;
  var CINEMATIC_TRACE_WORKGROUP_SIZE = 1;
  var TRACE_WORKGROUP_INVOCATIONS = RAY_TRACE_WORKGROUP_SIZE * RAY_TRACE_WORKGROUP_SIZE;
  var BVH_TRAVERSAL_STACK_CAPACITY = BVH_MAX_DEPTH + 1;
  var BVH_WORKGROUP_STACK_WORDS = TRACE_WORKGROUP_INVOCATIONS * BVH_TRAVERSAL_STACK_CAPACITY;
  var ANIMATED_TRIANGLE_DEFORMATION_SHADER = (
    /* wgsl */
    `
struct Globals {
  resolution_samples: vec4<f32>,
  output_mode: vec4<f32>,
  camera_position_tan_fov: vec4<f32>,
  camera_forward_exposure: vec4<f32>,
  camera_right_fog_density: vec4<f32>,
  camera_up_fog_height: vec4<f32>,
  fog_color_anisotropy: vec4<f32>,
  render_params: vec4<f32>,
  scene_counts: vec4<f32>,
  light_volume: vec4<f32>,
  previous_position_tan_fov: vec4<f32>,
  previous_forward: vec4<f32>,
  previous_right: vec4<f32>,
  previous_up: vec4<f32>,
  history_params: vec4<f32>,
  environment_params: vec4<f32>,
  ray_type_weights: vec4<f32>,
  path_controls: vec4<f32>,
  environment_texture_params: vec4<f32>,
  environment_irradiance_sh: array<vec4<f32>, 9>,
  environment_sunless_irradiance_sh: array<vec4<f32>, 9>,
  captured_sun_direction: vec4<f32>,
  captured_sun_irradiance: vec4<f32>,
  captured_sun_background_core: vec4<f32>,
  captured_sun_params: vec4<f32>,
  environment_orientation: vec4<f32>,
  backplate_texture_params: vec4<f32>,
  backplate_orientation: vec4<f32>,
  trace_dispatch_tile: vec4<f32>,
  cloud_bounds_min_mode: vec4<f32>,
  cloud_bounds_max_steps: vec4<f32>,
  cloud_scattering_albedo_extinction: vec4<f32>,
  cloud_density_anisotropy: vec4<f32>,
  camera_lens_params: vec4<f32>,
}

struct Triangle {
  v0: vec3<f32>,
  material_id: u32,
  edge1: vec3<f32>,
  wind_weight_0: f32,
  edge2: vec3<f32>,
  wind_weight_1: f32,
  normal_0: vec3<f32>,
  wind_weight_2: f32,
  normal_1: vec3<f32>,
  wind_amplitude: f32,
  normal_2: vec3<f32>,
  shadow_flags: u32,
  uv_0_1: vec4<f32>,
  uv_2_gradient_magnitudes: vec4<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read_write> animated_shadow_bvh: array<u32>;

const PI: f32 = 3.141592653589793;
const MAX_WIND_DISPLACEMENT: f32 = ${MAX_WIND_DISPLACEMENT};
const ANIMATED_SHADOW_BVH_MAGIC: u32 = 0x41534256u;
const ANIMATED_SHADOW_BVH_HEADER_WORDS: u32 = 4u;
const ANIMATED_SHADOW_BVH_NODE_WORDS: u32 = 8u;
const ANIMATED_SHADOW_TRIANGLE_WORDS: u32 = 12u;
const ANIMATED_SHADOW_TRIANGLE_MARKER: u32 = 0xc6979343u;

fn safe_normalize(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
  let length_squared = dot(value, value);
  if (length_squared <= 0.000000000001) {
    return fallback;
  }
  return value * inverseSqrt(length_squared);
}

fn fast_sine(value: f32) -> f32 {
  let wrapped = (fract(value / (2.0 * PI) + 0.5) - 0.5) * (2.0 * PI);
  let coarse = 1.27323954 * wrapped - 0.405284735 * wrapped * abs(wrapped);
  return 0.225 * (coarse * abs(coarse) - coarse) + coarse;
}

fn wind_waveform(position: vec3<f32>) -> vec3<f32> {
  var horizontal_direction = globals.history_params.yz;
  let horizontal_length_squared = dot(horizontal_direction, horizontal_direction);
  if (horizontal_length_squared <= 0.00000001) {
    horizontal_direction = vec2<f32>(1.0, 0.0);
  } else {
    horizontal_direction *= inverseSqrt(horizontal_length_squared);
  }

  let turbulence = clamp(globals.history_params.w, 0.0, 2.0);
  let time = globals.environment_params.y;
  let speed = max(globals.environment_params.w, 0.0);
  let perpendicular = vec2<f32>(-horizontal_direction.y, horizontal_direction.x);
  let along_wind = dot(position.xz, horizontal_direction);
  let across_wind = dot(position.xz, perpendicular);
  let phase = along_wind * 0.31 + time * speed + fast_sine(across_wind * 0.19 + time * 0.37) * turbulence;
  let wave_scalar = clamp(
    fast_sine(phase) * 0.62
      + fast_sine(phase * 1.73 + across_wind * 0.11 + 2.1) * 0.25
      + fast_sine(phase * 0.43 + position.y * 0.27 + 4.3) * 0.13,
    -1.0,
    1.0
  );
  let vertical_motion = fast_sine(phase * 1.31 + across_wind * 0.23) * turbulence * 0.12;
  let wave_direction = safe_normalize(
    vec3<f32>(horizontal_direction.x, vertical_motion, horizontal_direction.y),
    vec3<f32>(horizontal_direction.x, 0.0, horizontal_direction.y)
  );
  return wave_direction * wave_scalar;
}

fn wind_displacement(position: vec3<f32>, weight: f32, amplitude: f32) -> vec3<f32> {
  let strength = clamp(globals.environment_params.z / MAX_WIND_DISPLACEMENT, 0.0, 1.0);
  let safe_amplitude = max(amplitude, 0.0);
  let safe_weight = clamp(weight, 0.0, 1.0);
  if (safe_amplitude <= 0.0 || strength <= 0.0 || safe_weight <= 0.0) {
    return vec3<f32>(0.0);
  }
  return wind_waveform(position) * safe_amplitude * strength * safe_weight;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let word_count = arrayLength(&animated_shadow_bvh);
  if (word_count < ANIMATED_SHADOW_BVH_HEADER_WORDS
      || animated_shadow_bvh[0] != ANIMATED_SHADOW_BVH_MAGIC) {
    return;
  }
  let node_count = animated_shadow_bvh[1];
  let index_offset = animated_shadow_bvh[2];
  let index_count = animated_shadow_bvh[3];
  if (node_count > (word_count - ANIMATED_SHADOW_BVH_HEADER_WORDS) / ANIMATED_SHADOW_BVH_NODE_WORDS
      || index_offset != ANIMATED_SHADOW_BVH_HEADER_WORDS
        + node_count * ANIMATED_SHADOW_BVH_NODE_WORDS
      || index_offset > word_count
      || index_count > word_count - index_offset) {
    return;
  }
  let geometry_offset = index_offset + index_count;
  if (geometry_offset > word_count
      || index_count > (word_count - geometry_offset) / ANIMATED_SHADOW_TRIANGLE_WORDS) {
    return;
  }

  let slot = invocation.x;
  if (slot >= index_count) {
    return;
  }
  let triangle_index = animated_shadow_bvh[index_offset + slot];
  if (triangle_index >= arrayLength(&triangles)) {
    return;
  }
  let triangle = triangles[triangle_index];
  let amplitude = max(triangle.wind_amplitude, 0.0);
  let base_vertex_0 = triangle.v0;
  let base_vertex_1 = triangle.v0 + triangle.edge1;
  let base_vertex_2 = triangle.v0 + triangle.edge2;
  let animated_vertex_0 = base_vertex_0
    + wind_displacement(base_vertex_0, triangle.wind_weight_0, amplitude);
  let animated_vertex_1 = base_vertex_1
    + wind_displacement(base_vertex_1, triangle.wind_weight_1, amplitude);
  let animated_vertex_2 = base_vertex_2
    + wind_displacement(base_vertex_2, triangle.wind_weight_2, amplitude);
  let edge_1 = animated_vertex_1 - animated_vertex_0;
  let edge_2 = animated_vertex_2 - animated_vertex_0;
  let word = geometry_offset + slot * ANIMATED_SHADOW_TRIANGLE_WORDS;
  animated_shadow_bvh[word] = bitcast<u32>(animated_vertex_0.x);
  animated_shadow_bvh[word + 1u] = bitcast<u32>(animated_vertex_0.y);
  animated_shadow_bvh[word + 2u] = bitcast<u32>(animated_vertex_0.z);
  animated_shadow_bvh[word + 4u] = bitcast<u32>(edge_1.x);
  animated_shadow_bvh[word + 5u] = bitcast<u32>(edge_1.y);
  animated_shadow_bvh[word + 6u] = bitcast<u32>(edge_1.z);
  animated_shadow_bvh[word + 8u] = bitcast<u32>(edge_2.x);
  animated_shadow_bvh[word + 9u] = bitcast<u32>(edge_2.y);
  animated_shadow_bvh[word + 10u] = bitcast<u32>(edge_2.z);
  animated_shadow_bvh[word + 7u] = 0u;
  animated_shadow_bvh[word + 11u] = 0u;
  animated_shadow_bvh[word + 3u] = triangle_index ^ ANIMATED_SHADOW_TRIANGLE_MARKER;
}
`
  );
  var ADVANCED_PATH_TRACER_SHADER = (
    /* wgsl */
    `
struct Globals {
  resolution_samples: vec4<f32>,
  output_mode: vec4<f32>,
  camera_position_tan_fov: vec4<f32>,
  camera_forward_exposure: vec4<f32>,
  camera_right_fog_density: vec4<f32>,
  camera_up_fog_height: vec4<f32>,
  fog_color_anisotropy: vec4<f32>,
  render_params: vec4<f32>,
  scene_counts: vec4<f32>,
  light_volume: vec4<f32>,
  previous_position_tan_fov: vec4<f32>,
  previous_forward: vec4<f32>,
  previous_right: vec4<f32>,
  previous_up: vec4<f32>,
  history_params: vec4<f32>,
  environment_params: vec4<f32>,
  ray_type_weights: vec4<f32>,
  path_controls: vec4<f32>,
  environment_texture_params: vec4<f32>,
  environment_irradiance_sh: array<vec4<f32>, 9>,
  environment_sunless_irradiance_sh: array<vec4<f32>, 9>,
  captured_sun_direction: vec4<f32>,
  captured_sun_irradiance: vec4<f32>,
  captured_sun_background_core: vec4<f32>,
  captured_sun_params: vec4<f32>,
  environment_orientation: vec4<f32>,
  backplate_texture_params: vec4<f32>,
  backplate_orientation: vec4<f32>,
  trace_dispatch_tile: vec4<f32>,
  cloud_bounds_min_mode: vec4<f32>,
  cloud_bounds_max_steps: vec4<f32>,
  cloud_scattering_albedo_extinction: vec4<f32>,
  cloud_density_anisotropy: vec4<f32>,
  camera_lens_params: vec4<f32>,
}

struct Sphere {
  center_radius: vec4<f32>,
  color_roughness: vec4<f32>,
  material: vec4<f32>,
}

struct BvhNode {
  bounds_min: vec3<f32>,
  left_first: u32,
  bounds_max: vec3<f32>,
  count: u32,
}

struct Triangle {
  v0: vec3<f32>,
  material_id: u32,
  edge1: vec3<f32>,
  wind_weight_0: f32,
  edge2: vec3<f32>,
  wind_weight_1: f32,
  normal_0: vec3<f32>,
  wind_weight_2: f32,
  normal_1: vec3<f32>,
  wind_amplitude: f32,
  normal_2: vec3<f32>,
  shadow_flags: u32,
  uv_0_1: vec4<f32>,
  uv_2_gradient_magnitudes: vec4<f32>,
}

struct Material {
  base_color: vec3<f32>,
  roughness: f32,
  emission_color: vec3<f32>,
  emission_strength: f32,
  metallic: f32,
  transmission: f32,
  ior: f32,
  alpha: f32,
  flags: u32,
  wind_influence: f32,
  color_variation: f32,
  detail_scale: f32,
  roughness_variation: f32,
  normal_strength: f32,
  pbr_texture_layers: vec2<u32>,
  texture_layer: u32,
  texture_flags: u32,
  uv_repeat: vec2<f32>,
  atlas_offset: vec2<f32>,
  atlas_scale: vec2<f32>,
  texture_strength: f32,
  texture_normal_strength: f32,
  texture_roughness_strength: f32,
  texture_lod: f32,
}

struct SurfaceSample {
  albedo: vec3<f32>,
  normal: vec3<f32>,
  roughness: f32,
  unresolved_normal_variance: f32,
}

struct Light {
  vector: vec3<f32>,
  kind: u32,
  color: vec3<f32>,
  intensity: f32,
  range_or_angular_radius: f32,
  radius: f32,
  flags: u32,
  reserved: u32,
}

struct Ray {
  origin: vec3<f32>,
  direction: vec3<f32>,
  cone_width: f32,
  background_visibility: u32,
}

struct Hit {
  t: f32,
  position: vec3<f32>,
  normal: vec3<f32>,
  geometric_normal: vec3<f32>,
  albedo: vec3<f32>,
  roughness: f32,
  emission: vec3<f32>,
  metallic: f32,
  transmission: f32,
  ior: f32,
  material_id: u32,
  material_flags: u32,
  hit: u32,
  front_face: u32,
  reactive: f32,
}

struct TriangleGeometryHit {
  t: f32,
  u: f32,
  v: f32,
  triangle_index: u32,
  front_face: u32,
  hit: u32,
}

struct AnimatedTriangleGeometry {
  vertex_0: vec3<f32>,
  edge_1: vec3<f32>,
  edge_2: vec3<f32>,
  cached: u32,
}

struct ShadowHit {
  t: f32,
  transmission: f32,
  hit: u32,
  padding: u32,
  tint: vec3<f32>,
  reserved: f32,
}

struct SunVisibilityCache {
  visibility: vec3<f32>,
  direction: vec3<f32>,
  age: u32,
  enabled: u32,
  valid: u32,
}

struct PackedSunVisibility {
  visibility_age: u32,
  direction: u32,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(2) var previous_frame: texture_2d<f32>;
@group(0) @binding(3) var output_frame: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<storage, read> bvh_nodes: array<BvhNode>;
@group(0) @binding(5) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(6) var<storage, read> materials: array<Material>;
@group(0) @binding(7) var<storage, read> lights: array<Light>;
@group(0) @binding(8) var normal_depth: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var albedo_roughness: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(10) var motion_reactive: texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var material_textures: texture_2d_array<f32>;
@group(0) @binding(12) var material_texture_sampler: sampler;
@group(0) @binding(13) var environment_texture: texture_2d<f32>;
@group(0) @binding(14) var material_linear_textures: texture_2d_array<f32>;
@group(0) @binding(15) var sunless_environment_texture: texture_2d<f32>;
@group(0) @binding(16) var backplate_texture: texture_2d<f32>;
@group(0) @binding(17) var<storage, read> previous_sun_visibility: array<PackedSunVisibility>;
@group(0) @binding(18) var<storage, read_write> output_sun_visibility: array<PackedSunVisibility>;
@group(0) @binding(19) var previous_normal_depth: texture_2d<f32>;
@group(0) @binding(20) var<storage, read> animated_shadow_bvh: array<u32>;
@group(0) @binding(21) var cloud_density_texture: texture_3d<f32>;
@group(0) @binding(22) var cloud_density_sampler: sampler;

const PI: f32 = 3.141592653589793;
const INTERSECTION_EPSILON: f32 = ${INTERSECTION_EPSILON_METERS};
const RAY_ORIGIN_BIAS: f32 = ${RAY_ORIGIN_BIAS_METERS};
const MAX_BOUNCES: u32 = 16u;
const MAX_SAMPLES_PER_FRAME: u32 = 8u;
const MAX_SPHERES: u32 = 256u;
const MAX_LIGHTS: u32 = 32u;
const BVH_STACK_CAPACITY: u32 = ${BVH_TRAVERSAL_STACK_CAPACITY}u;
const BVH_STACK_BOUNDS_ACCEPTED_FLAG: u32 = 0x80000000u;
const BVH_STACK_NODE_INDEX_MASK: u32 = 0x7fffffffu;
const MAX_VOLUME_STEPS: u32 = 64u;
const MIN_ADAPTIVE_VOLUME_STEPS: u32 = 8u;
const MAX_CLOUD_STEPS: u32 = 96u;
const MAX_CLOUD_LIGHT_STEPS: u32 = 16u;
const SUN_VISIBILITY_STATIC_REFRESH_PERIOD: u32 = 16u;
const SUN_VISIBILITY_NORMAL_COSINE: f32 = 0.985;
const SHADOW_FILTER_ALL: u32 = 0u;
const SHADOW_FILTER_STATIC: u32 = 1u;
const SHADOW_FILTER_ANIMATED: u32 = 2u;
const BVH_NODE_PRIMITIVE_COUNT_MASK: u32 = 0x3fffffffu;
const BVH_NODE_HAS_STATIC_SHADOW_TRIANGLES: u32 = 0x40000000u;
const BVH_NODE_HAS_ANIMATED_SHADOW_TRIANGLES: u32 = 0x80000000u;
const ANIMATED_SHADOW_BVH_MAGIC: u32 = 0x41534256u;
const ANIMATED_SHADOW_BVH_HEADER_WORDS: u32 = 4u;
const ANIMATED_SHADOW_BVH_NODE_WORDS: u32 = 8u;
const ANIMATED_SHADOW_TRIANGLE_WORDS: u32 = 12u;
const ANIMATED_SHADOW_TRIANGLE_MARKER: u32 = 0xc6979343u;
const TRIANGLE_ANIMATED_SHADOW_SLOT_SHIFT: u32 = 3u;
const TRIANGLE_ANIMATED_SHADOW_SLOT_MASK: u32 = 0x003ffff8u;
const MAX_SAFE_RADIANCE: f32 = 65504.0;
const NO_LIGHT_INDEX: u32 = 0xffffffffu;
const NO_MATERIAL_INDEX: u32 = 0xffffffffu;
const SOLAR_DISC_PRESERVE_ROUGHNESS_START: f32 = 0.04;
const SOLAR_DISC_PRESERVE_ROUGHNESS_END: f32 = 0.12;
const MAX_WIND_DISPLACEMENT: f32 = ${MAX_WIND_DISPLACEMENT};
const MATERIAL_DOUBLE_SIDED: u32 = 1u;
const MATERIAL_FOLIAGE: u32 = 2u;
const MATERIAL_BARK: u32 = 4u;
const MATERIAL_GRASS: u32 = 8u;
const MATERIAL_GROUND: u32 = 16u;
const MATERIAL_PLASTER: u32 = 32u;
const MATERIAL_WOOD: u32 = 64u;
const MATERIAL_FABRIC: u32 = 128u;
const MATERIAL_GLASS: u32 = 256u;
const MATERIAL_UNLIT: u32 = 512u;
const MATERIAL_SOLID_GLASS: u32 = 1024u;
const TRIANGLE_SHADOW_CACHE_VALID: u32 = 1u;
const TRIANGLE_SHADOW_ACCEPTS_BACKFACE: u32 = 2u;
const TRIANGLE_SHADOW_OPAQUE: u32 = 4u;
const NO_TEXTURE_LAYER: u32 = 0xffffffffu;
const MATERIAL_TEXTURE_BASE_COLOR: u32 = 1u;
const MATERIAL_TEXTURE_LUMINANCE_NORMAL: u32 = 2u;
var<workgroup> bvh_traversal_stack: array<u32, ${BVH_WORKGROUP_STACK_WORDS}>;
const MATERIAL_TEXTURE_LUMINANCE_ROUGHNESS: u32 = 4u;
const MATERIAL_TEXTURE_INDEPENDENT_NORMAL: u32 = 8u;
const MATERIAL_TEXTURE_INDEPENDENT_ROUGHNESS: u32 = 16u;
const ATLAS_MIP_GUARD_LEVELS: f32 = 2.0;
const ATLAS_MIP_GUARD_TEXELS: f32 = 2.0;
const BACKPLATE_TRANSMISSION_ROUGHNESS_MAX: f32 = 0.04;
const BACKPLATE_TRANSMISSION_MINIMUM: f32 = 0.5;
const BACKPLATE_GBUFFER_ROUGHNESS_MARKER: f32 = 0.00784313725490196;

fn ray_cone_width_at_distance(ray: Ray, distance: f32) -> f32 {
  if (globals.environment_texture_params.w <= 0.5) {
    return 0.0;
  }
  let internal_height = max(globals.resolution_samples.y, 1.0);
  let pixel_spread = 2.0 * globals.camera_position_tan_fov.w / internal_height;
  return max(ray.cone_width, 0.0) + max(distance, 0.0) * pixel_spread;
}
const MATERIAL_LEGACY_DETAIL_MASK: u32 = 30u;
const MATERIAL_INTERIOR_DETAIL_MASK: u32 = 496u;

fn sanitize_component(value: f32) -> f32 {
  if (value != value) {
    return 0.0;
  }
  return clamp(value, 0.0, MAX_SAFE_RADIANCE);
}

fn sanitize_radiance(value: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    sanitize_component(value.x),
    sanitize_component(value.y),
    sanitize_component(value.z)
  );
}

fn hash_u32(value: u32) -> u32 {
  var result = value + 0x9e3779b9u;
  result = (result ^ (result >> 16u)) * 0x21f0aaadu;
  result = (result ^ (result >> 15u)) * 0x735a2d97u;
  return result ^ (result >> 15u);
}

fn random(state: ptr<function, u32>) -> f32 {
  *state = hash_u32(*state);
  return f32(*state) * (1.0 / 4294967296.0);
}

fn default_sun_direction() -> vec3<f32> {
  return normalize(vec3<f32>(0.48, 0.82, 0.31));
}

fn primary_directional_light_index() -> u32 {
  let light_count = min(min(u32(globals.light_volume.x), arrayLength(&lights)), MAX_LIGHTS);
  for (var light_index = 0u; light_index < MAX_LIGHTS; light_index += 1u) {
    if (light_index >= light_count) {
      break;
    }
    if (lights[light_index].kind != 0u) {
      return light_index;
    }
  }
  return NO_LIGHT_INDEX;
}

fn environment_source_direction(world_direction: vec3<f32>) -> vec3<f32> {
  let safe_direction = normalize(world_direction);
  let yaw = globals.environment_texture_params.z;
  let yaw_cosine = cos(yaw);
  let yaw_sine = sin(yaw);
  let yaw_local_direction = vec3<f32>(
    yaw_cosine * safe_direction.x - yaw_sine * safe_direction.z,
    safe_direction.y,
    yaw_sine * safe_direction.x + yaw_cosine * safe_direction.z
  );
  let pitch_cosine = globals.environment_orientation.y;
  let pitch_sine = globals.environment_orientation.z;
  return vec3<f32>(
    yaw_local_direction.x,
    pitch_cosine * yaw_local_direction.y - pitch_sine * yaw_local_direction.z,
    pitch_sine * yaw_local_direction.y + pitch_cosine * yaw_local_direction.z
  );
}

fn backplate_source_direction(world_direction: vec3<f32>) -> vec3<f32> {
  let safe_direction = normalize(world_direction);
  let yaw = globals.backplate_texture_params.z;
  let yaw_cosine = cos(yaw);
  let yaw_sine = sin(yaw);
  let yaw_local_direction = vec3<f32>(
    yaw_cosine * safe_direction.x - yaw_sine * safe_direction.z,
    safe_direction.y,
    yaw_sine * safe_direction.x + yaw_cosine * safe_direction.z
  );
  let pitch_cosine = globals.backplate_orientation.y;
  let pitch_sine = globals.backplate_orientation.z;
  return vec3<f32>(
    yaw_local_direction.x,
    pitch_cosine * yaw_local_direction.y - pitch_sine * yaw_local_direction.z,
    pitch_sine * yaw_local_direction.y + pitch_cosine * yaw_local_direction.z
  );
}

fn environment_uv_for_dimensions(
  direction: vec3<f32>,
  lod_dimensions: vec2<u32>
) -> vec2<f32> {
  let source_direction = environment_source_direction(direction);
  let longitude = atan2(source_direction.z, source_direction.x);
  let half_texel_v = 0.5 / max(f32(lod_dimensions.y), 1.0);
  return vec2<f32>(
    fract(longitude / (2.0 * PI) + 0.5),
    clamp(
      acos(clamp(source_direction.y, -1.0, 1.0)) / PI,
      half_texel_v,
      1.0 - half_texel_v
    )
  );
}

fn backplate_uv_for_dimensions(
  direction: vec3<f32>,
  lod_dimensions: vec2<u32>
) -> vec2<f32> {
  let source_direction = backplate_source_direction(direction);
  let longitude = atan2(source_direction.z, source_direction.x);
  let half_texel_v = 0.5 / max(f32(lod_dimensions.y), 1.0);
  return vec2<f32>(
    fract(longitude / (2.0 * PI) + 0.5),
    clamp(
      acos(clamp(source_direction.y, -1.0, 1.0)) / PI,
      half_texel_v,
      1.0 - half_texel_v
    )
  );
}

fn camera_visible_equirectangular_lod(
  source_direction: vec3<f32>,
  base_dimensions: vec2<u32>,
  mip_count: u32
) -> f32 {
  if (globals.environment_texture_params.w <= 0.5) {
    return 0.0;
  }
  let maximum_lod = f32(max(mip_count, 1u) - 1u);
  let internal_height = max(globals.resolution_samples.y, 1.0);
  let vertical_field_of_view = 2.0 * atan(max(globals.camera_position_tan_fov.w, 0.000001));
  let angular_pixel_height = vertical_field_of_view / internal_height;
  let vertical_footprint = angular_pixel_height * f32(base_dimensions.y) / PI;
  let latitude_cosine = max(length(source_direction.xz), 0.25);
  let horizontal_footprint = angular_pixel_height
    * f32(base_dimensions.x)
    / (2.0 * PI * latitude_cosine);
  let texel_footprint = max(vertical_footprint, horizontal_footprint);
  return clamp(log2(max(texel_footprint, 1.0)), 0.0, maximum_lod);
}

fn environment_radiance_lod(
  direction: vec3<f32>,
  lod: f32,
  include_procedural_sun: bool
) -> vec3<f32> {
  let sky_factor = smoothstep(-0.15, 0.8, direction.y);
  let sky = mix(
    vec3<f32>(0.78, 0.87, 1.02),
    vec3<f32>(0.12, 0.32, 0.82),
    sky_factor
  );
  let lower_hemisphere = vec3<f32>(0.035, 0.045, 0.065);
  let horizon_blend = smoothstep(-0.18, 0.18, direction.y);
  var base = mix(lower_hemisphere, sky, horizon_blend);
  if (globals.environment_texture_params.x > 0.5) {
    let maximum_lod = f32(max(textureNumLevels(environment_texture), 1u) - 1u);
    let clamped_lod = clamp(lod, 0.0, maximum_lod);
    let lod_dimensions = textureDimensions(environment_texture, i32(ceil(clamped_lod)));
    let environment_uv = environment_uv_for_dimensions(direction, lod_dimensions);
    base = max(
      textureSampleLevel(
        environment_texture,
        material_texture_sampler,
        environment_uv,
        clamped_lod
      ).rgb,
      vec3<f32>(0.0)
    ) * max(globals.environment_texture_params.y, 0.0);
  }
  let directional_index = primary_directional_light_index();
  var sun_direction = default_sun_direction();
  var sun_color = vec3<f32>(1.0, 0.78, 0.48);
  var sun_intensity = 24.0;
  if (directional_index != NO_LIGHT_INDEX) {
    let directional = lights[directional_index];
    sun_direction = normalize(-directional.vector);
    sun_color = max(directional.color, vec3<f32>(0.0));
    sun_intensity = max(directional.intensity, 0.0) * 3.5;
  }
  let procedural_sun_weight = select(
    0.0,
    1.0,
    include_procedural_sun && globals.environment_texture_params.x <= 0.5
  );
  let sun = pow(max(dot(direction, sun_direction), 0.0), 1024.0)
    * sun_intensity
    * procedural_sun_weight;
  return (base + sun_color * sun) * max(globals.environment_params.x, 0.0);
}

fn environment(direction: vec3<f32>) -> vec3<f32> {
  return environment_radiance_lod(direction, 0.0, true);
}

fn solar_disc_preservation_for_roughness(roughness: f32) -> f32 {
  return 1.0 - smoothstep(
    SOLAR_DISC_PRESERVE_ROUGHNESS_START,
    SOLAR_DISC_PRESERVE_ROUGHNESS_END,
    clamp(roughness, 0.0, 1.0)
  );
}

fn captured_sun_matching_active() -> bool {
  return globals.captured_sun_direction.w > 0.5
    && globals.captured_sun_params.w > 0.5
    && globals.environment_texture_params.x > 0.5;
}

fn captured_sun_world_direction() -> vec3<f32> {
  return safe_normalize(
    globals.captured_sun_direction.xyz,
    default_sun_direction()
  );
}

fn captured_sun_environment_scale() -> f32 {
  return max(globals.environment_texture_params.y, 0.0)
    * max(globals.environment_params.x, 0.0);
}

fn sunless_environment_lod_offset() -> f32 {
  let full_base_dimensions = textureDimensions(environment_texture, 0);
  let sunless_base_dimensions = textureDimensions(sunless_environment_texture, 0);
  return log2(max(
    f32(full_base_dimensions.y) / max(f32(sunless_base_dimensions.y), 1.0),
    1.0
  ));
}

fn sunless_environment_radiance_lod(direction: vec3<f32>, full_environment_lod: f32) -> vec3<f32> {
  let lod_offset = sunless_environment_lod_offset();
  let maximum_lod = f32(max(textureNumLevels(sunless_environment_texture), 1u) - 1u);
  let adjusted_lod = clamp(full_environment_lod - lod_offset, 0.0, maximum_lod);
  let lod_dimensions = textureDimensions(sunless_environment_texture, i32(ceil(adjusted_lod)));
  let environment_uv = environment_uv_for_dimensions(direction, lod_dimensions);
  return max(textureSampleLevel(
    sunless_environment_texture,
    material_texture_sampler,
    environment_uv,
    adjusted_lod
  ).rgb, vec3<f32>(0.0))
    * captured_sun_environment_scale();
}

fn partitioned_environment_radiance_lod(
  direction: vec3<f32>,
  lod: f32,
  solar_disc_preservation: f32,
  include_procedural_sun: bool
) -> vec3<f32> {
  let full_environment = environment_radiance_lod(direction, lod, include_procedural_sun);
  let preservation = clamp(solar_disc_preservation, 0.0, 1.0);
  if (!captured_sun_matching_active() || preservation >= 0.9999) {
    return full_environment;
  }
  if (lod >= sunless_environment_lod_offset()) {
    let sunless_environment = sunless_environment_radiance_lod(direction, lod);
    return max(mix(sunless_environment, full_environment, preservation), vec3<f32>(0.0));
  }

  let sun_direction = captured_sun_world_direction();
  let safe_direction = safe_normalize(direction, sun_direction);
  let alignment = dot(safe_direction, sun_direction);
  let stored_core = clamp(globals.captured_sun_background_core.w, 0.0001, PI - 0.001);
  let stored_extraction = clamp(
    max(globals.captured_sun_params.x, stored_core + 0.0001),
    stored_core + 0.0001,
    PI - 0.0001
  );
  let base_spatial_weight = smoothstep(
    cos(stored_extraction),
    cos(stored_core),
    alignment
  );
  let environment_scale = captured_sun_environment_scale();
  let local_background = max(globals.captured_sun_background_core.xyz, vec3<f32>(0.0))
    * environment_scale;
  let available_excess = max(full_environment - local_background, vec3<f32>(0.0));
  let captured_excess = available_excess * base_spatial_weight;

  return max(
    full_environment - captured_excess * (1.0 - preservation),
    vec3<f32>(0.0)
  );
}

fn stochastic_environment_radiance(
  direction: vec3<f32>,
  solar_disc_preservation: f32
) -> vec3<f32> {
  return partitioned_environment_radiance_lod(
    direction,
    0.0,
    solar_disc_preservation,
    true
  );
}

fn backplate_transmission_preserves_visibility(hit: Hit) -> bool {
  return hit.transmission >= BACKPLATE_TRANSMISSION_MINIMUM
    && hit.metallic < 0.999999
    && hit.roughness <= BACKPLATE_TRANSMISSION_ROUGHNESS_MAX
    && !hit_is_foliage(hit)
    && surface_transport_roughness(hit) <= BACKPLATE_TRANSMISSION_ROUGHNESS_MAX;
}

fn backplate_radiance(direction: vec3<f32>) -> vec3<f32> {
  let base_dimensions = textureDimensions(backplate_texture, 0);
  let source_direction = backplate_source_direction(direction);
  let lod = camera_visible_equirectangular_lod(
    source_direction,
    base_dimensions,
    textureNumLevels(backplate_texture)
  );
  let lod_dimensions = textureDimensions(backplate_texture, i32(ceil(lod)));
  let uv = backplate_uv_for_dimensions(direction, lod_dimensions);
  return sanitize_radiance(
    max(textureSampleLevel(
      backplate_texture,
      material_texture_sampler,
      uv,
      lod
    ).rgb, vec3<f32>(0.0)) * max(globals.backplate_texture_params.y, 0.0)
  );
}

fn camera_visible_environment_radiance(
  ray: Ray,
  solar_disc_preservation: f32
) -> vec3<f32> {
  if (ray.background_visibility != 0u && globals.backplate_texture_params.x > 0.5) {
    return backplate_radiance(ray.direction);
  }
  let environment_base_dimensions = textureDimensions(environment_texture, 0);
  let environment_lod = camera_visible_equirectangular_lod(
    environment_source_direction(ray.direction),
    environment_base_dimensions,
    textureNumLevels(environment_texture)
  );
  return partitioned_environment_radiance_lod(
    ray.direction,
    environment_lod,
    solar_disc_preservation,
    true
  );
}

fn captured_environment_sun_irradiance() -> vec3<f32> {
  if (!captured_sun_matching_active()) {
    return vec3<f32>(0.0);
  }
  return max(globals.captured_sun_irradiance.xyz, vec3<f32>(0.0))
    * captured_sun_environment_scale();
}

fn diffuse_environment_sh_coefficient(index: u32) -> vec3<f32> {
  if (captured_sun_matching_active()) {
    return globals.environment_sunless_irradiance_sh[index].xyz;
  }
  return globals.environment_irradiance_sh[index].xyz;
}

fn diffuse_environment_irradiance(normal_value: vec3<f32>) -> vec3<f32> {
  let normal = safe_normalize(normal_value, vec3<f32>(0.0, 1.0, 0.0));
  if (globals.environment_texture_params.x <= 0.5) {
    let local_sky = environment_radiance_lod(normal, 0.0, false);
    let upper_sky = environment_radiance_lod(vec3<f32>(0.0, 1.0, 0.0), 0.0, false);
    return sanitize_radiance((local_sky * 0.72 + upper_sky * 0.28) * PI);
  }

  let direction = environment_source_direction(normal);
  let x = direction.x;
  let y = direction.y;
  let z = direction.z;
  var irradiance = diffuse_environment_sh_coefficient(0u) * 0.2820947918;
  irradiance += diffuse_environment_sh_coefficient(1u) * (0.4886025119 * y);
  irradiance += diffuse_environment_sh_coefficient(2u) * (0.4886025119 * z);
  irradiance += diffuse_environment_sh_coefficient(3u) * (0.4886025119 * x);
  irradiance += diffuse_environment_sh_coefficient(4u) * (1.0925484306 * x * y);
  irradiance += diffuse_environment_sh_coefficient(5u) * (1.0925484306 * y * z);
  irradiance += diffuse_environment_sh_coefficient(6u) * (0.3153915653 * (3.0 * z * z - 1.0));
  irradiance += diffuse_environment_sh_coefficient(7u) * (1.0925484306 * x * z);
  irradiance += diffuse_environment_sh_coefficient(8u) * (0.5462742153 * (x * x - y * y));
  return sanitize_radiance(
    max(irradiance, vec3<f32>(0.0))
      * max(globals.environment_texture_params.y, 0.0)
      * max(globals.environment_params.x, 0.0)
  );
}

fn safe_normalize(value: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
  let length_squared = dot(value, value);
  if (length_squared <= 0.000000000001) {
    return fallback;
  }
  return value * inverseSqrt(length_squared);
}

fn procedural_wave(position: vec3<f32>) -> f32 {
  return clamp(
    0.5
      + sin(dot(position, vec3<f32>(1.37, 2.11, 0.73))) * 0.24
      + sin(dot(position, vec3<f32>(-0.61, 1.19, 2.53)) + 1.7) * 0.16
      + sin(dot(position, vec3<f32>(2.83, -0.47, 1.31)) + 4.1) * 0.10,
    0.0,
    1.0
  );
}

fn microdetail_has_uv(uv_gradient_magnitudes: vec2<f32>) -> bool {
  return max(uv_gradient_magnitudes.x, uv_gradient_magnitudes.y) > 0.0000001;
}

fn stable_microdetail_position(
  position: vec3<f32>,
  uv: vec2<f32>,
  uv_gradient_magnitudes: vec2<f32>
) -> vec3<f32> {
  if (microdetail_has_uv(uv_gradient_magnitudes)) {
    return vec3<f32>(uv, dot(uv, vec2<f32>(0.371, 0.613)));
  }
  return position;
}

fn procedural_surface_color(
  material: Material,
  position: vec3<f32>,
  uv: vec2<f32>,
  uv_gradient_magnitudes: vec2<f32>
) -> vec3<f32> {
  let amount = clamp(material.color_variation, 0.0, 1.0);
  if (amount <= 0.0 || (material.flags & MATERIAL_LEGACY_DETAIL_MASK) == 0u) {
    return max(material.base_color, vec3<f32>(0.0));
  }
  let scale = clamp(material.detail_scale, 0.001, 10000.0);
  let detail_position = stable_microdetail_position(
    position,
    uv,
    uv_gradient_magnitudes
  ) * scale;
  var centered = 0.0;
  var response = vec3<f32>(0.0);

  if ((material.flags & MATERIAL_FOLIAGE) != 0u) {
    centered = procedural_wave(detail_position) * 2.0 - 1.0;
    let mottling = sin(
      detail_position.x * 1.7
        + detail_position.z * 2.3
        + sin(detail_position.y * 1.1) * 0.8
    );
    centered = clamp(centered * 0.72 + mottling * 0.28, -1.0, 1.0);
    response = vec3<f32>(centered * 0.34, centered * 0.52, centered * 0.28);
  } else if ((material.flags & MATERIAL_BARK) != 0u) {
    centered = procedural_wave(detail_position) * 2.0 - 1.0;
    let ridges = sin((detail_position.x + detail_position.z) * 5.0 + sin(detail_position.y * 0.65));
    centered = clamp(centered * 0.42 + ridges * 0.58, -1.0, 1.0);
    response = vec3<f32>(centered * 0.46, centered * 0.34, centered * 0.24);
  } else if ((material.flags & MATERIAL_GRASS) != 0u) {
    centered = procedural_wave(detail_position) * 2.0 - 1.0;
    let blades = sin(detail_position.x * 3.1 + detail_position.z * 4.7 + detail_position.y * 0.9);
    centered = clamp(centered * 0.64 + blades * 0.36, -1.0, 1.0);
    response = vec3<f32>(centered * 0.30, centered * 0.48, centered * 0.22);
  } else if ((material.flags & MATERIAL_GROUND) != 0u) {
    let patches = smooth_value_noise(
      detail_position * vec3<f32>(0.431, 0.173, 0.619)
        + vec3<f32>(11.731, 5.317, 23.119)
    ) * 2.0 - 1.0;
    centered = clamp(patches, -1.0, 1.0);
    response = vec3<f32>(centered * 0.30);
  }

  return clamp(
    material.base_color * (vec3<f32>(1.0) + response * amount),
    vec3<f32>(0.0),
    vec3<f32>(1.0)
  );
}

fn fast_sine(value: f32) -> f32 {
  let wrapped = (fract(value / (2.0 * PI) + 0.5) - 0.5) * (2.0 * PI);
  let coarse = 1.27323954 * wrapped - 0.405284735 * wrapped * abs(wrapped);
  return 0.225 * (coarse * abs(coarse) - coarse) + coarse;
}

fn dominant_planar_coordinates(position: vec3<f32>, normal_value: vec3<f32>) -> vec2<f32> {
  let absolute_normal = abs(normal_value);
  if (absolute_normal.y >= absolute_normal.x && absolute_normal.y >= absolute_normal.z) {
    return position.xz;
  }
  if (absolute_normal.x >= absolute_normal.z) {
    return position.yz;
  }
  return position.xy;
}

fn stable_microdetail_planar_coordinates(
  position: vec3<f32>,
  normal_value: vec3<f32>,
  uv: vec2<f32>,
  uv_gradient_magnitudes: vec2<f32>
) -> vec2<f32> {
  if (microdetail_has_uv(uv_gradient_magnitudes)) {
    return uv;
  }
  return dominant_planar_coordinates(position, normal_value);
}

fn lattice_noise_hash(cell: vec3<i32>) -> f32 {
  var value = hash_u32(bitcast<u32>(cell.x) ^ 0x68bc21ebu);
  value = hash_u32(value ^ bitcast<u32>(cell.y) ^ 0x02e5be93u);
  value = hash_u32(value ^ bitcast<u32>(cell.z) ^ 0x967a889bu);
  return f32(value) * (1.0 / 4294967296.0);
}

fn smooth_value_noise(position: vec3<f32>) -> f32 {
  let base = floor(position);
  let cell = vec3<i32>(base);
  let fraction = position - base;
  let fade = fraction * fraction * (vec3<f32>(3.0) - 2.0 * fraction);
  let x00 = mix(
    lattice_noise_hash(cell + vec3<i32>(0, 0, 0)),
    lattice_noise_hash(cell + vec3<i32>(1, 0, 0)),
    fade.x
  );
  let x10 = mix(
    lattice_noise_hash(cell + vec3<i32>(0, 1, 0)),
    lattice_noise_hash(cell + vec3<i32>(1, 1, 0)),
    fade.x
  );
  let x01 = mix(
    lattice_noise_hash(cell + vec3<i32>(0, 0, 1)),
    lattice_noise_hash(cell + vec3<i32>(1, 0, 1)),
    fade.x
  );
  let x11 = mix(
    lattice_noise_hash(cell + vec3<i32>(0, 1, 1)),
    lattice_noise_hash(cell + vec3<i32>(1, 1, 1)),
    fade.x
  );
  return mix(mix(x00, x10, fade.y), mix(x01, x11, fade.y), fade.z);
}

fn detail_octave_visibility(
  world_footprint: f32,
  detail_scale: f32,
  cycles_per_detail_unit: f32
) -> f32 {
  let projected_width = max(world_footprint, 0.0)
    * clamp(detail_scale, 0.001, 10000.0)
    * max(cycles_per_detail_unit, 0.0);
  return 1.0 - smoothstep(0.18, 0.72, projected_width);
}

fn irregular_detail_coordinates(position: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    dot(position, vec3<f32>(0.819172, 0.312083, 0.480871)),
    dot(position, vec3<f32>(-0.420921, 0.901137, 0.105233)),
    dot(position, vec3<f32>(-0.388127, -0.298173, 0.872641))
  );
}

fn interior_detail_octave_weights(
  world_footprint: f32,
  detail_scale: f32
) -> vec3<f32> {
  return vec3<f32>(
    detail_octave_visibility(world_footprint, detail_scale, 0.347),
    detail_octave_visibility(world_footprint, detail_scale, 1.731),
    select(
      0.0,
      detail_octave_visibility(world_footprint, detail_scale, 6.613),
      PIPELINE_PATH_TRACING_ENABLED
    )
  );
}

fn interior_path_detail_visibility(
  world_footprint: f32,
  detail_scale: f32
) -> f32 {
  return select(
    0.0,
    detail_octave_visibility(world_footprint, detail_scale, 23.117),
    PIPELINE_PATH_TRACING_ENABLED
  );
}

fn interior_detail_fields(
  position: vec3<f32>,
  scale: f32,
  octave_weights: vec3<f32>
) -> vec4<f32> {
  let detail_position = irregular_detail_coordinates(
    position * clamp(scale, 0.001, 10000.0)
  );
  var broad = 0.0;
  var medium = 0.0;
  var fine = 0.0;
  if (octave_weights.x > 0.0001) {
    broad = (smooth_value_noise(
      detail_position * 0.347 + vec3<f32>(13.719, 3.173, 8.947)
    ) * 2.0 - 1.0) * octave_weights.x;
  }
  if (octave_weights.y > 0.0001) {
    medium = (smooth_value_noise(
      detail_position * 1.731 + vec3<f32>(2.357, 19.113, 5.731)
    ) * 2.0 - 1.0) * octave_weights.y;
  }
  if (octave_weights.z > 0.0001) {
    fine = (smooth_value_noise(
      detail_position * 6.613 + vec3<f32>(29.173, 7.419, 17.537)
    ) * 2.0 - 1.0) * octave_weights.z;
  }
  let combined = clamp(broad * 0.52 + medium * 0.31 + fine * 0.17, -1.0, 1.0);
  return vec4<f32>(broad, medium, fine, combined);
}

fn interior_path_detail_field(
  position: vec3<f32>,
  scale: f32,
  visibility: f32
) -> f32 {
  if (!PIPELINE_PATH_TRACING_ENABLED || visibility <= 0.0001) {
    return 0.0;
  }
  let detail_position = irregular_detail_coordinates(
    position * clamp(scale, 0.001, 10000.0)
  );
  return (smooth_value_noise(
    detail_position * 23.117 + vec3<f32>(41.719, 31.337, 11.173)
  ) * 2.0 - 1.0) * visibility;
}

fn nonperiodic_phase_coordinates(
  coordinates: vec2<f32>,
  fields: vec4<f32>,
  path_detail: f32
) -> vec2<f32> {
  let broad_warp = vec2<f32>(
    fields.x * 0.413 + fields.y * 0.127,
    fields.y * -0.371 + fields.x * 0.109
  );
  let fine_warp = vec2<f32>(
    fields.z * 0.071 + path_detail * 0.019,
    fields.w * -0.053 + path_detail * 0.027
  );
  return coordinates + broad_warp + fine_warp;
}

fn plaster_secondary_fine_field(
  position: vec3<f32>,
  scale: f32,
  visibility: f32
) -> f32 {
  let bounded_visibility = clamp(visibility, 0.0, 1.0);
  if (bounded_visibility <= 0.0001) {
    return 0.0;
  }
  let detail_position = irregular_detail_coordinates(
    position * clamp(scale, 0.001, 10000.0)
  );
  let decorrelated_position = vec3<f32>(
    dot(detail_position, vec3<f32>(0.173, -0.519, 0.837)),
    dot(detail_position, vec3<f32>(0.911, 0.397, 0.109)),
    dot(detail_position, vec3<f32>(-0.375, 0.757, 0.535))
  );
  return (smooth_value_noise(
    decorrelated_position * 6.613 + vec3<f32>(37.193, 13.719, 43.117)
  ) * 2.0 - 1.0) * bounded_visibility;
}

fn sparse_micro_scratch(
  coordinates: vec2<f32>,
  fields: vec4<f32>,
  path_detail: f32,
  direction: vec2<f32>,
  frequency: f32,
  sharpness: f32,
  visibility: f32
) -> f32 {
  let warped = nonperiodic_phase_coordinates(coordinates, fields, path_detail);
  let across = dot(warped, vec2<f32>(-direction.y, direction.x));
  let phase = across * frequency
    + fields.y * 1.713
    + fields.x * 0.437
    + path_detail * 0.213;
  let line = pow(
    clamp(1.0 - abs(fast_sine(phase)), 0.0, 1.0),
    clamp(sharpness, 4.0, 24.0)
  );
  let segment_field = clamp(
    fields.z * 0.61 + fields.x * 0.23 + path_detail * 0.16,
    -1.0,
    1.0
  );
  let segment_gate = smoothstep(0.39, 0.91, segment_field);
  return line * segment_gate * clamp(visibility, 0.0, 1.0);
}

fn bounded_micro_albedo(
  base_albedo: vec3<f32>,
  centered_signal: f32,
  amount: f32,
  maximum_contrast: f32
) -> vec3<f32> {
  let bounded_base = clamp(base_albedo, vec3<f32>(0.0), vec3<f32>(1.0));
  let signal = clamp(centered_signal, -1.0, 1.0);
  let contrast = clamp(maximum_contrast, 0.0, 0.2);
  let darker = bounded_base * (1.0 - contrast);
  let lighter = bounded_base + (vec3<f32>(1.0) - bounded_base) * contrast;
  var micro_target = mix(bounded_base, darker, max(-signal, 0.0));
  if (signal > 0.0) {
    micro_target = mix(bounded_base, lighter, signal);
  }
  return clamp(
    mix(bounded_base, micro_target, clamp(amount, 0.0, 1.0)),
    vec3<f32>(0.0),
    vec3<f32>(1.0)
  );
}

fn interior_micro_normal_multiplier(material: Material) -> f32 {
  if ((material.flags & MATERIAL_PLASTER) != 0u) {
    return 0.11;
  }
  if ((material.flags & MATERIAL_WOOD) != 0u) {
    return 0.14;
  }
  if ((material.flags & MATERIAL_FABRIC) != 0u) {
    return 0.16;
  }
  if ((material.flags & MATERIAL_GROUND) != 0u) {
    return 0.09;
  }
  if ((material.flags & MATERIAL_GLASS) != 0u) {
    return 0.018;
  }
  return select(0.055, 0.045, material.metallic > 0.25);
}

fn procedural_surface_sample(
  material: Material,
  position: vec3<f32>,
  normal_value: vec3<f32>,
  incoming_direction: vec3<f32>,
  world_footprint: f32,
  uv: vec2<f32>,
  tangent_value: vec3<f32>,
  bitangent_value: vec3<f32>,
  uv_gradient_magnitudes: vec2<f32>
) -> SurfaceSample {
  let base_albedo = procedural_surface_color(
    material,
    position,
    uv,
    uv_gradient_magnitudes
  );
  let base_roughness = clamp(material.roughness, 0.02, 1.0);
  let color_amount = clamp(material.color_variation, 0.0, 1.0);
  let roughness_amount = clamp(material.roughness_variation, 0.0, 1.0);
  let normal_amount = clamp(material.normal_strength, 0.0, 1.0);
  let maximum_micro_amount = max(color_amount, max(roughness_amount, normal_amount));
  if (maximum_micro_amount <= 0.000001) {
    return SurfaceSample(base_albedo, normal_value, base_roughness, 0.0);
  }
  let named_micro_surface = (material.flags & MATERIAL_INTERIOR_DETAIL_MASK) != 0u;
  let legacy_organic_surface = (material.flags & MATERIAL_LEGACY_DETAIL_MASK) != 0u;
  if ((!named_micro_surface && legacy_organic_surface)
      || (material.flags & MATERIAL_UNLIT) != 0u) {
    return SurfaceSample(base_albedo, normal_value, base_roughness, 0.0);
  }

  let stable_position = stable_microdetail_position(position, uv, uv_gradient_magnitudes);
  let uv_footprint_scale = select(
    1.0,
    max(uv_gradient_magnitudes.x, uv_gradient_magnitudes.y),
    microdetail_has_uv(uv_gradient_magnitudes)
  );
  let detail_footprint = max(world_footprint, 0.0) * uv_footprint_scale;
  let octave_weights = interior_detail_octave_weights(detail_footprint, material.detail_scale);
  let path_detail_visibility = interior_path_detail_visibility(
    detail_footprint,
    material.detail_scale
  );
  let normal_multiplier = interior_micro_normal_multiplier(material);
  if (max(
    octave_weights.x,
    max(octave_weights.y, max(octave_weights.z, path_detail_visibility))
  ) <= 0.0001) {
    let unresolved_variance = normal_amount
      * normal_amount
      * normal_multiplier
      * normal_multiplier
      * 0.5;
    return SurfaceSample(
      base_albedo,
      normal_value,
      base_roughness,
      unresolved_variance
    );
  }
  let fields = interior_detail_fields(stable_position, material.detail_scale, octave_weights);
  let path_detail = interior_path_detail_field(
    stable_position,
    material.detail_scale,
    path_detail_visibility
  );
  let normal_lod_visibility = max(
    octave_weights.y,
    max(octave_weights.z, path_detail_visibility)
  );
  var resolved_normal_visibility = normal_lod_visibility;
  var color_signal = fields.w;
  var color_contrast = 0.02;
  var roughness_signal = fields.z * 0.65 + fields.w * 0.35;
  var detail_vector = vec3<f32>(fields.y, fields.z, fields.w);

  if ((material.flags & MATERIAL_PLASTER) != 0u) {
    // The retained plaster scan already carries real trowel marks. Keep the
    // procedural layer isotropic and micro-scale: treating a scalar medium
    // noise field as one tangent slope creates false surface-axis bands under
    // grazing light, especially at low sample counts.
    let plaster_mottle = clamp(
      fields.x * 0.59 + fields.y * 0.27 + fields.z * 0.14,
      -1.0,
      1.0
    );
    let pore_source = fields.z * 0.82 + path_detail * 0.18;
    let pore_visibility = max(octave_weights.z, path_detail_visibility);
    let pore_mask = smoothstep(0.49, 0.93, pore_source) * pore_visibility;
    let centered_pores = clamp(pore_mask - 0.045 * pore_visibility, -1.0, 1.0);
    color_signal = clamp(plaster_mottle * 0.84 - centered_pores * 0.16, -1.0, 1.0);
    color_contrast = 0.055;
    roughness_signal = clamp(plaster_mottle * 0.3 + centered_pores * 0.7, -1.0, 1.0);
    let secondary_fine = plaster_secondary_fine_field(
      stable_position,
      material.detail_scale,
      octave_weights.z
    );
    detail_vector = vec3<f32>(
      fields.z,
      secondary_fine,
      0.0
    );
    resolved_normal_visibility = octave_weights.z;
  } else if ((material.flags & MATERIAL_WOOD) != 0u) {
    let coordinates = stable_microdetail_planar_coordinates(
      position, normal_value, uv, uv_gradient_magnitudes
    ) * clamp(material.detail_scale, 0.001, 10000.0);
    let phase_coordinates = nonperiodic_phase_coordinates(coordinates, fields, path_detail);
    let grain_visibility = detail_octave_visibility(
      detail_footprint,
      material.detail_scale,
      1.136
    );
    let growth_phase = phase_coordinates.y * (0.617 + fields.x * 0.031)
      + phase_coordinates.x * 0.043
      + fields.x * 0.93
      - fields.y * 0.19;
    let growth = fast_sine(growth_phase) * octave_weights.x;
    let grain_phase = phase_coordinates.x * (7.137 + fields.x * 0.311)
      + phase_coordinates.y * 0.137
      + fields.y * 2.03
      + growth * 0.61
      + path_detail * 0.09;
    let grain = fast_sine(
      grain_phase
    ) * grain_visibility;
    let vessel_source = fields.z * 0.84 + path_detail * 0.16;
    let pore_mask = smoothstep(0.54, 0.94, vessel_source)
      * max(grain_visibility, path_detail_visibility);
    color_signal = clamp(
      growth * 0.4 + grain * 0.34 + fields.w * 0.23
        + path_detail * 0.03 - pore_mask * 0.08,
      -1.0,
      1.0
    );
    color_contrast = 0.12;
    roughness_signal = clamp(
      grain * 0.43 + fields.z * 0.34 + pore_mask * 0.23,
      -1.0,
      1.0
    );
    detail_vector = vec3<f32>(
      grain,
      fields.z * 0.49 - pore_mask * 0.18 + path_detail * 0.04,
      growth * 0.31
    );
  } else if ((material.flags & MATERIAL_FABRIC) != 0u) {
    let coordinates = stable_microdetail_planar_coordinates(
      position, normal_value, uv, uv_gradient_magnitudes
    ) * clamp(material.detail_scale, 0.001, 10000.0);
    let phase_coordinates = nonperiodic_phase_coordinates(coordinates, fields, path_detail);
    let weave_visibility = detail_octave_visibility(
      detail_footprint,
      material.detail_scale,
      1.02
    );
    let warp_spacing = clamp(1.0 + fields.x * 0.031 + fields.y * 0.011, 0.95, 1.05);
    let weft_spacing = clamp(1.0 + fields.y * 0.027 - fields.x * 0.009, 0.95, 1.05);
    let warp = fast_sine(
      phase_coordinates.x * 6.2831853 * warp_spacing
        + phase_coordinates.y * 0.061
        + fields.x * 0.81
        + fields.z * 0.19
        + path_detail * 0.05
    ) * weave_visibility;
    let weft = fast_sine(
      phase_coordinates.y * 6.4076169 * weft_spacing
        - phase_coordinates.x * 0.047
        + fields.y * 0.73
        - fields.z * 0.17
        - path_detail * 0.04
    ) * weave_visibility;
    let slub = clamp(
      fields.y * 0.57 + fields.z * 0.31 + path_detail * 0.12,
      -1.0,
      1.0
    );
    let yarn_irregularity = clamp(0.91 + fields.x * 0.06 + slub * 0.03, 0.78, 1.0);
    let weave = warp * weft * yarn_irregularity;
    let loose_fiber = smoothstep(
      0.47,
      0.93,
      fields.z * 0.72 + path_detail * 0.28
    ) * max(octave_weights.z, path_detail_visibility);
    color_signal = clamp(
      weave * 0.5 + slub * 0.28 + fields.w * 0.22 - loose_fiber * 0.06,
      -1.0,
      1.0
    );
    color_contrast = 0.07;
    roughness_signal = clamp(
      (abs(warp - weft) * 0.67 - 0.23) * weave_visibility
        + fields.w * 0.2
        + loose_fiber * 0.13,
      -1.0,
      1.0
    );
    detail_vector = vec3<f32>(
      warp,
      weft,
      slub * 0.3 + loose_fiber * 0.08 + path_detail * 0.04
    );
  } else if ((material.flags & MATERIAL_GROUND) != 0u) {
    let coordinates = stable_microdetail_planar_coordinates(
      position, normal_value, uv, uv_gradient_magnitudes
    ) * clamp(material.detail_scale, 0.001, 10000.0);
    let phase_coordinates = nonperiodic_phase_coordinates(coordinates, fields, path_detail);
    let pile_visibility = detail_octave_visibility(
      detail_footprint,
      material.detail_scale,
      1.913
    );
    let pile_phase = dot(phase_coordinates, vec2<f32>(0.937, 0.349)) * 5.731
      + fields.y * 1.19
      + path_detail * 0.11;
    let pile_ridge = pow(
      clamp(0.5 + fast_sine(pile_phase) * 0.5, 0.0, 1.0),
      3.0
    ) * pile_visibility;
    let pit_source = fields.z * 0.8 + path_detail * 0.2;
    let pit_mask = smoothstep(0.56, 0.94, pit_source)
      * max(octave_weights.z, path_detail_visibility);
    color_signal = clamp(
      fields.x * 0.45 + fields.y * 0.31 + fields.w * 0.18
        + (pile_ridge - 0.25 * pile_visibility) * 0.06
        - pit_mask * 0.14,
      -1.0,
      1.0
    );
    color_contrast = 0.075;
    roughness_signal = clamp(
      fields.w * 0.43 + pit_mask * 0.39 + pile_ridge * 0.18,
      -1.0,
      1.0
    );
    detail_vector = vec3<f32>(
      fields.y + pile_ridge * 0.18,
      -pit_mask,
      fields.z - fields.x * 0.18 + path_detail * 0.06
    );
  } else if ((material.flags & MATERIAL_GLASS) != 0u) {
    let coordinates = stable_microdetail_planar_coordinates(
      position, normal_value, uv, uv_gradient_magnitudes
    ) * clamp(material.detail_scale, 0.001, 10000.0);
    let scratch_visibility = detail_octave_visibility(
      detail_footprint,
      material.detail_scale,
      2.978
    );
    let primary_scratch = sparse_micro_scratch(
      coordinates,
      fields,
      path_detail,
      vec2<f32>(0.996, 0.087),
      18.711,
      12.0,
      scratch_visibility
    );
    var cross_scratch = 0.0;
    if (PIPELINE_PATH_TRACING_ENABLED && path_detail_visibility > 0.0001) {
      cross_scratch = sparse_micro_scratch(
        coordinates,
        fields,
        path_detail,
        vec2<f32>(-0.241, 0.971),
        31.337,
        18.0,
        path_detail_visibility
      );
    }
    let scratch_line = clamp(primary_scratch + cross_scratch * 0.32, 0.0, 1.0);
    color_signal = clamp(fields.x * 0.12 - scratch_line * 0.88, -1.0, 1.0);
    color_contrast = 0.004;
    roughness_signal = clamp(fields.w * 0.18 + scratch_line * 0.82, -1.0, 1.0);
    detail_vector = vec3<f32>(scratch_line, fields.y * 0.12, fields.z * 0.08);
  } else {
    let coordinates = stable_microdetail_planar_coordinates(
      position, normal_value, uv, uv_gradient_magnitudes
    ) * clamp(material.detail_scale, 0.001, 10000.0);
    if (material.metallic > 0.25) {
      let scratch_visibility = detail_octave_visibility(
        detail_footprint,
        material.detail_scale,
        3.491
      );
      let primary_scratch = sparse_micro_scratch(
        coordinates,
        fields,
        path_detail,
        vec2<f32>(0.982, 0.188),
        21.935,
        14.0,
        scratch_visibility
      );
      var cross_scratch = 0.0;
      if (PIPELINE_PATH_TRACING_ENABLED && path_detail_visibility > 0.0001) {
        cross_scratch = sparse_micro_scratch(
          coordinates,
          fields,
          path_detail,
          vec2<f32>(-0.337, 0.942),
          37.193,
          20.0,
          path_detail_visibility
        );
      }
      let scratch_line = clamp(primary_scratch + cross_scratch * 0.38, 0.0, 1.0);
      color_signal = clamp(fields.x * 0.16 - scratch_line * 0.84, -1.0, 1.0);
      color_contrast = 0.008;
      roughness_signal = clamp(fields.w * 0.28 + scratch_line * 0.72, -1.0, 1.0);
      detail_vector = vec3<f32>(scratch_line, fields.y * 0.11, fields.z * 0.06);
    } else {
      color_signal = clamp(fields.w * 0.94 + path_detail * 0.06, -1.0, 1.0);
      color_contrast = 0.015;
      roughness_signal = clamp(
        fields.y * 0.42 + fields.w * 0.54 + path_detail * 0.04,
        -1.0,
        1.0
      );
      detail_vector = vec3<f32>(fields.y, fields.z, fields.w + path_detail * 0.05);
    }
  }

  let albedo = bounded_micro_albedo(
    base_albedo,
    color_signal,
    color_amount,
    color_contrast
  );
  let requested_roughness_delta = clamp(roughness_signal, -1.0, 1.0)
    * roughness_amount;
  let negative_roughness_limit = min(
    max(base_roughness - 0.035, 0.0),
    base_roughness * 0.45
  );
  let positive_roughness_limit = min(1.0 - base_roughness, 0.32);
  let roughness = clamp(
    base_roughness + clamp(
      requested_roughness_delta,
      -negative_roughness_limit,
      positive_roughness_limit
    ),
    0.02,
    1.0
  );
  var tangent_perturbation = detail_vector - normal_value * dot(detail_vector, normal_value);
  if (microdetail_has_uv(uv_gradient_magnitudes)) {
    tangent_perturbation = tangent_value * detail_vector.x
      + bitangent_value * detail_vector.y;
  }
  let perturbation_length_squared = dot(tangent_perturbation, tangent_perturbation);
  if (perturbation_length_squared > 1.0) {
    tangent_perturbation *= inverseSqrt(perturbation_length_squared);
  }
  var detail_normal = safe_normalize(
    normal_value
      + tangent_perturbation
        * normal_amount
        * normal_multiplier
        * resolved_normal_visibility,
    normal_value
  );
  if (dot(detail_normal, incoming_direction) >= -0.00001) {
    detail_normal = normal_value;
  }
  var resolved_normal_detail = clamp(
    octave_weights.y * 0.55
      + octave_weights.z * 0.3
      + path_detail_visibility * 0.15,
    0.0,
    1.0
  );
  if ((material.flags & MATERIAL_PLASTER) != 0u) {
    resolved_normal_detail = clamp(octave_weights.z, 0.0, 1.0);
  }
  let unresolved_normal_variance = (1.0 - resolved_normal_detail)
    * normal_amount
    * normal_amount
    * normal_multiplier
    * normal_multiplier
    * 0.5;
  return SurfaceSample(
    albedo,
    detail_normal,
    roughness,
    unresolved_normal_variance
  );
}

fn texture_luminance(color: vec3<f32>) -> f32 {
  return dot(max(color, vec3<f32>(0.0)), vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn material_texture_layer_available(layer: u32) -> bool {
  return layer != NO_TEXTURE_LAYER && layer < textureNumLayers(material_textures);
}

fn material_texture_available(material: Material) -> bool {
  return material_texture_layer_available(material.texture_layer)
    && (material.texture_flags & 7u) != 0u;
}

fn material_uses_atlas_tile(material: Material) -> bool {
  return material.atlas_offset.x > 0.000001
    || material.atlas_offset.y > 0.000001
    || material.atlas_scale.x < 0.999999
    || material.atlas_scale.y < 0.999999;
}

fn material_maximum_texture_lod(material: Material) -> f32 {
  let mip_count = max(textureNumLevels(material_textures), 1u);
  var maximum_lod = f32(mip_count - 1u);
  if (material_uses_atlas_tile(material)) {
    let base_dimensions = vec2<f32>(textureDimensions(material_textures, 0));
    let tile_dimensions = max(
      base_dimensions * clamp(material.atlas_scale, vec2<f32>(0.0001), vec2<f32>(1.0)),
      vec2<f32>(1.0)
    );
    let tile_maximum_lod = floor(log2(max(min(tile_dimensions.x, tile_dimensions.y), 1.0)))
      - ATLAS_MIP_GUARD_LEVELS;
    maximum_lod = min(maximum_lod, tile_maximum_lod);
  }
  return max(maximum_lod, 0.0);
}

fn material_texture_lod(
  material: Material,
  world_footprint: f32,
  uv_gradient_magnitudes: vec2<f32>,
  maximum_lod: f32
) -> f32 {
  var requested_lod = max(material.texture_lod, 0.0);
  if (globals.environment_texture_params.w > 0.5) {
    let base_dimensions = vec2<f32>(textureDimensions(material_textures, 0));
    let texel_gradients = max(uv_gradient_magnitudes, vec2<f32>(0.0))
      * max(material.uv_repeat, vec2<f32>(0.0001))
      * clamp(material.atlas_scale, vec2<f32>(0.0001), vec2<f32>(1.0))
      * base_dimensions;
    let footprint_squared = max(world_footprint, 0.0) * max(world_footprint, 0.0);
    let rho_squared = dot(texel_gradients, texel_gradients) * footprint_squared;
    let automatic_lod = 0.5 * log2(max(rho_squared, 1.0));
    requested_lod = max(requested_lod, automatic_lod);
  }
  return clamp(requested_lod, 0.0, maximum_lod);
}

fn material_atlas_clamp_bounds(
  material: Material,
  lod: f32,
  maximum_lod: f32
) -> vec4<f32> {
  if (!material_uses_atlas_tile(material)) {
    return vec4<f32>(0.0, 0.0, 1.0, 1.0);
  }
  let level = i32(ceil(clamp(lod, 0.0, maximum_lod)));
  let dimensions = vec2<f32>(textureDimensions(material_textures, level));
  let guard_band = ATLAS_MIP_GUARD_TEXELS / max(dimensions, vec2<f32>(1.0));
  let tile_start = clamp(material.atlas_offset, vec2<f32>(0.0), vec2<f32>(1.0));
  let tile_end = clamp(
    material.atlas_offset + material.atlas_scale,
    vec2<f32>(0.0),
    vec2<f32>(1.0)
  );
  let tile_center = (tile_start + tile_end) * 0.5;
  let tile_minimum = min(tile_start + guard_band, tile_center);
  let tile_maximum = max(tile_end - guard_band, tile_center);
  return vec4<f32>(tile_minimum, tile_maximum);
}

fn clamp_material_atlas_coordinates(
  material: Material,
  coordinates: vec2<f32>,
  clamp_bounds: vec4<f32>
) -> vec2<f32> {
  if (!material_uses_atlas_tile(material)) {
    return coordinates;
  }
  return clamp(coordinates, clamp_bounds.xy, clamp_bounds.zw);
}

fn material_texture_coordinates(
  material: Material,
  uv: vec2<f32>,
  clamp_bounds: vec4<f32>
) -> vec2<f32> {
  let repeated = fract(uv * max(material.uv_repeat, vec2<f32>(0.0001)));
  let coordinates = repeated * clamp(material.atlas_scale, vec2<f32>(0.0001), vec2<f32>(1.0))
    + clamp(material.atlas_offset, vec2<f32>(0.0), vec2<f32>(1.0));
  return clamp_material_atlas_coordinates(material, coordinates, clamp_bounds);
}

fn sample_material_texture(material: Material, atlas_uv: vec2<f32>, lod: f32) -> vec4<f32> {
  return textureSampleLevel(
    material_textures,
    material_texture_sampler,
    atlas_uv,
    i32(material.texture_layer),
    lod
  );
}

fn sample_linear_material_texture(layer: u32, atlas_uv: vec2<f32>, lod: f32) -> vec4<f32> {
  return textureSampleLevel(
    material_linear_textures,
    material_texture_sampler,
    atlas_uv,
    i32(layer),
    lod
  );
}

fn normal_variance_filtered_roughness(
  base_roughness: f32,
  decoded_mean_normal: vec3<f32>,
  normal_strength: f32
) -> f32 {
  let bounded_roughness = clamp(base_roughness, 0.02, 1.0);
  if (globals.environment_texture_params.w <= 0.5 || globals.output_mode.z <= 0.5) {
    return bounded_roughness;
  }

  // Trilinear normal-map filtering shortens the decoded mean normal whenever
  // sub-texel directions disagree. Preserve that variance as microfacet
  // roughness before normalizing the direction; otherwise distant normal maps
  // collapse into implausibly sharp, temporally unstable highlights.
  let mean_length = clamp(length(decoded_mean_normal), 0.0, 1.0);
  let strength = clamp(normal_strength, 0.0, 4.0);
  let unresolved_variance = (1.0 - mean_length) * min(strength * strength, 4.0);
  return sqrt(clamp(
    bounded_roughness * bounded_roughness + unresolved_variance,
    0.0004,
    1.0
  ));
}

fn resolve_surface_normal_variance(surface_value: SurfaceSample) -> SurfaceSample {
  var surface = surface_value;
  let unresolved_variance = clamp(surface.unresolved_normal_variance, 0.0, 1.0);
  surface.roughness = sqrt(clamp(
    surface.roughness * surface.roughness + unresolved_variance,
    0.0004,
    1.0
  ));
  surface.unresolved_normal_variance = 0.0;
  return surface;
}

fn material_surface_sample(
  material: Material,
  position: vec3<f32>,
  normal_value: vec3<f32>,
  incoming_direction: vec3<f32>,
  uv: vec2<f32>,
  tangent_value: vec3<f32>,
  bitangent_value: vec3<f32>,
  texture_world_footprint: f32,
  uv_gradient_magnitudes: vec2<f32>
) -> SurfaceSample {
  var surface = procedural_surface_sample(
    material,
    position,
    normal_value,
    incoming_direction,
    texture_world_footprint,
    uv,
    tangent_value,
    bitangent_value,
    uv_gradient_magnitudes
  );
  let base_texture_available = material_texture_available(material);
  let normal_mapping_enabled = material.texture_normal_strength > 0.000001;
  let roughness_mapping_enabled = material.texture_roughness_strength > 0.000001;
  let independent_normal_available = (material.texture_flags & MATERIAL_TEXTURE_INDEPENDENT_NORMAL) != 0u
    && material_texture_layer_available(material.pbr_texture_layers.x)
    && normal_mapping_enabled;
  let independent_roughness_available = (material.texture_flags & MATERIAL_TEXTURE_INDEPENDENT_ROUGHNESS) != 0u
    && material_texture_layer_available(material.pbr_texture_layers.y)
    && roughness_mapping_enabled;
  if (!base_texture_available && !independent_normal_available && !independent_roughness_available) {
    return resolve_surface_normal_variance(surface);
  }

  let maximum_lod = material_maximum_texture_lod(material);
  let lod = material_texture_lod(
    material,
    texture_world_footprint,
    uv_gradient_magnitudes,
    maximum_lod
  );
  let atlas_clamp_bounds = material_atlas_clamp_bounds(material, lod, maximum_lod);
  let atlas_uv = material_texture_coordinates(material, uv, atlas_clamp_bounds);
  var texture_value = vec4<f32>(1.0);
  if (base_texture_available) {
    texture_value = sample_material_texture(material, atlas_uv, lod);
    if ((material.texture_flags & MATERIAL_TEXTURE_BASE_COLOR) != 0u) {
      let textured_albedo = surface.albedo * max(texture_value.rgb, vec3<f32>(0.0));
      surface.albedo = mix(surface.albedo, textured_albedo, clamp(material.texture_strength, 0.0, 1.0));
    }
  }
  if (independent_roughness_available) {
    let mapped_roughness = sample_linear_material_texture(
      material.pbr_texture_layers.y,
      atlas_uv,
      lod
    ).r;
    surface.roughness = clamp(mix(
      surface.roughness,
      clamp(mapped_roughness, 0.02, 1.0),
      clamp(material.texture_roughness_strength, 0.0, 1.0)
    ), 0.02, 1.0);
  } else if (
    base_texture_available
      && roughness_mapping_enabled
      && (material.texture_flags & MATERIAL_TEXTURE_LUMINANCE_ROUGHNESS) != 0u
  ) {
    surface.roughness = clamp(mix(
      surface.roughness,
      clamp(texture_luminance(texture_value.rgb), 0.02, 1.0),
      clamp(material.texture_roughness_strength, 0.0, 1.0)
    ), 0.02, 1.0);
  }
  if (independent_normal_available) {
    let encoded_normal = sample_linear_material_texture(
      material.pbr_texture_layers.x,
      atlas_uv,
      lod
    ).rgb;
    let tangent_normal = encoded_normal * 2.0 - vec3<f32>(1.0);
    let normal_strength = clamp(material.texture_normal_strength, 0.0, 4.0);
    surface.roughness = normal_variance_filtered_roughness(
      surface.roughness,
      tangent_normal,
      normal_strength
    );
    let mapped_tangent = safe_normalize(
      tangent_value - surface.normal * dot(tangent_value, surface.normal),
      tangent_value
    );
    let mapped_handedness = select(
      -1.0,
      1.0,
      dot(cross(surface.normal, mapped_tangent), bitangent_value) >= 0.0
    );
    let mapped_bitangent = safe_normalize(
      cross(surface.normal, mapped_tangent),
      bitangent_value
    ) * mapped_handedness;
    let textured_normal = safe_normalize(
      mapped_tangent * tangent_normal.x * normal_strength
        + mapped_bitangent * tangent_normal.y * normal_strength
        + surface.normal * max(tangent_normal.z, 0.0001),
      surface.normal
    );
    if (dot(textured_normal, incoming_direction) < -0.00001) {
      surface.normal = textured_normal;
    }
  } else if (
    base_texture_available
      && normal_mapping_enabled
      && (material.texture_flags & MATERIAL_TEXTURE_LUMINANCE_NORMAL) != 0u
  ) {
    let dimensions = vec2<f32>(textureDimensions(
      material_textures,
      i32(ceil(clamp(lod, 0.0, maximum_lod)))
    ));
    let texel = vec2<f32>(1.0) / max(dimensions, vec2<f32>(1.0));
    let left = texture_luminance(sample_material_texture(
      material,
      clamp_material_atlas_coordinates(
        material,
        atlas_uv - vec2<f32>(texel.x, 0.0),
        atlas_clamp_bounds
      ),
      lod
    ).rgb);
    let right = texture_luminance(sample_material_texture(
      material,
      clamp_material_atlas_coordinates(
        material,
        atlas_uv + vec2<f32>(texel.x, 0.0),
        atlas_clamp_bounds
      ),
      lod
    ).rgb);
    let down = texture_luminance(sample_material_texture(
      material,
      clamp_material_atlas_coordinates(
        material,
        atlas_uv - vec2<f32>(0.0, texel.y),
        atlas_clamp_bounds
      ),
      lod
    ).rgb);
    let up = texture_luminance(sample_material_texture(
      material,
      clamp_material_atlas_coordinates(
        material,
        atlas_uv + vec2<f32>(0.0, texel.y),
        atlas_clamp_bounds
      ),
      lod
    ).rgb);
    let helper = select(
      vec3<f32>(0.0, 1.0, 0.0),
      vec3<f32>(1.0, 0.0, 0.0),
      abs(surface.normal.y) > 0.98
    );
    let tangent = safe_normalize(cross(helper, surface.normal), vec3<f32>(1.0, 0.0, 0.0));
    let bitangent = safe_normalize(cross(surface.normal, tangent), vec3<f32>(0.0, 0.0, 1.0));
    let gradient = vec2<f32>(right - left, up - down);
    let normal_strength = clamp(material.texture_normal_strength, 0.0, 4.0);
    let textured_normal = safe_normalize(
      surface.normal - tangent * gradient.x * normal_strength - bitangent * gradient.y * normal_strength,
      surface.normal
    );
    if (dot(textured_normal, incoming_direction) < -0.00001) {
      surface.normal = textured_normal;
    }
  }
  return resolve_surface_normal_variance(surface);
}

fn wind_waveform(position: vec3<f32>) -> vec3<f32> {
  var horizontal_direction = globals.history_params.yz;
  let horizontal_length_squared = dot(horizontal_direction, horizontal_direction);
  if (horizontal_length_squared <= 0.00000001) {
    horizontal_direction = vec2<f32>(1.0, 0.0);
  } else {
    horizontal_direction *= inverseSqrt(horizontal_length_squared);
  }

  let turbulence = clamp(globals.history_params.w, 0.0, 2.0);
  let time = globals.environment_params.y;
  let speed = max(globals.environment_params.w, 0.0);
  let perpendicular = vec2<f32>(-horizontal_direction.y, horizontal_direction.x);
  let along_wind = dot(position.xz, horizontal_direction);
  let across_wind = dot(position.xz, perpendicular);
  let phase = along_wind * 0.31 + time * speed + fast_sine(across_wind * 0.19 + time * 0.37) * turbulence;
  let wave_scalar = clamp(
    fast_sine(phase) * 0.62
      + fast_sine(phase * 1.73 + across_wind * 0.11 + 2.1) * 0.25
      + fast_sine(phase * 0.43 + position.y * 0.27 + 4.3) * 0.13,
    -1.0,
    1.0
  );
  let vertical_motion = fast_sine(phase * 1.31 + across_wind * 0.23) * turbulence * 0.12;
  let wave_direction = safe_normalize(
    vec3<f32>(horizontal_direction.x, vertical_motion, horizontal_direction.y),
    vec3<f32>(horizontal_direction.x, 0.0, horizontal_direction.y)
  );
  return wave_direction * wave_scalar;
}

fn wind_displacement(position: vec3<f32>, weight: f32, amplitude: f32) -> vec3<f32> {
  let strength = clamp(globals.environment_params.z / MAX_WIND_DISPLACEMENT, 0.0, 1.0);
  let safe_amplitude = max(amplitude, 0.0);
  let safe_weight = clamp(weight, 0.0, 1.0);
  if (safe_amplitude <= 0.0 || strength <= 0.0 || safe_weight <= 0.0) {
    return vec3<f32>(0.0);
  }
  return wind_waveform(position) * safe_amplitude * strength * safe_weight;
}

fn load_precomputed_animated_geometry(triangle: Triangle) -> AnimatedTriangleGeometry {
  let fallback = AnimatedTriangleGeometry(triangle.v0, triangle.edge1, triangle.edge2, 0u);
  let encoded_slot = (triangle.shadow_flags & TRIANGLE_ANIMATED_SHADOW_SLOT_MASK)
    >> TRIANGLE_ANIMATED_SHADOW_SLOT_SHIFT;
  if (encoded_slot == 0u || !animated_shadow_bvh_is_valid()) {
    return fallback;
  }
  let slot = encoded_slot - 1u;
  let index_offset = animated_shadow_bvh[2];
  let index_count = animated_shadow_bvh[3];
  if (slot >= index_count) {
    return fallback;
  }
  let triangle_index = animated_shadow_bvh[index_offset + slot];
  let geometry_offset = index_offset + index_count;
  let word = geometry_offset + slot * ANIMATED_SHADOW_TRIANGLE_WORDS;
  if (animated_shadow_bvh[word + 3u] != (triangle_index ^ ANIMATED_SHADOW_TRIANGLE_MARKER)) {
    return fallback;
  }
  return AnimatedTriangleGeometry(
    vec3<f32>(
      bitcast<f32>(animated_shadow_bvh[word]),
      bitcast<f32>(animated_shadow_bvh[word + 1u]),
      bitcast<f32>(animated_shadow_bvh[word + 2u])
    ),
    vec3<f32>(
      bitcast<f32>(animated_shadow_bvh[word + 4u]),
      bitcast<f32>(animated_shadow_bvh[word + 5u]),
      bitcast<f32>(animated_shadow_bvh[word + 6u])
    ),
    vec3<f32>(
      bitcast<f32>(animated_shadow_bvh[word + 8u]),
      bitcast<f32>(animated_shadow_bvh[word + 9u]),
      bitcast<f32>(animated_shadow_bvh[word + 10u])
    ),
    1u
  );
}

fn animated_triangle_geometry(triangle: Triangle) -> AnimatedTriangleGeometry {
  var geometry = AnimatedTriangleGeometry(triangle.v0, triangle.edge1, triangle.edge2, 0u);
  let amplitude = max(triangle.wind_amplitude, 0.0);
  if (amplitude > 0.0 && globals.environment_params.z > 0.0) {
    let precomputed = load_precomputed_animated_geometry(triangle);
    if (precomputed.cached != 0u) {
      return precomputed;
    }
    let base_vertex_0 = triangle.v0;
    let base_vertex_1 = triangle.v0 + triangle.edge1;
    let base_vertex_2 = triangle.v0 + triangle.edge2;
    let animated_vertex_0 = base_vertex_0
      + wind_displacement(base_vertex_0, triangle.wind_weight_0, amplitude);
    let animated_vertex_1 = base_vertex_1
      + wind_displacement(base_vertex_1, triangle.wind_weight_1, amplitude);
    let animated_vertex_2 = base_vertex_2
      + wind_displacement(base_vertex_2, triangle.wind_weight_2, amplitude);
    geometry.vertex_0 = animated_vertex_0;
    geometry.edge_1 = animated_vertex_1 - animated_vertex_0;
    geometry.edge_2 = animated_vertex_2 - animated_vertex_0;
  }
  return geometry;
}

fn miss() -> Hit {
  return Hit(
    1e30,
    vec3<f32>(0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0),
    1.0,
    vec3<f32>(0.0),
    0.0,
    0.0,
    1.5,
    NO_MATERIAL_INDEX,
    0u,
    0u,
    0u,
    0.0
  );
}

fn geometry_miss(maximum_distance: f32) -> TriangleGeometryHit {
  return TriangleGeometryHit(maximum_distance, 0.0, 0.0, 0u, 0u, 0u);
}

fn shadow_miss(maximum_distance: f32) -> ShadowHit {
  return ShadowHit(maximum_distance, 0.0, 0u, 0u, vec3<f32>(1.0), 0.0);
}

fn material_hit(
  distance: f32,
  position: vec3<f32>,
  normal_value: vec3<f32>,
  geometric_normal_value: vec3<f32>,
  material_id: u32,
  front_face: u32,
  reactive_value: f32,
  incoming_direction: vec3<f32>,
  uv: vec2<f32>,
  tangent_value: vec3<f32>,
  bitangent_value: vec3<f32>,
  texture_world_footprint: f32,
  uv_gradient_magnitudes: vec2<f32>
) -> Hit {
  let material_count = min(u32(globals.scene_counts.w), arrayLength(&materials));
  if (material_id >= material_count) {
    return Hit(
      distance,
      position,
      normal_value,
      geometric_normal_value,
      vec3<f32>(0.8),
      0.5,
      vec3<f32>(0.0),
      0.0,
      0.0,
      1.5,
      material_id,
      0u,
      1u,
      front_face,
      reactive_value
    );
  }
  let material = materials[material_id];
  let surface = material_surface_sample(
    material,
    position,
    normal_value,
    incoming_direction,
    uv,
    tangent_value,
    bitangent_value,
    texture_world_footprint,
    uv_gradient_magnitudes
  );
  let reactive = select(
    reactive_value,
    1.0,
    material.wind_influence > 0.0001 && globals.environment_params.z > 0.0001
  );
  let unlit = (material.flags & MATERIAL_UNLIT) != 0u;
  let emission = select(
    sanitize_radiance(material.emission_color * material.emission_strength),
    sanitize_radiance(surface.albedo * material.emission_color * material.emission_strength),
    unlit
  );
  let geometric_normal = safe_normalize(geometric_normal_value, normal_value);
  var shading_normal = safe_normalize(surface.normal, normal_value);
  if (dot(shading_normal, geometric_normal) < 0.0) {
    shading_normal = -shading_normal;
  }
  let geometric_alignment = dot(shading_normal, geometric_normal);
  if (geometric_alignment < 0.025) {
    let tangent_component = shading_normal - geometric_normal * geometric_alignment;
    shading_normal = safe_normalize(
      tangent_component + geometric_normal * 0.025,
      geometric_normal
    );
  }
  if (dot(shading_normal, incoming_direction) >= -0.00001) {
    shading_normal = geometric_normal;
  }
  return Hit(
    distance,
    position,
    shading_normal,
    geometric_normal,
    surface.albedo,
    surface.roughness,
    emission,
    clamp(material.metallic, 0.0, 1.0),
    max(clamp(material.transmission, 0.0, 1.0), 1.0 - clamp(material.alpha, 0.0, 1.0)),
    clamp(material.ior, 1.0, 3.0),
    material_id,
    material.flags,
    1u,
    front_face,
    reactive
  );
}

fn hit_has_material_flag(hit: Hit, flag: u32) -> bool {
  return (hit.material_flags & flag) != 0u;
}

fn hit_is_unlit(hit: Hit) -> bool {
  return hit_has_material_flag(hit, MATERIAL_UNLIT);
}

fn hit_is_fabric(hit: Hit) -> bool {
  return hit_has_material_flag(hit, MATERIAL_FABRIC);
}

fn hit_is_foliage(hit: Hit) -> bool {
  return hit_has_material_flag(hit, MATERIAL_FOLIAGE);
}

fn hit_is_solid_glass(hit: Hit) -> bool {
  return hit_has_material_flag(hit, MATERIAL_SOLID_GLASS);
}

fn hit_is_thin_glass(hit: Hit) -> bool {
  return hit_has_material_flag(hit, MATERIAL_GLASS)
    && !hit_is_solid_glass(hit)
    && hit.transmission >= BACKPLATE_TRANSMISSION_MINIMUM
    && hit.metallic < 0.999999
    && hit.roughness <= BACKPLATE_TRANSMISSION_ROUGHNESS_MAX;
}

fn offset_hit_position(hit: Hit, outgoing_direction: vec3<f32>) -> vec3<f32> {
  let side = select(-1.0, 1.0, dot(outgoing_direction, hit.geometric_normal) >= 0.0);
  return hit.position + hit.geometric_normal * (RAY_ORIGIN_BIAS * side);
}

fn valid_geometric_reflection(
  hit: Hit,
  outgoing_direction: vec3<f32>,
  incident_direction: vec3<f32>
) -> bool {
  return dot(outgoing_direction, hit.geometric_normal)
    * dot(incident_direction, hit.geometric_normal) > 0.0;
}

fn valid_geometric_transmission(
  hit: Hit,
  outgoing_direction: vec3<f32>,
  incident_direction: vec3<f32>
) -> bool {
  return dot(outgoing_direction, hit.geometric_normal)
    * dot(incident_direction, hit.geometric_normal) < 0.0;
}

fn material_accepts_backface(material_id: u32) -> bool {
  let material_count = min(u32(globals.scene_counts.w), arrayLength(&materials));
  if (material_id >= material_count) {
    return false;
  }
  let material = materials[material_id];
  let effective_transmission = max(
    clamp(material.transmission, 0.0, 1.0),
    1.0 - clamp(material.alpha, 0.0, 1.0)
  );
  return (material.flags & MATERIAL_DOUBLE_SIDED) != 0u
    || effective_transmission > 0.001;
}

fn triangle_accepts_backface(material_id: u32, shadow_flags: u32) -> bool {
  if ((shadow_flags & TRIANGLE_SHADOW_CACHE_VALID) != 0u) {
    return (shadow_flags & TRIANGLE_SHADOW_ACCEPTS_BACKFACE) != 0u;
  }
  return material_accepts_backface(material_id);
}

fn intersect_sphere_distance(ray: Ray, sphere: Sphere, closest: f32) -> f32 {
  let center = sphere.center_radius.xyz;
  let radius = sphere.center_radius.w;
  if (radius <= INTERSECTION_EPSILON) {
    return 1e30;
  }
  let offset = ray.origin - center;
  let half_b = dot(offset, ray.direction);
  let c = dot(offset, offset) - radius * radius;
  let discriminant = half_b * half_b - c;
  if (discriminant < 0.0) {
    return 1e30;
  }

  let root = sqrt(discriminant);
  var distance = -half_b - root;
  if (distance <= INTERSECTION_EPSILON) {
    distance = -half_b + root;
  }
  if (distance <= INTERSECTION_EPSILON || distance >= closest) {
    return 1e30;
  }
  return distance;
}

fn intersect_sphere(ray: Ray, sphere: Sphere, closest: f32) -> Hit {
  let distance = intersect_sphere_distance(ray, sphere, closest);
  if (distance >= closest) {
    return miss();
  }

  let center = sphere.center_radius.xyz;
  let position = ray.origin + ray.direction * distance;
  var normal_value = normalize(position - center);
  let front_face = dot(normal_value, ray.direction) < 0.0;
  if (!front_face) {
    normal_value = -normal_value;
  }
  return Hit(
    distance,
    position,
    normal_value,
    normal_value,
    max(sphere.color_roughness.xyz, vec3<f32>(0.0)),
    clamp(sphere.color_roughness.w, 0.02, 1.0),
    sanitize_radiance(sphere.color_roughness.xyz * sphere.material.y),
    clamp(sphere.material.x, 0.0, 1.0),
    0.0,
    1.5,
    NO_MATERIAL_INDEX,
    0u,
    1u,
    select(0u, 1u, front_face),
    0.0
  );
}

fn intersect_triangle_geometry(
  ray: Ray,
  triangle: Triangle,
  triangle_index: u32,
  closest: f32
) -> TriangleGeometryHit {
  let geometry = animated_triangle_geometry(triangle);
  let vertex_0 = geometry.vertex_0;
  let edge_1 = geometry.edge_1;
  let edge_2 = geometry.edge_2;

  let p = cross(ray.direction, edge_2);
  let determinant = dot(edge_1, p);
  if (abs(determinant) < 0.0000001) {
    return geometry_miss(closest);
  }
  if (determinant < 0.0 && !triangle_accepts_backface(triangle.material_id, triangle.shadow_flags)) {
    return geometry_miss(closest);
  }
  let inverse_determinant = 1.0 / determinant;
  let offset = ray.origin - vertex_0;
  let u = dot(offset, p) * inverse_determinant;
  if (u < 0.0 || u > 1.0) {
    return geometry_miss(closest);
  }
  let q = cross(offset, edge_1);
  let v = dot(ray.direction, q) * inverse_determinant;
  if (v < 0.0 || u + v > 1.0) {
    return geometry_miss(closest);
  }
  let distance = dot(edge_2, q) * inverse_determinant;
  if (distance <= INTERSECTION_EPSILON || distance >= closest) {
    return geometry_miss(closest);
  }

  return TriangleGeometryHit(
    distance,
    u,
    v,
    triangle_index,
    select(0u, 1u, determinant > 0.0),
    1u
  );
}

fn intersect_shadow_triangle_distance(
  ray: Ray,
  triangle: Triangle,
  closest: f32
) -> f32 {
  let geometry = animated_triangle_geometry(triangle);
  let vertex_0 = geometry.vertex_0;
  let edge_1 = geometry.edge_1;
  let edge_2 = geometry.edge_2;

  let p = cross(ray.direction, edge_2);
  let determinant = dot(edge_1, p);
  if (abs(determinant) < 0.0000001) {
    return closest;
  }
  if (determinant < 0.0 && !triangle_accepts_backface(triangle.material_id, triangle.shadow_flags)) {
    return closest;
  }
  let inverse_determinant = 1.0 / determinant;
  let offset = ray.origin - vertex_0;
  let u = dot(offset, p) * inverse_determinant;
  if (u < 0.0 || u > 1.0) {
    return closest;
  }
  let q = cross(offset, edge_1);
  let v = dot(ray.direction, q) * inverse_determinant;
  if (v < 0.0 || u + v > 1.0) {
    return closest;
  }
  let distance = dot(edge_2, q) * inverse_determinant;
  return select(closest, distance, distance > INTERSECTION_EPSILON && distance < closest);
}

fn materialize_triangle_hit(
  ray: Ray,
  triangle: Triangle,
  geometry: TriangleGeometryHit
) -> Hit {
  let animated_geometry = animated_triangle_geometry(triangle);
  let vertex_0 = animated_geometry.vertex_0;
  let edge_1 = animated_geometry.edge_1;
  let edge_2 = animated_geometry.edge_2;
  let u = geometry.u;
  let v = geometry.v;

  let raw_geometric_normal = safe_normalize(cross(edge_1, edge_2), vec3<f32>(0.0, 1.0, 0.0));
  var geometric_normal = raw_geometric_normal;
  if (dot(geometric_normal, ray.direction) > 0.0) {
    geometric_normal = -geometric_normal;
  }
  var normal_value = safe_normalize(
    triangle.normal_0 * (1.0 - u - v) + triangle.normal_1 * u + triangle.normal_2 * v,
    raw_geometric_normal
  );
  if (dot(normal_value, raw_geometric_normal) < 0.0) {
    normal_value = -normal_value;
  }
  if (dot(normal_value, geometric_normal) < 0.0) {
    normal_value = -normal_value;
  }
  if (dot(normal_value, ray.direction) >= -0.00001) {
    normal_value = geometric_normal;
  }
  let material_id = triangle.material_id;
  let uv = triangle.uv_0_1.xy * (1.0 - u - v)
    + triangle.uv_0_1.zw * u
    + triangle.uv_2_gradient_magnitudes.xy * v;
  let uv_edge_1 = triangle.uv_0_1.zw - triangle.uv_0_1.xy;
  let uv_edge_2 = triangle.uv_2_gradient_magnitudes.xy - triangle.uv_0_1.xy;
  let uv_determinant = uv_edge_1.x * uv_edge_2.y - uv_edge_1.y * uv_edge_2.x;
  let fallback_helper = select(
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0),
    abs(normal_value.y) > 0.98
  );
  var tangent_value = safe_normalize(cross(fallback_helper, normal_value), vec3<f32>(1.0, 0.0, 0.0));
  var bitangent_value = safe_normalize(cross(normal_value, tangent_value), vec3<f32>(0.0, 0.0, 1.0));
  if (abs(uv_determinant) > 0.0000001) {
    let inverse_uv_determinant = 1.0 / uv_determinant;
    let raw_tangent = (edge_1 * uv_edge_2.y - edge_2 * uv_edge_1.y) * inverse_uv_determinant;
    let raw_bitangent = (edge_2 * uv_edge_1.x - edge_1 * uv_edge_2.x) * inverse_uv_determinant;
    tangent_value = safe_normalize(
      raw_tangent - normal_value * dot(raw_tangent, normal_value),
      tangent_value
    );
    let handedness = select(-1.0, 1.0, dot(cross(normal_value, tangent_value), raw_bitangent) >= 0.0);
    bitangent_value = safe_normalize(cross(normal_value, tangent_value), bitangent_value) * handedness;
  }
  let reactive = select(
    0.0,
    1.0,
    max(triangle.wind_amplitude, 0.0) > 0.0 && globals.environment_params.z > 0.0
  );
  let cone_width = ray_cone_width_at_distance(ray, geometry.t);
  let grazing_cosine = max(abs(dot(ray.direction, raw_geometric_normal)), 0.125);
  let texture_world_footprint = cone_width / grazing_cosine;
  return material_hit(
    geometry.t,
    ray.origin + ray.direction * geometry.t,
    normal_value,
    geometric_normal,
    material_id,
    geometry.front_face,
    reactive,
    ray.direction,
    uv,
    tangent_value,
    bitangent_value,
    texture_world_footprint,
    triangle.uv_2_gradient_magnitudes.zw
  );
}

fn intersect_bounds(
  ray: Ray,
  inverse_direction: vec3<f32>,
  node: BvhNode,
  closest: f32
) -> f32 {
  let first = (node.bounds_min - ray.origin) * inverse_direction;
  let second = (node.bounds_max - ray.origin) * inverse_direction;
  let minimum = min(first, second);
  let maximum = max(first, second);
  let near_distance = max(max(minimum.x, minimum.y), max(minimum.z, 0.0));
  let far_distance = min(min(maximum.x, maximum.y), maximum.z);
  return select(1e30, near_distance, far_distance >= near_distance && near_distance < closest);
}

fn intersect_bvh_geometry(
  ray: Ray,
  maximum_distance: f32,
  local_invocation_index: u32
) -> TriangleGeometryHit {
  let node_count = min(u32(globals.scene_counts.y), arrayLength(&bvh_nodes));
  let triangle_count = min(u32(globals.scene_counts.z), arrayLength(&triangles));
  if (node_count == 0u || triangle_count == 0u) {
    return geometry_miss(maximum_distance);
  }

  var closest_hit = geometry_miss(maximum_distance);
  let near_zero = abs(ray.direction) < vec3<f32>(0.0000001);
  let safe_direction = select(ray.direction, vec3<f32>(0.0000001), near_zero);
  let inverse_direction = vec3<f32>(1.0) / safe_direction;
  let stack_base = local_invocation_index * BVH_STACK_CAPACITY;
  var stack_size = 1u;
  bvh_traversal_stack[stack_base] = 0u;

  // Valid trees have one parent per node, so no traversal can pop more nodes.
  for (var visited_nodes = 0u; visited_nodes < node_count; visited_nodes += 1u) {
    if (stack_size == 0u) {
      break;
    }
    stack_size -= 1u;
    let stack_entry = bvh_traversal_stack[stack_base + stack_size];
    let node_index = stack_entry & BVH_STACK_NODE_INDEX_MASK;
    let bounds_accepted = (stack_entry & BVH_STACK_BOUNDS_ACCEPTED_FLAG) != 0u;
    if (node_index >= node_count) {
      continue;
    }
    let node = bvh_nodes[node_index];
    if (!bounds_accepted
        && intersect_bounds(ray, inverse_direction, node, closest_hit.t) >= closest_hit.t) {
      continue;
    }

    let left_first = node.left_first;
    let primitive_count = node.count & BVH_NODE_PRIMITIVE_COUNT_MASK;
    if (primitive_count > 0u) {
      for (var primitive = 0u; primitive < 4u; primitive += 1u) {
        if (primitive >= primitive_count || left_first >= triangle_count
            || primitive >= triangle_count - left_first) {
          break;
        }
        let triangle_index = left_first + primitive;
        let candidate = intersect_triangle_geometry(
          ray,
          triangles[triangle_index],
          triangle_index,
          closest_hit.t
        );
        if (candidate.hit != 0u) {
          closest_hit = candidate;
        }
      }
    } else if (left_first > node_index && left_first < node_count - 1u
        && stack_size <= BVH_STACK_CAPACITY - 2u) {
      let left_distance = intersect_bounds(
        ray,
        inverse_direction,
        bvh_nodes[left_first],
        closest_hit.t
      );
      let right_distance = intersect_bounds(
        ray,
        inverse_direction,
        bvh_nodes[left_first + 1u],
        closest_hit.t
      );
      let left_accepted = left_distance < closest_hit.t;
      let right_accepted = right_distance < closest_hit.t;
      if (left_accepted && right_accepted) {
        if (left_distance < right_distance) {
          bvh_traversal_stack[stack_base + stack_size] = left_first + 1u;
          stack_size += 1u;
          bvh_traversal_stack[stack_base + stack_size] = left_first
            | BVH_STACK_BOUNDS_ACCEPTED_FLAG;
          stack_size += 1u;
        } else {
          bvh_traversal_stack[stack_base + stack_size] = left_first;
          stack_size += 1u;
          bvh_traversal_stack[stack_base + stack_size] = (left_first + 1u)
            | BVH_STACK_BOUNDS_ACCEPTED_FLAG;
          stack_size += 1u;
        }
      } else if (left_accepted) {
        bvh_traversal_stack[stack_base + stack_size] = left_first
          | BVH_STACK_BOUNDS_ACCEPTED_FLAG;
        stack_size += 1u;
      } else if (right_accepted) {
        bvh_traversal_stack[stack_base + stack_size] = (left_first + 1u)
          | BVH_STACK_BOUNDS_ACCEPTED_FLAG;
        stack_size += 1u;
      }
    }
  }

  return closest_hit;
}

fn intersect_scene(ray: Ray, local_invocation_index: u32) -> Hit {
  var closest_hit = miss();
  let sphere_count = min(min(u32(globals.scene_counts.x), arrayLength(&spheres)), MAX_SPHERES);
  let triangle_count = min(u32(globals.scene_counts.z), arrayLength(&triangles));
  for (var sphere_index = 0u; sphere_index < sphere_count; sphere_index += 1u) {
    let candidate = intersect_sphere(ray, spheres[sphere_index], closest_hit.t);
    if (candidate.hit != 0u) {
      closest_hit = candidate;
    }
  }

  let triangle_hit = intersect_bvh_geometry(ray, closest_hit.t, local_invocation_index);
  if (triangle_hit.hit != 0u) {
    closest_hit = materialize_triangle_hit(
      ray,
      triangles[triangle_hit.triangle_index],
      triangle_hit
    );
  }

  if (triangle_count == 0u && ray.direction.y < -0.00001) {
    let plane_distance = -ray.origin.y / ray.direction.y;
    if (plane_distance > INTERSECTION_EPSILON && plane_distance < closest_hit.t) {
      let position = ray.origin + ray.direction * plane_distance;
      let checker_index = (i32(floor(position.x)) + i32(floor(position.z))) & 1;
      let checker = select(0.32, 0.72, checker_index == 0);
      closest_hit = Hit(
        plane_distance,
        position,
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(checker, checker * 0.96, checker * 0.88),
        0.88,
        vec3<f32>(0.0),
        0.0,
        0.0,
        1.5,
        NO_MATERIAL_INDEX,
        0u,
        1u,
        1u,
        0.0
      );
    }
  }

  return closest_hit;
}

fn shadow_hit_from_triangle(distance: f32, triangle: Triangle) -> ShadowHit {
  let material_count = min(u32(globals.scene_counts.w), arrayLength(&materials));
  if (triangle.material_id >= material_count) {
    return ShadowHit(distance, 0.0, 1u, 0u, vec3<f32>(0.8), 0.0);
  }
  let material = materials[triangle.material_id];
  if ((material.flags & MATERIAL_UNLIT) != 0u) {
    return ShadowHit(distance, 1.0, 1u, 0u, vec3<f32>(1.0), 0.0);
  }
  let transmission = max(
    clamp(material.transmission, 0.0, 1.0),
    1.0 - clamp(material.alpha, 0.0, 1.0)
  );
  return ShadowHit(
    distance,
    transmission,
    1u,
    0u,
    max(material.base_color, vec3<f32>(0.0)),
    0.0
  );
}

fn triangle_is_unlit(triangle: Triangle) -> bool {
  let material_count = min(u32(globals.scene_counts.w), arrayLength(&materials));
  return triangle.material_id < material_count
    && (materials[triangle.material_id].flags & MATERIAL_UNLIT) != 0u;
}

fn triangle_is_wind_animated(triangle: Triangle) -> bool {
  return triangle.wind_amplitude > 0.0;
}

fn shadow_filter_accepts_triangle(triangle: Triangle, shadow_filter: u32) -> bool {
  let animated = triangle_is_wind_animated(triangle);
  return shadow_filter == SHADOW_FILTER_ALL
    || (shadow_filter == SHADOW_FILTER_STATIC && !animated)
    || (shadow_filter == SHADOW_FILTER_ANIMATED && animated);
}

fn shadow_filter_accepts_node(node: BvhNode, shadow_filter: u32) -> bool {
  if (shadow_filter == SHADOW_FILTER_ALL) {
    return true;
  }
  let content = node.count & (
    BVH_NODE_HAS_STATIC_SHADOW_TRIANGLES | BVH_NODE_HAS_ANIMATED_SHADOW_TRIANGLES
  );
  if (content == 0u) {
    return true;
  }
  let required = select(
    BVH_NODE_HAS_ANIMATED_SHADOW_TRIANGLES,
    BVH_NODE_HAS_STATIC_SHADOW_TRIANGLES,
    shadow_filter == SHADOW_FILTER_STATIC
  );
  return (content & required) != 0u;
}

fn animated_shadow_bvh_is_valid() -> bool {
  let word_count = arrayLength(&animated_shadow_bvh);
  if (word_count < ANIMATED_SHADOW_BVH_HEADER_WORDS
      || animated_shadow_bvh[0] != ANIMATED_SHADOW_BVH_MAGIC) {
    return false;
  }
  let node_count = animated_shadow_bvh[1];
  let index_offset = animated_shadow_bvh[2];
  let index_count = animated_shadow_bvh[3];
  if (node_count > (word_count - ANIMATED_SHADOW_BVH_HEADER_WORDS) / ANIMATED_SHADOW_BVH_NODE_WORDS
      || index_offset != ANIMATED_SHADOW_BVH_HEADER_WORDS
        + node_count * ANIMATED_SHADOW_BVH_NODE_WORDS
      || index_offset > word_count
      || index_count > word_count - index_offset) {
    return false;
  }
  let geometry_offset = index_offset + index_count;
  return geometry_offset <= word_count
    && index_count <= (word_count - geometry_offset) / ANIMATED_SHADOW_TRIANGLE_WORDS;
}

fn load_animated_shadow_bvh_node(node_index: u32) -> BvhNode {
  let word = ANIMATED_SHADOW_BVH_HEADER_WORDS
    + node_index * ANIMATED_SHADOW_BVH_NODE_WORDS;
  return BvhNode(
    vec3<f32>(
      bitcast<f32>(animated_shadow_bvh[word]),
      bitcast<f32>(animated_shadow_bvh[word + 1u]),
      bitcast<f32>(animated_shadow_bvh[word + 2u])
    ),
    animated_shadow_bvh[word + 3u],
    vec3<f32>(
      bitcast<f32>(animated_shadow_bvh[word + 4u]),
      bitcast<f32>(animated_shadow_bvh[word + 5u]),
      bitcast<f32>(animated_shadow_bvh[word + 6u])
    ),
    animated_shadow_bvh[word + 7u]
  );
}

fn intersect_animated_shadow_bvh(
  ray: Ray,
  maximum_distance: f32,
  local_invocation_index: u32
) -> ShadowHit {
  let node_count = animated_shadow_bvh[1];
  let index_offset = animated_shadow_bvh[2];
  let index_count = animated_shadow_bvh[3];
  let triangle_count = min(u32(globals.scene_counts.z), arrayLength(&triangles));
  if (node_count == 0u || triangle_count == 0u) {
    return shadow_miss(maximum_distance);
  }

  var closest_hit = shadow_miss(maximum_distance);
  let near_zero = abs(ray.direction) < vec3<f32>(0.0000001);
  let safe_direction = select(ray.direction, vec3<f32>(0.0000001), near_zero);
  let inverse_direction = vec3<f32>(1.0) / safe_direction;
  let stack_base = local_invocation_index * BVH_STACK_CAPACITY;
  var stack_size = 1u;
  bvh_traversal_stack[stack_base] = 0u;

  for (var visited_nodes = 0u; visited_nodes < node_count; visited_nodes += 1u) {
    if (stack_size == 0u) {
      break;
    }
    stack_size -= 1u;
    let stack_entry = bvh_traversal_stack[stack_base + stack_size];
    let node_index = stack_entry & BVH_STACK_NODE_INDEX_MASK;
    let bounds_accepted = (stack_entry & BVH_STACK_BOUNDS_ACCEPTED_FLAG) != 0u;
    if (node_index >= node_count) {
      continue;
    }
    let node = load_animated_shadow_bvh_node(node_index);
    if (!bounds_accepted
        && intersect_bounds(ray, inverse_direction, node, closest_hit.t) >= closest_hit.t) {
      continue;
    }
    let left_first = node.left_first;
    let primitive_count = node.count;
    if (primitive_count > 0u) {
      for (var primitive = 0u; primitive < 4u; primitive += 1u) {
        if (primitive >= primitive_count || left_first >= index_count
            || primitive >= index_count - left_first) {
          break;
        }
        let triangle_index = animated_shadow_bvh[index_offset + left_first + primitive];
        if (triangle_index >= triangle_count) {
          continue;
        }
        let triangle = triangles[triangle_index];
        if (triangle_is_unlit(triangle)) {
          continue;
        }
        let distance = intersect_shadow_triangle_distance(ray, triangle, closest_hit.t);
        if (distance >= closest_hit.t) {
          continue;
        }
        if ((triangle.shadow_flags & TRIANGLE_SHADOW_CACHE_VALID) != 0u
            && (triangle.shadow_flags & TRIANGLE_SHADOW_OPAQUE) != 0u) {
          return ShadowHit(distance, 0.0, 1u, 0u, vec3<f32>(1.0), 0.0);
        }
        let candidate = shadow_hit_from_triangle(distance, triangle);
        if (candidate.transmission <= 0.001) {
          return candidate;
        }
        closest_hit = candidate;
      }
    } else if (left_first > node_index && left_first < node_count - 1u
        && stack_size <= BVH_STACK_CAPACITY - 2u) {
      let left_node = load_animated_shadow_bvh_node(left_first);
      let right_node = load_animated_shadow_bvh_node(left_first + 1u);
      let left_distance = intersect_bounds(ray, inverse_direction, left_node, closest_hit.t);
      let right_distance = intersect_bounds(ray, inverse_direction, right_node, closest_hit.t);
      if (left_distance < closest_hit.t && right_distance < closest_hit.t) {
        let near_child = select(left_first + 1u, left_first, left_distance <= right_distance);
        let far_child = select(left_first, left_first + 1u, left_distance <= right_distance);
        bvh_traversal_stack[stack_base + stack_size] = far_child;
        stack_size += 1u;
        bvh_traversal_stack[stack_base + stack_size] = near_child
          | BVH_STACK_BOUNDS_ACCEPTED_FLAG;
        stack_size += 1u;
      } else if (left_distance < closest_hit.t) {
        bvh_traversal_stack[stack_base + stack_size] = left_first
          | BVH_STACK_BOUNDS_ACCEPTED_FLAG;
        stack_size += 1u;
      } else if (right_distance < closest_hit.t) {
        bvh_traversal_stack[stack_base + stack_size] = (left_first + 1u)
          | BVH_STACK_BOUNDS_ACCEPTED_FLAG;
        stack_size += 1u;
      }
    }
  }
  return closest_hit;
}

fn intersect_shadow_bvh(
  ray: Ray,
  maximum_distance: f32,
  shadow_filter: u32,
  local_invocation_index: u32
) -> ShadowHit {
  if (shadow_filter == SHADOW_FILTER_ANIMATED && animated_shadow_bvh_is_valid()) {
    return intersect_animated_shadow_bvh(ray, maximum_distance, local_invocation_index);
  }
  let node_count = min(u32(globals.scene_counts.y), arrayLength(&bvh_nodes));
  let triangle_count = min(u32(globals.scene_counts.z), arrayLength(&triangles));
  if (node_count == 0u || triangle_count == 0u) {
    return shadow_miss(maximum_distance);
  }

  var closest_hit = shadow_miss(maximum_distance);
  let near_zero = abs(ray.direction) < vec3<f32>(0.0000001);
  let safe_direction = select(ray.direction, vec3<f32>(0.0000001), near_zero);
  let inverse_direction = vec3<f32>(1.0) / safe_direction;
  let stack_base = local_invocation_index * BVH_STACK_CAPACITY;
  var stack_size = 1u;
  bvh_traversal_stack[stack_base] = 0u;

  for (var visited_nodes = 0u; visited_nodes < node_count; visited_nodes += 1u) {
    if (stack_size == 0u) {
      break;
    }
    stack_size -= 1u;
    let stack_entry = bvh_traversal_stack[stack_base + stack_size];
    let node_index = stack_entry & BVH_STACK_NODE_INDEX_MASK;
    let bounds_accepted = (stack_entry & BVH_STACK_BOUNDS_ACCEPTED_FLAG) != 0u;
    if (node_index >= node_count) {
      continue;
    }
    let node = bvh_nodes[node_index];
    if (!shadow_filter_accepts_node(node, shadow_filter)) {
      continue;
    }
    if (!bounds_accepted
        && intersect_bounds(ray, inverse_direction, node, closest_hit.t) >= closest_hit.t) {
      continue;
    }

    let left_first = node.left_first;
    let primitive_count = node.count & BVH_NODE_PRIMITIVE_COUNT_MASK;
    if (primitive_count > 0u) {
      for (var primitive = 0u; primitive < 4u; primitive += 1u) {
        if (primitive >= primitive_count || left_first >= triangle_count
            || primitive >= triangle_count - left_first) {
          break;
        }
        let triangle_index = left_first + primitive;
        let triangle = triangles[triangle_index];
        if (!shadow_filter_accepts_triangle(triangle, shadow_filter)) {
          continue;
        }
        // Visual-only backdrop cards must not become the nearest shadow hit or
        // consume one of the bounded transparent-shadow traversal layers.
        if (triangle_is_unlit(triangle)) {
          continue;
        }
        let distance = intersect_shadow_triangle_distance(
          ray,
          triangle,
          closest_hit.t
        );
        if (distance >= closest_hit.t) {
          continue;
        }
        if ((triangle.shadow_flags & TRIANGLE_SHADOW_CACHE_VALID) != 0u
            && (triangle.shadow_flags & TRIANGLE_SHADOW_OPAQUE) != 0u) {
          return ShadowHit(distance, 0.0, 1u, 0u, vec3<f32>(1.0), 0.0);
        }
        let candidate = shadow_hit_from_triangle(distance, triangle);
        if (candidate.transmission <= 0.001) {
          return candidate;
        }
        closest_hit = candidate;
      }
    } else if (left_first > node_index && left_first < node_count - 1u
        && stack_size <= BVH_STACK_CAPACITY - 2u) {
      let left_node = bvh_nodes[left_first];
      let right_node = bvh_nodes[left_first + 1u];
      var left_distance = 1e30;
      var right_distance = 1e30;
      if (shadow_filter_accepts_node(left_node, shadow_filter)) {
        left_distance = intersect_bounds(ray, inverse_direction, left_node, closest_hit.t);
      }
      if (shadow_filter_accepts_node(right_node, shadow_filter)) {
        right_distance = intersect_bounds(ray, inverse_direction, right_node, closest_hit.t);
      }
      if (left_distance < closest_hit.t && right_distance < closest_hit.t) {
        let near_child = select(left_first + 1u, left_first, left_distance <= right_distance);
        let far_child = select(left_first, left_first + 1u, left_distance <= right_distance);
        bvh_traversal_stack[stack_base + stack_size] = far_child;
        stack_size += 1u;
        bvh_traversal_stack[stack_base + stack_size] = near_child
          | BVH_STACK_BOUNDS_ACCEPTED_FLAG;
        stack_size += 1u;
      } else if (left_distance < closest_hit.t) {
        bvh_traversal_stack[stack_base + stack_size] = left_first
          | BVH_STACK_BOUNDS_ACCEPTED_FLAG;
        stack_size += 1u;
      } else if (right_distance < closest_hit.t) {
        bvh_traversal_stack[stack_base + stack_size] = (left_first + 1u)
          | BVH_STACK_BOUNDS_ACCEPTED_FLAG;
        stack_size += 1u;
      }
    }
  }
  return closest_hit;
}

fn intersect_shadow_scene(
  ray: Ray,
  maximum_distance: f32,
  shadow_filter: u32,
  local_invocation_index: u32
) -> ShadowHit {
  let sphere_count = min(min(u32(globals.scene_counts.x), arrayLength(&spheres)), MAX_SPHERES);
  if (shadow_filter != SHADOW_FILTER_ANIMATED) {
    for (var sphere_index = 0u; sphere_index < sphere_count; sphere_index += 1u) {
      let distance = intersect_sphere_distance(ray, spheres[sphere_index], maximum_distance);
      if (distance < maximum_distance) {
        return ShadowHit(
          distance,
          0.0,
          1u,
          0u,
          max(spheres[sphere_index].color_roughness.xyz, vec3<f32>(0.0)),
          0.0
        );
      }
    }
  }

  let triangle_count = min(u32(globals.scene_counts.z), arrayLength(&triangles));
  let triangle_hit = intersect_shadow_bvh(
    ray,
    maximum_distance,
    shadow_filter,
    local_invocation_index
  );
  if (triangle_hit.hit != 0u) {
    return triangle_hit;
  }
  if (shadow_filter != SHADOW_FILTER_ANIMATED
      && triangle_count == 0u
      && ray.direction.y < -0.00001) {
    let plane_distance = -ray.origin.y / ray.direction.y;
    if (plane_distance > INTERSECTION_EPSILON && plane_distance < maximum_distance) {
      return ShadowHit(plane_distance, 0.0, 1u, 0u, vec3<f32>(0.5), 0.0);
    }
  }
  return shadow_miss(maximum_distance);
}

fn shadow_transmittance_filtered(
  origin: vec3<f32>,
  direction: vec3<f32>,
  maximum_distance: f32,
  shadow_filter: u32,
  local_invocation_index: u32
) -> vec3<f32> {
  var ray = Ray(origin, direction, 0.0, 0u);
  var traveled = 0.0;
  var visibility = vec3<f32>(1.0);
  for (var layer = 0u; layer < 8u; layer += 1u) {
    let remaining_distance = maximum_distance - traveled;
    if (remaining_distance <= INTERSECTION_EPSILON) {
      return visibility;
    }
    let shadow_hit = intersect_shadow_scene(
      ray,
      remaining_distance,
      shadow_filter,
      local_invocation_index
    );
    if (shadow_hit.hit == 0u) {
      return visibility;
    }
    if (shadow_hit.transmission <= 0.001) {
      return vec3<f32>(0.0);
    }
    visibility *= mix(vec3<f32>(1.0), shadow_hit.tint, 0.2) * shadow_hit.transmission;
    if (max(visibility.r, max(visibility.g, visibility.b)) < 0.005) {
      return vec3<f32>(0.0);
    }
    let advance = shadow_hit.t + RAY_ORIGIN_BIAS;
    traveled += advance;
    ray.origin += direction * advance;
  }
  return visibility;
}

fn shadow_transmittance(
  origin: vec3<f32>,
  direction: vec3<f32>,
  maximum_distance: f32,
  local_invocation_index: u32
) -> vec3<f32> {
  return shadow_transmittance_filtered(
    origin,
    direction,
    maximum_distance,
    SHADOW_FILTER_ALL,
    local_invocation_index
  );
}

fn make_basis_direction(normal_value: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
  let radius = sqrt(random(state));
  let angle = 2.0 * PI * random(state);
  let local = vec3<f32>(radius * cos(angle), sqrt(max(0.0, 1.0 - radius * radius)), radius * sin(angle));
  let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal_value.y) > 0.98);
  let tangent = normalize(cross(helper, normal_value));
  let bitangent = cross(normal_value, tangent);
  return normalize(tangent * local.x + normal_value * local.y + bitangent * local.z);
}

fn random_unit_vector(state: ptr<function, u32>) -> vec3<f32> {
  let z = random(state) * 2.0 - 1.0;
  let angle = random(state) * 2.0 * PI;
  let radius = sqrt(max(1.0 - z * z, 0.0));
  return vec3<f32>(radius * cos(angle), z, radius * sin(angle));
}

fn sanitized_ray_weight(value: f32) -> f32 {
  if (value != value || value <= 0.0) {
    return 0.0;
  }
  return min(value, 1000000.0);
}

fn normalized_ray_lobes(base_probabilities: vec3<f32>) -> vec3<f32> {
  let weights = vec3<f32>(
    sanitized_ray_weight(globals.ray_type_weights.x),
    sanitized_ray_weight(globals.ray_type_weights.y),
    sanitized_ray_weight(globals.ray_type_weights.z)
  );
  let weighted = max(base_probabilities, vec3<f32>(0.0)) * weights;
  let largest = max(weighted.x, max(weighted.y, weighted.z));
  if (largest <= 0.00000001) {
    return vec3<f32>(0.0);
  }
  let scaled = weighted / largest;
  let total = scaled.x + scaled.y + scaled.z;
  if (total <= 0.00000001 || total != total) {
    return vec3<f32>(0.0);
  }
  return scaled / total;
}

fn ray_family_enabled(weight: f32) -> f32 {
  return select(0.0, 1.0, sanitized_ray_weight(weight) > 0.0);
}

fn direct_surface_lobe_controls(
  base_probabilities: vec3<f32>,
  total_internal_reflection: bool
) -> vec2<f32> {
  let diffuse_enabled = base_probabilities.x > 0.00000001
    && ray_family_enabled(globals.ray_type_weights.x) > 0.0;
  let glossy_family_enabled = ray_family_enabled(globals.ray_type_weights.y) > 0.0
    || (total_internal_reflection && ray_family_enabled(globals.ray_type_weights.z) > 0.0);
  let glossy_enabled = (base_probabilities.y > 0.00000001 || total_internal_reflection)
    && glossy_family_enabled;
  return vec2<f32>(
    select(0.0, 1.0, diffuse_enabled),
    select(0.0, 1.0, glossy_enabled)
  );
}

fn sample_direction_cone(direction: vec3<f32>, angular_radius: f32, state: ptr<function, u32>) -> vec3<f32> {
  let radius = tan(clamp(angular_radius, 0.0, 1.55)) * sqrt(random(state));
  let angle = random(state) * 2.0 * PI;
  let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(direction.y) > 0.98);
  let tangent = normalize(cross(helper, direction));
  let bitangent = cross(direction, tangent);
  return normalize(direction + (tangent * cos(angle) + bitangent * sin(angle)) * radius);
}

fn fresnel_schlick(cosine: f32, reflectance_zero: vec3<f32>) -> vec3<f32> {
  let complement = clamp(1.0 - cosine, 0.0, 1.0);
  let squared = complement * complement;
  let fifth_power = squared * squared * complement;
  return reflectance_zero + (vec3<f32>(1.0) - reflectance_zero) * fifth_power;
}

fn luminance(value: vec3<f32>) -> f32 {
  return dot(max(value, vec3<f32>(0.0)), vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn surface_reflectance_zero(hit: Hit) -> vec3<f32> {
  let dielectric_f0 = clamp(pow((hit.ior - 1.0) / (hit.ior + 1.0), 2.0), 0.0, 1.0);
  return mix(vec3<f32>(dielectric_f0), max(hit.albedo, vec3<f32>(0.0)), hit.metallic);
}

fn surface_transport_roughness(hit: Hit) -> f32 {
  let authored_roughness = clamp(hit.roughness, 0.02, 1.0);
  return select(authored_roughness, max(authored_roughness, 0.5), hit_is_foliage(hit));
}

fn thin_sheet_fresnel(
  hit: Hit,
  incoming_direction: vec3<f32>,
  surface_normal: vec3<f32>
) -> vec3<f32> {
  let view_cosine = clamp(abs(dot(surface_normal, incoming_direction)), 0.0, 1.0);
  let interface_fresnel = fresnel_schlick(view_cosine, surface_reflectance_zero(hit));
  return clamp(
    (2.0 * interface_fresnel) / max(vec3<f32>(1.0) + interface_fresnel, vec3<f32>(0.00001)),
    vec3<f32>(0.0),
    vec3<f32>(1.0)
  );
}

fn thin_glass_transmission_coefficient(
  hit: Hit,
  sheet_fresnel: vec3<f32>
) -> vec3<f32> {
  let tint = mix(vec3<f32>(1.0), clamp(hit.albedo, vec3<f32>(0.0), vec3<f32>(1.0)), 0.2);
  let transmitted_energy = (vec3<f32>(1.0) - sheet_fresnel)
    * (1.0 - hit.metallic)
    * hit.transmission
    * tint;
  return clamp(transmitted_energy, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn thin_glass_verified_backplate_transmission(
  direction: vec3<f32>,
  transmission_coefficient: vec3<f32>
) -> vec3<f32> {
  return sanitize_radiance(
    backplate_radiance(direction) * transmission_coefficient
  );
}

fn thin_glass_lobe_probabilities(
  reflection_coefficient: vec3<f32>,
  transmission_coefficient: vec3<f32>,
  ray_family_weights: vec3<f32>
) -> vec2<f32> {
  let weighted = vec2<f32>(
    luminance(reflection_coefficient) * sanitized_ray_weight(ray_family_weights.y),
    luminance(transmission_coefficient) * sanitized_ray_weight(ray_family_weights.z)
  );
  let largest = max(weighted.x, weighted.y);
  if (largest <= 0.00000001) {
    return vec2<f32>(0.0);
  }
  let scaled = weighted / largest;
  return scaled / max(scaled.x + scaled.y, 0.00000001);
}

fn thin_glass_environment_reflection(
  hit: Hit,
  incoming_direction: vec3<f32>,
  path_solar_disc_preservation: f32
) -> vec3<f32> {
  let outgoing_direction = safe_normalize(-incoming_direction, hit.normal);
  var reflection_normal = hit.normal;
  var reflection_direction = safe_normalize(
    reflect(incoming_direction, reflection_normal),
    hit.geometric_normal
  );
  if (!valid_geometric_reflection(hit, outgoing_direction, reflection_direction)) {
    reflection_normal = hit.geometric_normal;
    reflection_direction = safe_normalize(
      reflect(incoming_direction, reflection_normal),
      hit.geometric_normal
    );
  }
  let roughness = surface_transport_roughness(hit);
  let maximum_lod = f32(max(textureNumLevels(environment_texture), 1u) - 1u);
  let environment = partitioned_environment_radiance_lod(
    reflection_direction,
    roughness * roughness * maximum_lod,
    min(
      clamp(path_solar_disc_preservation, 0.0, 1.0),
      solar_disc_preservation_for_roughness(roughness)
    ),
    false
  );
  return sanitize_radiance(
    environment * thin_sheet_fresnel(hit, incoming_direction, reflection_normal)
  );
}

fn foliage_transmission_tint(hit: Hit) -> vec3<f32> {
  let bounded_albedo = clamp(hit.albedo, vec3<f32>(0.0), vec3<f32>(1.0));
  return mix(bounded_albedo, sqrt(bounded_albedo), 0.35);
}

fn foliage_transmission_profile(
  outgoing_direction: vec3<f32>,
  incident_direction: vec3<f32>
) -> f32 {
  let forward_alignment = clamp(dot(outgoing_direction, -incident_direction), 0.0, 1.0);
  return mix(0.65, 1.0, forward_alignment * forward_alignment);
}

fn foliage_diffuse_transmission_response(
  hit: Hit,
  outgoing_direction: vec3<f32>,
  incident_direction: vec3<f32>
) -> vec3<f32> {
  if (!hit_is_foliage(hit) || hit.transmission <= 0.000001 || hit.metallic >= 0.999999) {
    return vec3<f32>(0.0);
  }
  let outgoing_fresnel = fresnel_schlick(
    clamp(dot(hit.normal, outgoing_direction), 0.0, 1.0),
    surface_reflectance_zero(hit)
  );
  let incident_fresnel = fresnel_schlick(
    clamp(dot(-hit.normal, incident_direction), 0.0, 1.0),
    surface_reflectance_zero(hit)
  );
  let interface_transmission = sqrt(max(
    (vec3<f32>(1.0) - outgoing_fresnel)
      * (vec3<f32>(1.0) - incident_fresnel),
    vec3<f32>(0.0)
  ));
  return interface_transmission
    * (1.0 - hit.metallic)
    * hit.transmission
    * foliage_transmission_tint(hit)
    * foliage_transmission_profile(outgoing_direction, incident_direction);
}

fn foliage_environment_transmission_response(
  hit: Hit,
  outgoing_direction: vec3<f32>
) -> vec3<f32> {
  if (!hit_is_foliage(hit) || hit.transmission <= 0.000001 || hit.metallic >= 0.999999) {
    return vec3<f32>(0.0);
  }
  let fresnel = fresnel_schlick(
    clamp(dot(hit.normal, outgoing_direction), 0.0, 1.0),
    surface_reflectance_zero(hit)
  );
  return (vec3<f32>(1.0) - fresnel)
    * (1.0 - hit.metallic)
    * hit.transmission
    * foliage_transmission_tint(hit)
    * 0.79;
}

fn fifth_power_scalar(value: f32) -> f32 {
  let bounded = clamp(value, 0.0, 1.0);
  let squared = bounded * bounded;
  return squared * squared * bounded;
}

fn fabric_sheen_weight(hit: Hit) -> f32 {
  if (!hit_is_fabric(hit) || hit.metallic >= 0.999999 || hit.transmission >= 0.999999) {
    return 0.0;
  }
  return mix(0.12, 0.24, clamp(hit.roughness, 0.0, 1.0));
}

fn fabric_rough_diffuse_factor(
  hit: Hit,
  outgoing_direction: vec3<f32>,
  incident_direction: vec3<f32>
) -> f32 {
  if (!hit_is_fabric(hit)) {
    return 1.0;
  }
  let normal_dot_view = clamp(dot(hit.normal, outgoing_direction), 0.0, 1.0);
  let normal_dot_light = clamp(dot(hit.normal, incident_direction), 0.0, 1.0);
  let sigma_squared = hit.roughness * hit.roughness * 0.65 * 0.65;
  let oren_a = 1.0 - 0.5 * sigma_squared / (sigma_squared + 0.33);
  let oren_b = 0.45 * sigma_squared / (sigma_squared + 0.09);
  let tangent_correlation = dot(outgoing_direction, incident_direction)
    - normal_dot_view * normal_dot_light;
  let retroreflection = select(
    0.0,
    oren_b * tangent_correlation / max(max(normal_dot_view, normal_dot_light), 0.0001),
    tangent_correlation > 0.0
  );
  return clamp(oren_a + retroreflection, 0.0, 1.0);
}

fn diffuse_reflection_response(
  hit: Hit,
  outgoing_direction: vec3<f32>,
  incident_direction: vec3<f32>,
  fresnel: vec3<f32>
) -> vec3<f32> {
  let available_energy = (vec3<f32>(1.0) - fresnel)
    * (1.0 - hit.metallic)
    * (1.0 - hit.transmission);
  let bounded_albedo = clamp(hit.albedo, vec3<f32>(0.0), vec3<f32>(1.0));
  let sheen_weight = fabric_sheen_weight(hit);
  if (sheen_weight <= 0.0) {
    return available_energy * bounded_albedo;
  }

  let normal_dot_view = clamp(dot(hit.normal, outgoing_direction), 0.0, 1.0);
  let normal_dot_light = clamp(dot(hit.normal, incident_direction), 0.0, 1.0);
  let half_vector = safe_normalize(
    outgoing_direction + incident_direction,
    hit.normal
  );
  let view_dot_half = clamp(dot(outgoing_direction, half_vector), 0.0, 1.0);
  let grazing_fibers = sqrt(
    max((1.0 - normal_dot_view) * (1.0 - normal_dot_light), 0.0)
  );
  let retro_fibers = fifth_power_scalar(1.0 - view_dot_half);
  let sheen_profile = clamp(max(grazing_fibers, retro_fibers), 0.0, 1.0);
  let fiber_tint = sqrt(bounded_albedo);
  let allocated_albedo = bounded_albedo * (1.0 - sheen_weight)
    + fiber_tint * sheen_weight * sheen_profile;
  return available_energy
    * allocated_albedo
    * fabric_rough_diffuse_factor(hit, outgoing_direction, incident_direction);
}

fn diffuse_environment_response(
  hit: Hit,
  outgoing_direction: vec3<f32>,
  fresnel: vec3<f32>
) -> vec3<f32> {
  let available_energy = (vec3<f32>(1.0) - fresnel)
    * (1.0 - hit.metallic)
    * (1.0 - hit.transmission);
  let bounded_albedo = clamp(hit.albedo, vec3<f32>(0.0), vec3<f32>(1.0));
  let sheen_weight = fabric_sheen_weight(hit);
  if (sheen_weight <= 0.0) {
    return available_energy * bounded_albedo;
  }
  let normal_dot_view = clamp(dot(hit.normal, outgoing_direction), 0.0, 1.0);
  let integrated_sheen = clamp(
    0.28 + 0.42 * (1.0 - normal_dot_view) * (1.0 - normal_dot_view),
    0.0,
    1.0
  );
  let fiber_tint = sqrt(bounded_albedo);
  let allocated_albedo = bounded_albedo * (1.0 - sheen_weight)
    + fiber_tint * sheen_weight * integrated_sheen;
  let hemispherical_rough_diffuse = clamp(
    1.0 - hit.roughness * hit.roughness * 0.18,
    0.0,
    1.0
  );
  return available_energy * allocated_albedo * hemispherical_rough_diffuse;
}

fn physical_path_lobe_probabilities(
  hit: Hit,
  outgoing_direction: vec3<f32>,
  total_internal_reflection: bool
) -> vec3<f32> {
  if (total_internal_reflection) {
    return vec3<f32>(0.0, 1.0, 0.0);
  }
  let reflectance_zero = surface_reflectance_zero(hit);
  let fresnel = fresnel_schlick(
    clamp(dot(hit.normal, outgoing_direction), 0.0, 1.0),
    reflectance_zero
  );
  let nonmetal = 1.0 - hit.metallic;
  let diffuse_response = diffuse_environment_response(
    hit,
    outgoing_direction,
    fresnel
  );
  let dielectric_transmission_response = (vec3<f32>(1.0) - fresnel)
    * nonmetal
    * hit.transmission
    * mix(vec3<f32>(1.0), max(hit.albedo, vec3<f32>(0.0)), 0.2);
  let transmission_response = select(
    dielectric_transmission_response,
    foliage_environment_transmission_response(hit, outgoing_direction),
    hit_is_foliage(hit)
  );
  return vec3<f32>(
    luminance(diffuse_response),
    luminance(fresnel),
    luminance(transmission_response)
  );
}

fn shading_normal_correction(
  hit: Hit,
  outgoing_direction: vec3<f32>,
  incident_direction: vec3<f32>
) -> f32 {
  let numerator = abs(
    dot(outgoing_direction, hit.normal)
      * dot(incident_direction, hit.geometric_normal)
  );
  let denominator = abs(
    dot(outgoing_direction, hit.geometric_normal)
      * dot(incident_direction, hit.normal)
  );
  if (denominator <= 0.00001 || numerator != numerator || denominator != denominator) {
    return 0.0;
  }
  return clamp(numerator / denominator, 0.0, 4.0);
}

fn surface_has_total_internal_reflection(
  hit: Hit,
  incoming_direction: vec3<f32>
) -> bool {
  if (hit_is_foliage(hit) || hit.transmission <= 0.000001 || hit.metallic >= 0.999999) {
    return false;
  }
  let eta = select(hit.ior, 1.0 / hit.ior, hit.front_face != 0u);
  let refracted = refract(incoming_direction, hit.normal, eta);
  return dot(refracted, refracted) <= 0.000001;
}

fn terminal_ray_family_weights() -> vec3<f32> {
  return vec3<f32>(
    sanitized_ray_weight(globals.ray_type_weights.x),
    sanitized_ray_weight(globals.ray_type_weights.y),
    sanitized_ray_weight(globals.ray_type_weights.z)
  );
}

fn terminal_ray_trace_environment(
  hit: Hit,
  incoming_direction: vec3<f32>,
  ray_family_weights: vec3<f32>,
  path_solar_disc_preservation: f32,
  background_visibility: u32
) -> vec3<f32> {
  if (ray_family_weights.x + ray_family_weights.y + ray_family_weights.z <= 0.0) {
    return vec3<f32>(0.0);
  }
  let normal = safe_normalize(hit.normal, vec3<f32>(0.0, 1.0, 0.0));
  let view_direction = safe_normalize(-incoming_direction, normal);
  let normal_dot_view = clamp(dot(normal, view_direction), 0.0, 1.0);
  let reflectance_zero = surface_reflectance_zero(hit);
  let transmissive = hit.transmission > 0.000001 && hit.metallic < 0.999999;
  var raw_refraction = vec3<f32>(0.0);
  var refraction_length_squared = 0.0;
  var total_internal_reflection = false;
  if (transmissive) {
    if (!hit_is_foliage(hit)) {
      let eta = select(hit.ior, 1.0 / hit.ior, hit.front_face != 0u);
      raw_refraction = refract(incoming_direction, normal, eta);
      refraction_length_squared = dot(raw_refraction, raw_refraction);
      total_internal_reflection = refraction_length_squared <= 0.000001;
    }
  }
  let schlick_fresnel = fresnel_schlick(normal_dot_view, reflectance_zero);
  let fresnel = select(schlick_fresnel, vec3<f32>(1.0), total_internal_reflection);
  let diffuse_available = hit.metallic < 0.999999 && hit.transmission < 0.999999;
  let transmission_available = transmissive && !total_internal_reflection;
  var active_weights = vec3<f32>(
    select(0.0, ray_family_weights.x, diffuse_available),
    ray_family_weights.y,
    select(0.0, ray_family_weights.z, transmission_available)
  );
  if (total_internal_reflection) {
    active_weights.y = max(ray_family_weights.y, ray_family_weights.z);
    active_weights.z = 0.0;
  }
  let largest_active_weight = max(active_weights.x, max(active_weights.y, active_weights.z));
  if (largest_active_weight <= 0.00000001) {
    return vec3<f32>(0.0);
  }
  let ray_family_controls = active_weights / largest_active_weight;
  var result = vec3<f32>(0.0);

  if (ray_family_controls.x > 0.0 && hit.metallic < 0.999999 && hit.transmission < 0.999999) {
    let diffuse_response = diffuse_environment_response(
      hit,
      view_direction,
      fresnel
    );
    result += diffuse_environment_irradiance(normal)
      * diffuse_response
      * ray_family_controls.x
      / PI;
  }

  let reflection_control = select(
    ray_family_controls.y,
    max(ray_family_controls.y, ray_family_controls.z),
    total_internal_reflection
  );
  if (reflection_control > 0.0) {
    let roughness = surface_transport_roughness(hit);
    let reflection_direction = safe_normalize(reflect(incoming_direction, normal), normal);
    let maximum_lod = f32(max(textureNumLevels(environment_texture), 1u) - 1u);
    let incoming_sun_preservation = clamp(path_solar_disc_preservation, 0.0, 1.0);
    let reflected_sun_preservation = select(
      min(
        incoming_sun_preservation,
        solar_disc_preservation_for_roughness(surface_transport_roughness(hit))
      ),
      incoming_sun_preservation,
      total_internal_reflection
    );
    let reflected_environment = partitioned_environment_radiance_lod(
      reflection_direction,
      roughness * roughness * maximum_lod,
      reflected_sun_preservation,
      false
    );
    let brdf_c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let brdf_c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let brdf_r = roughness * brdf_c0 + brdf_c1;
    let brdf_a004 = min(brdf_r.x * brdf_r.x, exp2(-9.28 * normal_dot_view))
      * brdf_r.x + brdf_r.y;
    let brdf_ab = vec2<f32>(-1.04, 1.04) * brdf_a004 + brdf_r.zw;
    let split_sum_response = max(
      reflectance_zero * brdf_ab.x + vec3<f32>(brdf_ab.y),
      vec3<f32>(0.0)
    );
    let specular_response = select(
      split_sum_response,
      vec3<f32>(1.0),
      total_internal_reflection
    );
    result += reflected_environment * specular_response * reflection_control;
  }

  if (ray_family_controls.z > 0.0 && transmissive && !total_internal_reflection) {
    if (hit_is_foliage(hit)) {
      result += diffuse_environment_irradiance(-normal)
        * foliage_environment_transmission_response(hit, view_direction)
        * ray_family_controls.z
        / PI;
    } else {
      let roughness = surface_transport_roughness(hit);
      let maximum_lod = f32(max(textureNumLevels(environment_texture), 1u) - 1u);
      let refraction_direction = raw_refraction * inverseSqrt(refraction_length_squared);
      var transmitted_environment = partitioned_environment_radiance_lod(
        refraction_direction,
        roughness * maximum_lod * 0.5,
        clamp(path_solar_disc_preservation, 0.0, 1.0),
        false
      );
      if (background_visibility != 0u
          && globals.backplate_texture_params.x > 0.5
          && backplate_transmission_preserves_visibility(hit)) {
        transmitted_environment = backplate_radiance(refraction_direction);
      }
      let transmitted_energy = (vec3<f32>(1.0) - fresnel)
        * (1.0 - hit.metallic)
        * hit.transmission;
      result += transmitted_environment
        * transmitted_energy
        * mix(vec3<f32>(1.0), hit.albedo, 0.2)
        * ray_family_controls.z;
    }
  }
  return sanitize_radiance(result);
}

fn ggx_alpha(roughness: f32) -> f32 {
  let safe_roughness = clamp(roughness, 0.02, 1.0);
  return safe_roughness * safe_roughness;
}

fn ggx_distribution(normal_dot_half: f32, roughness: f32) -> f32 {
  let alpha = ggx_alpha(roughness);
  let alpha_squared = alpha * alpha;
  let denominator_term = normal_dot_half * normal_dot_half * (alpha_squared - 1.0) + 1.0;
  return alpha_squared / max(PI * denominator_term * denominator_term, 0.00000000000000000001);
}

fn ggx_smith_lambda(normal_dot_direction: f32, roughness: f32) -> f32 {
  let cosine = clamp(abs(normal_dot_direction), 0.00001, 1.0);
  let cosine_squared = cosine * cosine;
  let tangent_squared = max(1.0 - cosine_squared, 0.0) / cosine_squared;
  let alpha = ggx_alpha(roughness);
  return 0.5 * (sqrt(1.0 + alpha * alpha * tangent_squared) - 1.0);
}

fn ggx_smith_g1(normal_dot_direction: f32, roughness: f32) -> f32 {
  if (normal_dot_direction <= 0.0) {
    return 0.0;
  }
  return 1.0 / (1.0 + ggx_smith_lambda(normal_dot_direction, roughness));
}

fn ggx_smith_g2(
  normal_dot_view: f32,
  normal_dot_light: f32,
  roughness: f32
) -> f32 {
  return ggx_smith_g1(normal_dot_view, roughness)
    * ggx_smith_g1(normal_dot_light, roughness);
}

fn ggx_visible_normal_pdf(
  normal_dot_view: f32,
  normal_dot_half: f32,
  view_dot_half: f32,
  roughness: f32
) -> f32 {
  if (normal_dot_view <= 0.0 || normal_dot_half <= 0.0 || view_dot_half <= 0.0) {
    return 0.0;
  }
  return ggx_distribution(normal_dot_half, roughness)
    * ggx_smith_g1(normal_dot_view, roughness)
    * view_dot_half
    / max(normal_dot_view, 0.00001);
}

fn ggx_reflection_pdf(
  normal_dot_view: f32,
  normal_dot_half: f32,
  view_dot_half: f32,
  roughness: f32
) -> f32 {
  return ggx_visible_normal_pdf(
    normal_dot_view,
    normal_dot_half,
    view_dot_half,
    roughness
  ) / max(4.0 * view_dot_half, 0.00001);
}

fn sample_ggx_vndf(
  normal_value: vec3<f32>,
  outgoing_direction: vec3<f32>,
  roughness: f32,
  state: ptr<function, u32>
) -> vec3<f32> {
  let helper = select(
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0),
    abs(normal_value.y) > 0.98
  );
  let tangent = safe_normalize(cross(helper, normal_value), vec3<f32>(1.0, 0.0, 0.0));
  let bitangent = safe_normalize(cross(normal_value, tangent), vec3<f32>(0.0, 0.0, 1.0));
  let local_view = vec3<f32>(
    dot(outgoing_direction, tangent),
    max(dot(outgoing_direction, normal_value), 0.00001),
    dot(outgoing_direction, bitangent)
  );
  let alpha = ggx_alpha(roughness);
  let stretched_view = safe_normalize(
    vec3<f32>(alpha * local_view.x, local_view.y, alpha * local_view.z),
    vec3<f32>(0.0, 1.0, 0.0)
  );
  let tangent_length_squared = stretched_view.x * stretched_view.x
    + stretched_view.z * stretched_view.z;
  var first_tangent = vec3<f32>(1.0, 0.0, 0.0);
  if (tangent_length_squared > 0.00000001) {
    first_tangent = vec3<f32>(-stretched_view.z, 0.0, stretched_view.x)
      * inverseSqrt(tangent_length_squared);
  }
  let second_tangent = cross(first_tangent, stretched_view);
  let radius = sqrt(random(state));
  let phi = 2.0 * PI * random(state);
  let first_coordinate = radius * cos(phi);
  var second_coordinate = radius * sin(phi);
  let visible_blend = 0.5 * (1.0 + stretched_view.y);
  second_coordinate = mix(
    sqrt(max(1.0 - first_coordinate * first_coordinate, 0.0)),
    second_coordinate,
    visible_blend
  );
  let projected_normal = first_coordinate * first_tangent
    + second_coordinate * second_tangent
    + sqrt(max(
      1.0 - first_coordinate * first_coordinate - second_coordinate * second_coordinate,
      0.0
    )) * stretched_view;
  let local_half = safe_normalize(
    vec3<f32>(alpha * projected_normal.x, max(projected_normal.y, 0.0), alpha * projected_normal.z),
    vec3<f32>(0.0, 1.0, 0.0)
  );
  return safe_normalize(
    tangent * local_half.x + normal_value * local_half.y + bitangent * local_half.z,
    normal_value
  );
}

fn diffuse_scatter_weight(
  hit: Hit,
  outgoing_direction: vec3<f32>,
  incident_direction: vec3<f32>,
  lobe_probability: f32
) -> vec3<f32> {
  if (
    lobe_probability <= 0.00000001
    || !valid_geometric_reflection(hit, outgoing_direction, incident_direction)
  ) {
    return vec3<f32>(0.0);
  }
  let normal_dot_view = dot(hit.normal, outgoing_direction);
  let normal_dot_light = dot(hit.normal, incident_direction);
  if (normal_dot_view <= 0.0 || normal_dot_light <= 0.0) {
    return vec3<f32>(0.0);
  }
  let half_sum = outgoing_direction + incident_direction;
  let half_length_squared = dot(half_sum, half_sum);
  if (half_length_squared <= 0.000001) {
    return vec3<f32>(0.0);
  }
  let half_vector = half_sum * inverseSqrt(half_length_squared);
  let fresnel = fresnel_schlick(
    max(dot(outgoing_direction, half_vector), 0.0),
    surface_reflectance_zero(hit)
  );
  let response = diffuse_reflection_response(
    hit,
    outgoing_direction,
    incident_direction,
    fresnel
  );
  return response
    * shading_normal_correction(hit, outgoing_direction, incident_direction)
    / lobe_probability;
}

fn glossy_scatter_weight(
  hit: Hit,
  outgoing_direction: vec3<f32>,
  incident_direction: vec3<f32>,
  microfacet_normal: vec3<f32>,
  lobe_probability: f32
) -> vec3<f32> {
  if (
    lobe_probability <= 0.00000001
    || !valid_geometric_reflection(hit, outgoing_direction, incident_direction)
  ) {
    return vec3<f32>(0.0);
  }
  let normal_dot_view = dot(hit.normal, outgoing_direction);
  let normal_dot_light = dot(hit.normal, incident_direction);
  let normal_dot_half = dot(hit.normal, microfacet_normal);
  let view_dot_half = dot(outgoing_direction, microfacet_normal);
  if (
    normal_dot_view <= 0.0
    || normal_dot_light <= 0.0
    || normal_dot_half <= 0.0
    || view_dot_half <= 0.0
  ) {
    return vec3<f32>(0.0);
  }
  let roughness = surface_transport_roughness(hit);
  let distribution = ggx_distribution(normal_dot_half, roughness);
  let masking = ggx_smith_g2(normal_dot_view, normal_dot_light, roughness);
  let fresnel = fresnel_schlick(view_dot_half, surface_reflectance_zero(hit));
  let brdf = fresnel * distribution * masking
    / max(4.0 * normal_dot_view * normal_dot_light, 0.00001);
  let directional_pdf = ggx_reflection_pdf(
    normal_dot_view,
    normal_dot_half,
    view_dot_half,
    roughness
  );
  if (directional_pdf <= 0.00000001) {
    return vec3<f32>(0.0);
  }
  return brdf
    * normal_dot_light
    * shading_normal_correction(hit, outgoing_direction, incident_direction)
    / (directional_pdf * lobe_probability);
}

fn transmission_scatter_weight(
  hit: Hit,
  outgoing_direction: vec3<f32>,
  incident_direction: vec3<f32>,
  relative_ior: f32,
  lobe_probability: f32
) -> vec3<f32> {
  if (
    lobe_probability <= 0.00000001
    || !valid_geometric_transmission(hit, outgoing_direction, incident_direction)
  ) {
    return vec3<f32>(0.0);
  }
  let fresnel = fresnel_schlick(
    clamp(dot(hit.normal, outgoing_direction), 0.0, 1.0),
    surface_reflectance_zero(hit)
  );
  let tint = mix(vec3<f32>(1.0), max(hit.albedo, vec3<f32>(0.0)), 0.2);
  let radiance_compression = relative_ior * relative_ior;
  let response = (vec3<f32>(1.0) - fresnel)
    * (1.0 - hit.metallic)
    * hit.transmission
    * tint
    * radiance_compression;
  return response
    * shading_normal_correction(hit, outgoing_direction, incident_direction)
    / lobe_probability;
}

fn foliage_diffuse_transmission_scatter_weight(
  hit: Hit,
  outgoing_direction: vec3<f32>,
  incident_direction: vec3<f32>,
  lobe_probability: f32
) -> vec3<f32> {
  if (
    lobe_probability <= 0.00000001
    || !hit_is_foliage(hit)
    || !valid_geometric_transmission(hit, outgoing_direction, incident_direction)
  ) {
    return vec3<f32>(0.0);
  }
  let normal_dot_view = dot(hit.normal, outgoing_direction);
  let opposite_normal_dot_light = dot(-hit.normal, incident_direction);
  if (normal_dot_view <= 0.0 || opposite_normal_dot_light <= 0.0) {
    return vec3<f32>(0.0);
  }
  return foliage_diffuse_transmission_response(
    hit,
    outgoing_direction,
    incident_direction
  ) * shading_normal_correction(hit, outgoing_direction, incident_direction)
    / lobe_probability;
}

fn evaluate_light(
  hit: Hit,
  incoming_direction: vec3<f32>,
  light_direction: vec3<f32>,
  radiance: vec3<f32>,
  direct_lobe_controls: vec2<f32>
) -> vec3<f32> {
  let view_direction = safe_normalize(-incoming_direction, hit.normal);
  let reflection_valid = valid_geometric_reflection(hit, view_direction, light_direction);
  let foliage_transmission_valid = hit_is_foliage(hit)
    && hit.transmission > 0.000001
    && ray_family_enabled(globals.ray_type_weights.z) > 0.0
    && valid_geometric_transmission(hit, view_direction, light_direction);
  if (!reflection_valid && !foliage_transmission_valid) {
    return vec3<f32>(0.0);
  }
  if (foliage_transmission_valid) {
    let opposite_normal_dot_light = max(dot(-hit.normal, light_direction), 0.0);
    let normal_correction = shading_normal_correction(
      hit,
      view_direction,
      light_direction
    );
    return sanitize_radiance(
      foliage_diffuse_transmission_response(hit, view_direction, light_direction)
        * radiance
        * opposite_normal_dot_light
        * normal_correction
        / PI
    );
  }
  let n_dot_l = max(dot(hit.normal, light_direction), 0.0);
  let n_dot_v = max(dot(hit.normal, view_direction), 0.0);
  if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
    return vec3<f32>(0.0);
  }
  let half_sum = light_direction + view_direction;
  let half_length_squared = dot(half_sum, half_sum);
  if (half_length_squared <= 0.000001) {
    return vec3<f32>(0.0);
  }
  let half_vector = half_sum * inverseSqrt(half_length_squared);
  let n_dot_h = max(dot(hit.normal, half_vector), 0.0);
  let v_dot_h = max(dot(view_direction, half_vector), 0.0);
  let safe_roughness = surface_transport_roughness(hit);
  let reflectance_zero = surface_reflectance_zero(hit);
  let fresnel = fresnel_schlick(v_dot_h, reflectance_zero);
  let distribution = ggx_distribution(n_dot_h, safe_roughness);
  let masking = ggx_smith_g2(n_dot_v, n_dot_l, safe_roughness);
  let specular = fresnel * distribution * masking / max(4.0 * n_dot_v * n_dot_l, 0.0001);
  let diffuse = diffuse_reflection_response(
    hit,
    view_direction,
    light_direction,
    fresnel
  ) / PI;
  let normal_correction = shading_normal_correction(hit, view_direction, light_direction);
  return sanitize_radiance(
    (diffuse * direct_lobe_controls.x + specular * direct_lobe_controls.y)
      * radiance
      * n_dot_l
      * normal_correction
  );
}

fn foliage_environment_direct_lighting(
  hit: Hit,
  incoming_direction: vec3<f32>,
  state: ptr<function, u32>,
  diffuse_lobe: f32,
  local_invocation_index: u32
) -> vec3<f32> {
  let normal = safe_normalize(hit.normal, vec3<f32>(0.0, 1.0, 0.0));
  let view_direction = safe_normalize(-incoming_direction, normal);
  let view_fresnel = fresnel_schlick(
    clamp(dot(normal, view_direction), 0.0, 1.0),
    surface_reflectance_zero(hit)
  );
  let reflected_importance = luminance(
    diffuse_environment_response(hit, view_direction, view_fresnel)
  ) * max(diffuse_lobe, 0.0);
  let transmission_control = ray_family_enabled(globals.ray_type_weights.z);
  let transmitted_importance = luminance(
    foliage_environment_transmission_response(hit, view_direction)
  ) * transmission_control;
  let total_importance = reflected_importance + transmitted_importance;
  if (total_importance <= 0.00000001) {
    return vec3<f32>(0.0);
  }

  let transmission_probability = transmitted_importance / total_importance;
  let sample_transmission = transmission_probability > 0.0
    && (reflected_importance <= 0.0 || random(state) < transmission_probability);
  let sampling_normal = select(normal, -normal, sample_transmission);
  let environment_direction = make_basis_direction(sampling_normal, state);
  let geometry_valid = select(
    valid_geometric_reflection(hit, view_direction, environment_direction),
    valid_geometric_transmission(hit, view_direction, environment_direction),
    sample_transmission
  );
  if (!geometry_valid) {
    return vec3<f32>(0.0);
  }
  let normal_dot_light = max(dot(sampling_normal, environment_direction), 0.0);
  let cosine_pdf = normal_dot_light / PI;
  if (cosine_pdf <= 0.00000001) {
    return vec3<f32>(0.0);
  }

  let visibility = shadow_transmittance(
    offset_hit_position(hit, environment_direction),
    environment_direction,
    1e29,
    local_invocation_index
  );
  if (max(visibility.r, max(visibility.g, visibility.b)) <= 0.0) {
    return vec3<f32>(0.0);
  }
  let environment_lod = select(
    0.0,
    sunless_environment_lod_offset(),
    captured_sun_matching_active()
  );
  let incident_radiance = partitioned_environment_radiance_lod(
    environment_direction,
    environment_lod,
    0.0,
    false
  );
  var response = vec3<f32>(0.0);
  var family_control = max(diffuse_lobe, 0.0);
  var selected_probability = max(1.0 - transmission_probability, 0.00000001);
  if (sample_transmission) {
    response = foliage_diffuse_transmission_response(
      hit,
      view_direction,
      environment_direction
    ) / PI;
    family_control = transmission_control;
    selected_probability = max(transmission_probability, 0.00000001);
  } else {
    let half_vector = safe_normalize(
      environment_direction + view_direction,
      normal
    );
    let fresnel = fresnel_schlick(
      max(dot(view_direction, half_vector), 0.0),
      surface_reflectance_zero(hit)
    );
    response = diffuse_environment_response(hit, view_direction, fresnel) / PI;
  }
  let normal_correction = shading_normal_correction(
    hit,
    view_direction,
    environment_direction
  );
  return sanitize_radiance(
    response
      * incident_radiance
      * visibility
      * normal_dot_light
      * normal_correction
      * family_control
      / (cosine_pdf * selected_probability)
  );
}

fn diffuse_environment_direct_lighting(
  hit: Hit,
  incoming_direction: vec3<f32>,
  state: ptr<function, u32>,
  diffuse_lobe: f32,
  local_invocation_index: u32
) -> vec3<f32> {
  if (
    hit_is_foliage(hit)
    && hit.transmission > 0.000001
    && ray_family_enabled(globals.ray_type_weights.z) > 0.0
  ) {
    return foliage_environment_direct_lighting(
      hit,
      incoming_direction,
      state,
      diffuse_lobe,
      local_invocation_index
    );
  }
  if (diffuse_lobe <= 0.00000001 || hit.metallic >= 0.999999 || hit.transmission >= 0.999999) {
    return vec3<f32>(0.0);
  }

  let normal = safe_normalize(hit.normal, vec3<f32>(0.0, 1.0, 0.0));
  let environment_direction = make_basis_direction(normal, state);
  let view_direction = safe_normalize(-incoming_direction, normal);
  if (!valid_geometric_reflection(hit, view_direction, environment_direction)) {
    return vec3<f32>(0.0);
  }
  let normal_dot_light = max(dot(normal, environment_direction), 0.0);
  let cosine_pdf = normal_dot_light / PI;
  if (cosine_pdf <= 0.00000001) {
    return vec3<f32>(0.0);
  }

  let visibility = shadow_transmittance(
    offset_hit_position(hit, environment_direction),
    environment_direction,
    1e29,
    local_invocation_index
  );
  if (max(visibility.r, max(visibility.g, visibility.b)) <= 0.0) {
    return vec3<f32>(0.0);
  }

  let environment_lod = select(
    0.0,
    sunless_environment_lod_offset(),
    captured_sun_matching_active()
  );
  let incident_radiance = partitioned_environment_radiance_lod(
    environment_direction,
    environment_lod,
    0.0,
    false
  );
  let half_sum = environment_direction + view_direction;
  let half_length_squared = dot(half_sum, half_sum);
  if (half_length_squared <= 0.000001) {
    return vec3<f32>(0.0);
  }
  let half_vector = half_sum * inverseSqrt(half_length_squared);
  let reflectance_zero = surface_reflectance_zero(hit);
  let fresnel = fresnel_schlick(max(dot(view_direction, half_vector), 0.0), reflectance_zero);
  let diffuse_brdf = diffuse_reflection_response(
    hit,
    view_direction,
    environment_direction,
    fresnel
  ) / PI;
  let normal_correction = shading_normal_correction(hit, view_direction, environment_direction);
  return sanitize_radiance(
    diffuse_brdf
      * incident_radiance
      * visibility
      * normal_dot_light
      * normal_correction
      * max(diffuse_lobe, 0.0)
      / cosine_pdf
  );
}

fn direct_lighting(
  hit: Hit,
  incoming_direction: vec3<f32>,
  state: ptr<function, u32>,
  direct_lobe_controls: vec2<f32>,
  captured_sun_irradiance: vec3<f32>,
  path_solar_disc_preservation: f32,
  current_surface_total_internal_reflection: bool,
  sun_cache: SunVisibilityCache,
  sun_cache_pixel: vec2<u32>,
  sun_cache_dimensions: vec2<u32>,
  local_invocation_index: u32
) -> vec3<f32> {
  var result = vec3<f32>(0.0);
  var sun_cache_stored = false;
  let directional_index = primary_directional_light_index();
  if (directional_index == NO_LIGHT_INDEX) {
    let sun_direction = sample_direction_cone(default_sun_direction(), 0.00465, state);
    var sun_visibility = shadow_transmittance(
      offset_hit_position(hit, sun_direction),
      sun_direction,
      1e29,
      local_invocation_index
    );
    sun_visibility *= cloud_light_transmittance(
      offset_hit_position(hit, sun_direction),
      sun_direction,
      1e29
    );
    result += evaluate_light(
      hit,
      incoming_direction,
      sun_direction,
      vec3<f32>(5.4, 4.35, 3.15) * sun_visibility,
      direct_lobe_controls
    );
  }

  let light_count = min(min(u32(globals.light_volume.x), arrayLength(&lights)), MAX_LIGHTS);
  for (var light_index = 0u; light_index < light_count; light_index += 1u) {
    let light = lights[light_index];
    let kind = light.kind;
    var direction_to_light = normalize(-light.vector);
    var maximum_distance = 1e29;
    var attenuation = 1.0;
    if (kind == 0u) {
      let sampled_position = light.vector + random_unit_vector(state) * max(light.radius, 0.0);
      let delta = sampled_position - hit.position;
      let distance_squared = max(dot(delta, delta), 0.0001);
      maximum_distance = sqrt(distance_squared);
      direction_to_light = delta / maximum_distance;
      let range_value = light.range_or_angular_radius;
      let range_fade = select(
        1.0,
        clamp(1.0 - distance_squared / (range_value * range_value), 0.0, 1.0),
        range_value > 0.0
      );
      attenuation = range_fade * range_fade / distance_squared;
    } else {
      direction_to_light = sample_direction_cone(
        direction_to_light,
        max(light.range_or_angular_radius, 0.0),
        state
      );
      if (light_index == directional_index && sun_cache.valid != 0u) {
        direction_to_light = sun_cache.direction;
      }
    }
    let casts_shadow = (light.flags & 1u) != 0u;
    let reflection_facing = dot(hit.normal, direction_to_light) > 0.0
      && dot(hit.geometric_normal, direction_to_light) > 0.0;
    let foliage_transmission_facing = hit_is_foliage(hit)
      && hit.transmission > 0.000001
      && ray_family_enabled(globals.ray_type_weights.z) > 0.0
      && dot(hit.normal, direction_to_light) < 0.0
      && dot(hit.geometric_normal, direction_to_light) < 0.0;
    if (reflection_facing || foliage_transmission_facing) {
      var visibility = vec3<f32>(1.0);
      if (casts_shadow) {
        if (light_index == directional_index && sun_cache.enabled != 0u) {
          var static_visibility = vec3<f32>(1.0);
          if (sun_cache.valid != 0u) {
            static_visibility = sun_cache.visibility;
          } else {
            let static_filter = select(
              SHADOW_FILTER_ALL,
              SHADOW_FILTER_STATIC,
              globals.environment_params.z > 0.0
            );
            static_visibility = shadow_transmittance_filtered(
              offset_hit_position(hit, direction_to_light),
              direction_to_light,
              maximum_distance - RAY_ORIGIN_BIAS,
              static_filter,
              local_invocation_index
            );
          }
          store_sun_visibility(
            sun_cache_pixel,
            sun_cache_dimensions,
            sun_cache,
            static_visibility,
            direction_to_light
          );
          sun_cache_stored = true;
          visibility = static_visibility;
          if (globals.environment_params.z > 0.0
              && max(visibility.r, max(visibility.g, visibility.b)) > 0.0) {
            visibility *= shadow_transmittance_filtered(
              offset_hit_position(hit, direction_to_light),
              direction_to_light,
              maximum_distance - RAY_ORIGIN_BIAS,
              SHADOW_FILTER_ANIMATED,
              local_invocation_index
            );
          }
        } else {
          visibility = shadow_transmittance(
            offset_hit_position(hit, direction_to_light),
            direction_to_light,
            maximum_distance - RAY_ORIGIN_BIAS,
            local_invocation_index
          );
        }
        visibility *= cloud_light_transmittance(
          offset_hit_position(hit, direction_to_light),
          direction_to_light,
          maximum_distance - RAY_ORIGIN_BIAS
        );
      }
      var incident_radiance = max(light.color, vec3<f32>(0.0))
        * max(light.intensity, 0.0);
      var evaluated_surface_lobes = direct_lobe_controls;
      if (
        light_index == directional_index
        && max(captured_sun_irradiance.r, max(captured_sun_irradiance.g, captured_sun_irradiance.b)) > 0.0
      ) {
        incident_radiance = captured_sun_irradiance
          * max(light.color, vec3<f32>(0.0))
          * max(light.intensity, 0.0);
        let current_surface_sun_preservation = select(
          solar_disc_preservation_for_roughness(surface_transport_roughness(hit)),
          1.0,
          current_surface_total_internal_reflection
        );
        let reflected_sun_preservation = min(
          clamp(path_solar_disc_preservation, 0.0, 1.0),
          current_surface_sun_preservation
        );
        evaluated_surface_lobes.y *= 1.0 - reflected_sun_preservation;
      }
      let light_radiance = sanitize_radiance(incident_radiance * attenuation * visibility);
      result += evaluate_light(
        hit,
        incoming_direction,
        direction_to_light,
        light_radiance,
        evaluated_surface_lobes
      );
    }
  }
  if (globals.output_mode.z > 0.5) {
    result += diffuse_environment_direct_lighting(
      hit,
      incoming_direction,
      state,
      direct_lobe_controls.x,
      local_invocation_index
    );
  }
  if (sun_cache.enabled != 0u && !sun_cache_stored) {
    clear_current_sun_visibility(sun_cache_pixel, sun_cache_dimensions, true);
  }
  return sanitize_radiance(result);
}

fn sampled_direct_lighting(
  hit: Hit,
  incoming_direction: vec3<f32>,
  state: ptr<function, u32>,
  direct_lobe_controls: vec2<f32>,
  path_solar_disc_preservation: f32,
  current_surface_total_internal_reflection: bool,
  sun_cache: SunVisibilityCache,
  sun_cache_pixel: vec2<u32>,
  sun_cache_dimensions: vec2<u32>,
  local_invocation_index: u32
) -> vec3<f32> {
  var requested_samples = globals.path_controls.w;
  if (requested_samples != requested_samples) {
    requested_samples = 1.0;
  }
  let sample_count = u32(clamp(requested_samples, 1.0, 8.0));
  var result = vec3<f32>(0.0);
  let captured_sun_irradiance = captured_environment_sun_irradiance();
  for (var sample_index = 0u; sample_index < 8u; sample_index += 1u) {
    if (sample_index >= sample_count) {
      break;
    }
    result += direct_lighting(
      hit,
      incoming_direction,
      state,
      direct_lobe_controls,
      captured_sun_irradiance,
      path_solar_disc_preservation,
      current_surface_total_internal_reflection,
      sun_cache,
      sun_cache_pixel,
      sun_cache_dimensions,
      local_invocation_index
    );
  }
  return sanitize_radiance(result / f32(sample_count));
}

fn trace_ray(
  primary_ray: Ray,
  primary_hit: Hit,
  initial_state: u32,
  primary_sun_cache: SunVisibilityCache,
  sun_cache_pixel: vec2<u32>,
  sun_cache_dimensions: vec2<u32>,
  local_invocation_index: u32
) -> vec3<f32> {
  var state = initial_state;
  var ray = primary_ray;
  var throughput = vec3<f32>(1.0);
  var radiance = vec3<f32>(0.0);
  var path_solar_disc_preservation = 1.0;
  var cached_continuation_hit = miss();
  var cached_continuation_hit_valid = false;
  let bounce_limit = clamp(u32(globals.render_params.w), 1u, MAX_BOUNCES);

  for (var bounce = 0u; bounce < MAX_BOUNCES; bounce += 1u) {
    if (bounce >= bounce_limit) {
      break;
    }
    var hit = primary_hit;
    if (bounce > 0u) {
      if (cached_continuation_hit_valid) {
        hit = cached_continuation_hit;
        cached_continuation_hit_valid = false;
      } else {
        hit = intersect_scene(ray, local_invocation_index);
      }
    }
    let cloud_segment_distance = select(1e30, hit.t, hit.hit != 0u);
    let cloud_segment = integrate_cloud(
      ray,
      cloud_segment_distance,
      initial_state ^ ((bounce + 1u) * 0x9e3779b9u)
    );
    radiance += throughput * cloud_segment.xyz;
    throughput *= cloud_segment.w;
    if (max(throughput.x, max(throughput.y, throughput.z)) < 0.0001) {
      break;
    }
    if (hit.hit == 0u) {
      if (bounce == 0u && primary_sun_cache.enabled != 0u) {
        clear_current_sun_visibility(sun_cache_pixel, sun_cache_dimensions, true);
      }
      radiance += throughput * camera_visible_environment_radiance(
        ray,
        path_solar_disc_preservation
      );
      break;
    }

    let outgoing_direction = safe_normalize(-ray.direction, hit.normal);
    let current_surface_total_internal_reflection = surface_has_total_internal_reflection(
      hit,
      ray.direction
    );
    let physical_lobe_probabilities = physical_path_lobe_probabilities(
      hit,
      outgoing_direction,
      current_surface_total_internal_reflection
    );
    var lobe_probabilities = normalized_ray_lobes(physical_lobe_probabilities);
    if (current_surface_total_internal_reflection) {
      let ray_family_weights = terminal_ray_family_weights();
      lobe_probabilities = select(
        vec3<f32>(0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        max(ray_family_weights.y, ray_family_weights.z) > 0.0
      );
    }
    let direct_lobe_controls = direct_surface_lobe_controls(
      physical_lobe_probabilities,
      current_surface_total_internal_reflection
    );

    radiance += throughput * hit.emission;
    if (hit_is_unlit(hit)) {
      if (bounce == 0u && primary_sun_cache.enabled != 0u) {
        clear_current_sun_visibility(sun_cache_pixel, sun_cache_dimensions, true);
      }
      break;
    }
    if (
      ray.background_visibility != 0u
      && globals.backplate_texture_params.x > 0.5
      && hit_is_thin_glass(hit)
    ) {
      if (bounce == 0u && primary_sun_cache.enabled != 0u) {
        clear_current_sun_visibility(sun_cache_pixel, sun_cache_dimensions, true);
      }
      let ray_family_weights = terminal_ray_family_weights();
      let reflection_enabled = ray_family_enabled(ray_family_weights.y) > 0.0;
      let transmission_enabled = ray_family_enabled(ray_family_weights.z) > 0.0;
      var reflection_normal = hit.normal;
      var reflected_direction = safe_normalize(
        reflect(ray.direction, reflection_normal),
        hit.geometric_normal
      );
      if (!valid_geometric_reflection(hit, outgoing_direction, reflected_direction)) {
        reflection_normal = hit.geometric_normal;
        reflected_direction = safe_normalize(
          reflect(ray.direction, reflection_normal),
          hit.geometric_normal
        );
      }
      let reflection_coefficient = thin_sheet_fresnel(
        hit,
        ray.direction,
        reflection_normal
      );
      let transmission_coefficient = thin_glass_transmission_coefficient(
        hit,
        reflection_coefficient
      );
      let next_cone_width = ray_cone_width_at_distance(ray, hit.t);
      let transmitted_direction = ray.direction;
      let transmitted_ray = Ray(
        offset_hit_position(hit, transmitted_direction),
        transmitted_direction,
        next_cone_width,
        ray.background_visibility
      );
      let transmission_active = transmission_enabled
        && luminance(transmission_coefficient) > 0.00000001;
      var transmission_probe_hit = miss();
      if (transmission_active) {
        transmission_probe_hit = intersect_scene(transmitted_ray, local_invocation_index);
      }
      if (transmission_active && transmission_probe_hit.hit == 0u) {
        radiance += throughput * thin_glass_verified_backplate_transmission(
          transmitted_direction,
          transmission_coefficient
        );
        if (!reflection_enabled) {
          break;
        }
        if (bounce + 1u >= bounce_limit) {
          radiance += throughput * thin_glass_environment_reflection(
            hit,
            ray.direction,
            path_solar_disc_preservation
          );
          break;
        }
        throughput *= reflection_coefficient;
        ray = Ray(
          offset_hit_position(hit, reflected_direction),
          reflected_direction,
          next_cone_width,
          0u
        );
        path_solar_disc_preservation = min(
          path_solar_disc_preservation,
          solar_disc_preservation_for_roughness(surface_transport_roughness(hit))
        );
        if (max(throughput.x, max(throughput.y, throughput.z)) < 0.0001) {
          break;
        }
        continue;
      }
      if (bounce + 1u >= bounce_limit) {
        if (reflection_enabled) {
          radiance += throughput * thin_glass_environment_reflection(
            hit,
            ray.direction,
            path_solar_disc_preservation
          );
        }
        break;
      }
      let sheet_probabilities = thin_glass_lobe_probabilities(
        reflection_coefficient,
        transmission_coefficient,
        ray_family_weights
      );
      if (sheet_probabilities.x + sheet_probabilities.y <= 0.00000001) {
        break;
      }
      if (random(&state) < sheet_probabilities.y) {
        throughput *= transmission_coefficient
          / max(sheet_probabilities.y, 0.00000001);
        ray = transmitted_ray;
        cached_continuation_hit = transmission_probe_hit;
        cached_continuation_hit_valid = transmission_probe_hit.hit != 0u;
      } else {
        throughput *= reflection_coefficient
          / max(sheet_probabilities.x, 0.00000001);
        ray = Ray(
          offset_hit_position(hit, reflected_direction),
          reflected_direction,
          next_cone_width,
          0u
        );
        path_solar_disc_preservation = min(
          path_solar_disc_preservation,
          solar_disc_preservation_for_roughness(surface_transport_roughness(hit))
        );
      }
      if (max(throughput.x, max(throughput.y, throughput.z)) < 0.0001) {
        break;
      }
      continue;
    }
    if (
      ray.background_visibility != 0u
      && globals.backplate_texture_params.x > 0.5
      && backplate_transmission_preserves_visibility(hit)
      && !hit_is_thin_glass(hit)
      && !hit_is_solid_glass(hit)
    ) {
      if (bounce == 0u && primary_sun_cache.enabled != 0u) {
        clear_current_sun_visibility(sun_cache_pixel, sun_cache_dimensions, true);
      }
      radiance += throughput * terminal_ray_trace_environment(
        hit,
        ray.direction,
        terminal_ray_family_weights(),
        path_solar_disc_preservation,
        ray.background_visibility
      );
      break;
    }
    var bounce_sun_cache = disabled_sun_visibility_cache(false);
    if (bounce == 0u) {
      bounce_sun_cache = primary_sun_cache;
    }
    radiance += throughput * sampled_direct_lighting(
      hit,
      ray.direction,
      &state,
      direct_lobe_controls,
      path_solar_disc_preservation,
      current_surface_total_internal_reflection,
      bounce_sun_cache,
      sun_cache_pixel,
      sun_cache_dimensions,
      local_invocation_index
    );
    if (bounce + 1u >= bounce_limit) {
      let terminal_weights = terminal_ray_family_weights();
      radiance += throughput * terminal_ray_trace_environment(
        hit,
        ray.direction,
        terminal_weights,
        path_solar_disc_preservation,
        ray.background_visibility
      );
      break;
    }

    if (lobe_probabilities.x + lobe_probabilities.y + lobe_probabilities.z <= 0.00000001) {
      break;
    }
    let reflection_probability = lobe_probabilities.y;
    let transmission_probability = lobe_probabilities.z;
    let choose = random(&state);
    let next_cone_width = ray_cone_width_at_distance(ray, hit.t);
    if (current_surface_total_internal_reflection) {
      var reflected_direction = safe_normalize(
        reflect(ray.direction, hit.normal),
        hit.geometric_normal
      );
      if (!valid_geometric_reflection(hit, outgoing_direction, reflected_direction)) {
        reflected_direction = safe_normalize(
          reflect(ray.direction, hit.geometric_normal),
          hit.geometric_normal
        );
      }
      ray = Ray(
        offset_hit_position(hit, reflected_direction),
        reflected_direction,
        next_cone_width,
        0u
      );
      throughput /= max(reflection_probability, 0.00000001);
    } else if (choose < transmission_probability) {
      if (hit_is_foliage(hit)) {
        let transmitted_direction = make_basis_direction(-hit.normal, &state);
        throughput *= foliage_diffuse_transmission_scatter_weight(
          hit,
          outgoing_direction,
          transmitted_direction,
          transmission_probability
        );
        ray = Ray(
          offset_hit_position(hit, transmitted_direction),
          transmitted_direction,
          next_cone_width,
          0u
        );
        path_solar_disc_preservation = 0.0;
      } else {
        let eta = select(hit.ior, 1.0 / hit.ior, hit.front_face != 0u);
        let refracted = refract(ray.direction, hit.normal, eta);
        let refracted_length_squared = dot(refracted, refracted);
        if (refracted_length_squared <= 0.000001) {
          break;
        }
        let refracted_direction = refracted * inverseSqrt(refracted_length_squared);
        throughput *= transmission_scatter_weight(
          hit,
          outgoing_direction,
          refracted_direction,
          eta,
          transmission_probability
        );
        ray = Ray(
          offset_hit_position(hit, refracted_direction),
          refracted_direction,
          next_cone_width,
          select(
            0u,
            ray.background_visibility,
            backplate_transmission_preserves_visibility(hit)
          )
        );
      }
    } else if (choose < transmission_probability + reflection_probability) {
      let microfacet_normal = sample_ggx_vndf(
        hit.normal,
        outgoing_direction,
        surface_transport_roughness(hit),
        &state
      );
      let reflected_direction = safe_normalize(
        reflect(ray.direction, microfacet_normal),
        hit.normal
      );
      throughput *= glossy_scatter_weight(
        hit,
        outgoing_direction,
        reflected_direction,
        microfacet_normal,
        reflection_probability
      );
      ray = Ray(
        offset_hit_position(hit, reflected_direction),
        reflected_direction,
        next_cone_width,
        0u
      );
      path_solar_disc_preservation = min(
        path_solar_disc_preservation,
        solar_disc_preservation_for_roughness(surface_transport_roughness(hit))
      );
    } else {
      let diffuse_direction = make_basis_direction(hit.normal, &state);
      throughput *= diffuse_scatter_weight(
        hit,
        outgoing_direction,
        diffuse_direction,
        lobe_probabilities.x
      );
      ray = Ray(
        offset_hit_position(hit, diffuse_direction),
        diffuse_direction,
        next_cone_width,
        0u
      );
      path_solar_disc_preservation = 0.0;
    }
    if (max(throughput.x, max(throughput.y, throughput.z)) < 0.0001) {
      break;
    }
  }
  return sanitize_radiance(radiance);
}

fn trace_path(
  primary_ray: Ray,
  primary_hit: Hit,
  initial_state: u32,
  local_invocation_index: u32
) -> vec3<f32> {
  var state = initial_state;
  var ray = primary_ray;
  var throughput = vec3<f32>(1.0);
  var radiance = vec3<f32>(0.0);
  var path_solar_disc_preservation = 1.0;
  var previous_scatter_was_diffuse = false;
  var cached_continuation_hit = miss();
  var cached_continuation_hit_valid = false;
  let bounce_limit = clamp(u32(globals.render_params.w), 1u, MAX_BOUNCES);
  var requested_min_bounces = globals.path_controls.x;
  if (requested_min_bounces != requested_min_bounces) {
    requested_min_bounces = 2.0;
  }
  let minimum_bounces = u32(clamp(requested_min_bounces, 1.0, f32(MAX_BOUNCES)));

  for (var bounce = 0u; bounce < MAX_BOUNCES; bounce += 1u) {
    if (bounce >= bounce_limit) {
      break;
    }
    var hit = primary_hit;
    if (bounce > 0u) {
      if (cached_continuation_hit_valid) {
        hit = cached_continuation_hit;
        cached_continuation_hit_valid = false;
      } else {
        hit = intersect_scene(ray, local_invocation_index);
      }
    }
    let cloud_segment_distance = select(1e30, hit.t, hit.hit != 0u);
    let cloud_segment = integrate_cloud(
      ray,
      cloud_segment_distance,
      initial_state ^ ((bounce + 1u) * 0x85ebca6bu)
    );
    radiance += throughput * cloud_segment.xyz;
    throughput *= cloud_segment.w;
    if (max(throughput.x, max(throughput.y, throughput.z)) < 0.0001) {
      break;
    }
    if (hit.hit == 0u) {
      if (!previous_scatter_was_diffuse) {
        radiance += throughput * camera_visible_environment_radiance(
          ray,
          path_solar_disc_preservation
        );
      }
      break;
    }
    if (hit_is_unlit(hit)) {
      if (!previous_scatter_was_diffuse) {
        radiance += throughput * hit.emission;
      }
      break;
    }
    radiance += throughput * hit.emission;
    if (
      ray.background_visibility != 0u
      && globals.backplate_texture_params.x > 0.5
      && hit_is_thin_glass(hit)
    ) {
      let ray_family_weights = terminal_ray_family_weights();
      let reflection_enabled = ray_family_enabled(ray_family_weights.y) > 0.0;
      let transmission_enabled = ray_family_enabled(ray_family_weights.z) > 0.0;
      let outgoing_direction = safe_normalize(-ray.direction, hit.normal);
      var reflection_normal = hit.normal;
      var reflected_direction = safe_normalize(
        reflect(ray.direction, reflection_normal),
        hit.geometric_normal
      );
      if (!valid_geometric_reflection(hit, outgoing_direction, reflected_direction)) {
        reflection_normal = hit.geometric_normal;
        reflected_direction = safe_normalize(
          reflect(ray.direction, reflection_normal),
          hit.geometric_normal
        );
      }
      let reflection_coefficient = thin_sheet_fresnel(
        hit,
        ray.direction,
        reflection_normal
      );
      let transmission_coefficient = thin_glass_transmission_coefficient(
        hit,
        reflection_coefficient
      );
      let next_cone_width = ray_cone_width_at_distance(ray, hit.t);
      let transmitted_direction = ray.direction;
      let transmitted_ray = Ray(
        offset_hit_position(hit, transmitted_direction),
        transmitted_direction,
        next_cone_width,
        ray.background_visibility
      );
      let transmission_active = transmission_enabled
        && luminance(transmission_coefficient) > 0.00000001;
      var transmission_probe_hit = miss();
      if (transmission_active) {
        transmission_probe_hit = intersect_scene(transmitted_ray, local_invocation_index);
      }
      if (transmission_active && transmission_probe_hit.hit == 0u) {
        radiance += throughput * thin_glass_verified_backplate_transmission(
          transmitted_direction,
          transmission_coefficient
        );
        if (!reflection_enabled) {
          break;
        }
        if (bounce + 1u >= bounce_limit) {
          radiance += throughput * thin_glass_environment_reflection(
            hit,
            ray.direction,
            path_solar_disc_preservation
          );
          break;
        }
        throughput *= reflection_coefficient;
        ray.direction = reflected_direction;
        ray.origin = offset_hit_position(hit, reflected_direction);
        ray.cone_width = next_cone_width;
        ray.background_visibility = 0u;
        path_solar_disc_preservation = min(
        path_solar_disc_preservation,
        solar_disc_preservation_for_roughness(hit.roughness)
        );
        if (max(throughput.x, max(throughput.y, throughput.z)) < 0.0001) {
          break;
        }
        continue;
      }
      if (bounce + 1u >= bounce_limit) {
        if (reflection_enabled) {
          radiance += throughput * thin_glass_environment_reflection(
            hit,
            ray.direction,
            path_solar_disc_preservation
          );
        }
        break;
      }
      let sheet_probabilities = thin_glass_lobe_probabilities(
        reflection_coefficient,
        transmission_coefficient,
        ray_family_weights
      );
      if (sheet_probabilities.x + sheet_probabilities.y <= 0.00000001) {
        break;
      }
      if (random(&state) < sheet_probabilities.y) {
        throughput *= transmission_coefficient
          / max(sheet_probabilities.y, 0.00000001);
        ray = transmitted_ray;
        cached_continuation_hit = transmission_probe_hit;
        cached_continuation_hit_valid = transmission_probe_hit.hit != 0u;
      } else {
        throughput *= reflection_coefficient
          / max(sheet_probabilities.x, 0.00000001);
        ray.direction = reflected_direction;
        ray.origin = offset_hit_position(hit, reflected_direction);
        ray.cone_width = next_cone_width;
        ray.background_visibility = 0u;
        path_solar_disc_preservation = min(
        path_solar_disc_preservation,
        solar_disc_preservation_for_roughness(hit.roughness)
        );
      }
      if (max(throughput.x, max(throughput.y, throughput.z)) < 0.0001) {
        break;
      }
      continue;
    }
    if (
      ray.background_visibility != 0u
      && globals.backplate_texture_params.x > 0.5
      && backplate_transmission_preserves_visibility(hit)
      && !hit_is_thin_glass(hit)
      && !hit_is_solid_glass(hit)
    ) {
      radiance += throughput * terminal_ray_trace_environment(
        hit,
        ray.direction,
        terminal_ray_family_weights(),
        path_solar_disc_preservation,
        ray.background_visibility
      );
      break;
    }
    let outgoing_direction = safe_normalize(-ray.direction, hit.normal);
    let current_surface_total_internal_reflection = surface_has_total_internal_reflection(
      hit,
      ray.direction
    );
    var physical_lobe_probabilities = physical_path_lobe_probabilities(
      hit,
      outgoing_direction,
      current_surface_total_internal_reflection
    );
    var lobe_probabilities = normalized_ray_lobes(physical_lobe_probabilities);
    if (current_surface_total_internal_reflection) {
      let ray_family_weights = terminal_ray_family_weights();
      let reflected_family_weight = max(ray_family_weights.y, ray_family_weights.z);
      physical_lobe_probabilities = vec3<f32>(0.0, 1.0, 0.0);
      lobe_probabilities = select(
        vec3<f32>(0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        reflected_family_weight > 0.0
      );
    }
    if (lobe_probabilities.x + lobe_probabilities.y + lobe_probabilities.z <= 0.00000001) {
      break;
    }
    let direct_lobe_controls = direct_surface_lobe_controls(
      physical_lobe_probabilities,
      current_surface_total_internal_reflection
    );
    radiance += throughput * sampled_direct_lighting(
      hit,
      ray.direction,
      &state,
      direct_lobe_controls,
      path_solar_disc_preservation,
      current_surface_total_internal_reflection,
      disabled_sun_visibility_cache(false),
      vec2<u32>(0u),
      vec2<u32>(1u),
      local_invocation_index
    );
    let reflection_probability = lobe_probabilities.y;
    let transmission_probability = lobe_probabilities.z;
    let choose = random(&state);
    ray.cone_width = ray_cone_width_at_distance(ray, hit.t);
    let foliage_transmission_selected = !current_surface_total_internal_reflection
      && choose < transmission_probability
      && hit_is_foliage(hit);
    if (foliage_transmission_selected) {
      path_solar_disc_preservation = 0.0;
      previous_scatter_was_diffuse = true;
    }
    if (current_surface_total_internal_reflection) {
      var reflected_direction = safe_normalize(
        reflect(ray.direction, hit.normal),
        hit.geometric_normal
      );
      if (!valid_geometric_reflection(hit, outgoing_direction, reflected_direction)) {
        reflected_direction = safe_normalize(
          reflect(ray.direction, hit.geometric_normal),
          hit.geometric_normal
        );
      }
      ray.direction = reflected_direction;
      ray.origin = offset_hit_position(hit, reflected_direction);
      ray.background_visibility = 0u;
      throughput /= max(reflection_probability, 0.00000001);
      previous_scatter_was_diffuse = false;
    } else if (choose < transmission_probability) {
      if (hit_is_foliage(hit)) {
        let transmitted_direction = make_basis_direction(-hit.normal, &state);
        throughput *= foliage_diffuse_transmission_scatter_weight(
          hit,
          outgoing_direction,
          transmitted_direction,
          transmission_probability
        );
        ray.direction = transmitted_direction;
        ray.origin = offset_hit_position(hit, transmitted_direction);
        ray.background_visibility = 0u;
      } else {
        let eta = select(hit.ior, 1.0 / hit.ior, hit.front_face != 0u);
        let refracted = refract(ray.direction, hit.normal, eta);
        let refracted_length_squared = dot(refracted, refracted);
        if (refracted_length_squared > 0.000001) {
          let refracted_direction = refracted * inverseSqrt(refracted_length_squared);
          throughput *= transmission_scatter_weight(
            hit,
            outgoing_direction,
            refracted_direction,
            eta,
            transmission_probability
          );
          ray.direction = refracted_direction;
          ray.origin = offset_hit_position(hit, refracted_direction);
          ray.background_visibility = select(
            0u,
            ray.background_visibility,
            backplate_transmission_preserves_visibility(hit)
          );
        } else {
          let reflected_direction = safe_normalize(
            reflect(ray.direction, hit.geometric_normal),
            hit.geometric_normal
          );
          ray.direction = reflected_direction;
          ray.origin = offset_hit_position(hit, reflected_direction);
          ray.background_visibility = 0u;
          throughput /= max(transmission_probability, 0.00000001);
        }
        previous_scatter_was_diffuse = false;
      }
    } else if (choose < transmission_probability + reflection_probability) {
      let microfacet_normal = sample_ggx_vndf(
        hit.normal,
        outgoing_direction,
        surface_transport_roughness(hit),
        &state
      );
      let reflected_direction = safe_normalize(
        reflect(ray.direction, microfacet_normal),
        hit.normal
      );
      throughput *= glossy_scatter_weight(
        hit,
        outgoing_direction,
        reflected_direction,
        microfacet_normal,
        reflection_probability
      );
      ray.direction = reflected_direction;
      ray.origin = offset_hit_position(hit, reflected_direction);
      ray.background_visibility = 0u;
      path_solar_disc_preservation = min(
        path_solar_disc_preservation,
        solar_disc_preservation_for_roughness(surface_transport_roughness(hit))
      );
      previous_scatter_was_diffuse = false;
    } else {
      let diffuse_direction = make_basis_direction(hit.normal, &state);
      throughput *= diffuse_scatter_weight(
        hit,
        outgoing_direction,
        diffuse_direction,
        lobe_probabilities.x
      );
      ray.direction = diffuse_direction;
      ray.origin = offset_hit_position(hit, diffuse_direction);
      ray.background_visibility = 0u;
      path_solar_disc_preservation = 0.0;
      previous_scatter_was_diffuse = true;
    }

    if (bounce + 1u >= minimum_bounces) {
      let survival_probability = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.1, 0.95);
      if (random(&state) > survival_probability) {
        break;
      }
      throughput /= survival_probability;
    }
  }
  return sanitize_radiance(radiance);
}

fn henyey_greenstein(cosine: f32, anisotropy: f32) -> f32 {
  let g = clamp(anisotropy, -0.9, 0.9);
  let denominator = max(1.0 + g * g - 2.0 * g * cosine, 0.001);
  return (1.0 - g * g) / (4.0 * PI * denominator * sqrt(denominator));
}

fn cloud_axis_interval(origin: f32, direction: f32, minimum: f32, maximum: f32) -> vec2<f32> {
  if (abs(direction) < 0.000001) {
    if (origin < minimum || origin > maximum) {
      return vec2<f32>(1.0, -1.0);
    }
    return vec2<f32>(-1e30, 1e30);
  }
  let inverse_direction = 1.0 / direction;
  let first = (minimum - origin) * inverse_direction;
  let second = (maximum - origin) * inverse_direction;
  return vec2<f32>(min(first, second), max(first, second));
}

fn cloud_ray_interval(ray: Ray, maximum_distance: f32) -> vec2<f32> {
  let x_interval = cloud_axis_interval(
    ray.origin.x,
    ray.direction.x,
    globals.cloud_bounds_min_mode.x,
    globals.cloud_bounds_max_steps.x
  );
  let y_interval = cloud_axis_interval(
    ray.origin.y,
    ray.direction.y,
    globals.cloud_bounds_min_mode.y,
    globals.cloud_bounds_max_steps.y
  );
  let z_interval = cloud_axis_interval(
    ray.origin.z,
    ray.direction.z,
    globals.cloud_bounds_min_mode.z,
    globals.cloud_bounds_max_steps.z
  );
  let entry = max(max(x_interval.x, y_interval.x), max(z_interval.x, 0.0));
  let exit = min(min(x_interval.y, y_interval.y), min(z_interval.y, maximum_distance));
  return vec2<f32>(entry, exit);
}

fn cloud_density_at(position: vec3<f32>) -> f32 {
  let extent = globals.cloud_bounds_max_steps.xyz - globals.cloud_bounds_min_mode.xyz;
  if (min(extent.x, min(extent.y, extent.z)) <= 0.0) {
    return 0.0;
  }
  let uvw = (position - globals.cloud_bounds_min_mode.xyz) / extent;
  if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) {
    return 0.0;
  }
  return max(textureSampleLevel(
    cloud_density_texture,
    cloud_density_sampler,
    clamp(uvw, vec3<f32>(0.0), vec3<f32>(1.0)),
    0.0
  ).r, 0.0) * max(globals.cloud_density_anisotropy.x, 0.0);
}

fn cloud_sheet_density(position: vec3<f32>) -> f32 {
  let extent = globals.cloud_bounds_max_steps.xyz - globals.cloud_bounds_min_mode.xyz;
  if (min(extent.x, min(extent.y, extent.z)) <= 0.0) {
    return 0.0;
  }
  let horizontal = (position.xz - globals.cloud_bounds_min_mode.xz) / extent.xz;
  if (any(horizontal < vec2<f32>(0.0)) || any(horizontal > vec2<f32>(1.0))) {
    return 0.0;
  }
  let density_scale = max(globals.cloud_density_anisotropy.x, 0.0);
  var density = textureSampleLevel(
    cloud_density_texture,
    cloud_density_sampler,
    vec3<f32>(horizontal.x, 0.25, horizontal.y),
    0.0
  ).r;
  density = max(density, textureSampleLevel(
    cloud_density_texture,
    cloud_density_sampler,
    vec3<f32>(horizontal.x, 0.5, horizontal.y),
    0.0
  ).r);
  density = max(density, textureSampleLevel(
    cloud_density_texture,
    cloud_density_sampler,
    vec3<f32>(horizontal.x, 0.75, horizontal.y),
    0.0
  ).r);
  return max(density, 0.0) * density_scale;
}

fn cloud_sheet_transmittance(
  origin: vec3<f32>,
  direction: vec3<f32>,
  maximum_distance: f32
) -> f32 {
  if (abs(direction.y) < 0.000001) {
    return 1.0;
  }
  let bounds_minimum = globals.cloud_bounds_min_mode.xyz;
  let bounds_maximum = globals.cloud_bounds_max_steps.xyz;
  let plane_height = (bounds_minimum.y + bounds_maximum.y) * 0.5;
  let distance = (plane_height - origin.y) / direction.y;
  if (distance <= INTERSECTION_EPSILON || distance >= maximum_distance) {
    return 1.0;
  }
  let density = cloud_sheet_density(origin + direction * distance);
  let vertical_thickness = max(bounds_maximum.y - bounds_minimum.y, 0.0001);
  let path_length = min(
    vertical_thickness / max(abs(direction.y), 0.08),
    vertical_thickness * 8.0
  );
  let extinction = max(globals.cloud_scattering_albedo_extinction.w, 0.0);
  return exp(-density * extinction * path_length);
}

fn cloud_volume_transmittance(
  origin: vec3<f32>,
  direction: vec3<f32>,
  maximum_distance: f32,
  requested_steps: u32
) -> f32 {
  if (requested_steps == 0u) {
    return 1.0;
  }
  let ray = Ray(origin, direction, 0.0, 0u);
  let interval = cloud_ray_interval(ray, maximum_distance);
  if (interval.y <= interval.x) {
    return 1.0;
  }
  let step_count = min(requested_steps, MAX_CLOUD_LIGHT_STEPS);
  let step_length = (interval.y - interval.x) / f32(step_count);
  var optical_depth = 0.0;
  for (var step = 0u; step < MAX_CLOUD_LIGHT_STEPS; step += 1u) {
    if (step >= step_count) {
      break;
    }
    let distance = interval.x + (f32(step) + 0.5) * step_length;
    optical_depth += cloud_density_at(ray.origin + ray.direction * distance) * step_length;
  }
  return exp(-optical_depth * max(globals.cloud_scattering_albedo_extinction.w, 0.0));
}

fn cloud_light_transmittance(
  origin: vec3<f32>,
  direction: vec3<f32>,
  maximum_distance: f32
) -> f32 {
  if (
    globals.cloud_density_anisotropy.w <= 0.5
    || globals.cloud_density_anisotropy.z < 0.5
  ) {
    return 1.0;
  }
  if (globals.cloud_bounds_min_mode.w < 1.5) {
    return cloud_sheet_transmittance(origin, direction, maximum_distance);
  }
  return cloud_volume_transmittance(
    origin,
    direction,
    maximum_distance,
    min(u32(globals.cloud_density_anisotropy.z), MAX_CLOUD_LIGHT_STEPS)
  );
}

fn cloud_incident_radiance(position: vec3<f32>, view_direction: vec3<f32>) -> vec3<f32> {
  let directional_index = primary_directional_light_index();
  var sun_direction = default_sun_direction();
  var sun_radiance = vec3<f32>(5.4, 4.35, 3.15);
  if (directional_index != NO_LIGHT_INDEX) {
    let directional = lights[directional_index];
    sun_direction = normalize(-directional.vector);
    sun_radiance = max(directional.color, vec3<f32>(0.0)) * max(directional.intensity, 0.0);
    let captured_sun_irradiance = captured_environment_sun_irradiance();
    if (max(captured_sun_irradiance.r, max(captured_sun_irradiance.g, captured_sun_irradiance.b)) > 0.0) {
      sun_radiance = captured_sun_irradiance
        * max(directional.color, vec3<f32>(0.0))
        * max(directional.intensity, 0.0);
    }
  }
  let light_steps = min(u32(globals.cloud_density_anisotropy.z), MAX_CLOUD_LIGHT_STEPS);
  var sun_transmittance = 1.0;
  if (light_steps > 0u) {
    sun_transmittance = cloud_light_transmittance(
      position + sun_direction * RAY_ORIGIN_BIAS,
      sun_direction,
      1e29
    );
  }
  let phase = henyey_greenstein(
    dot(view_direction, sun_direction),
    globals.cloud_density_anisotropy.y
  );
  var ambient = diffuse_environment_irradiance(vec3<f32>(0.0, 1.0, 0.0)) * (0.08 / PI);
  if (globals.environment_texture_params.x <= 0.5) {
    ambient = environment(vec3<f32>(0.0, 1.0, 0.0)) * 0.08;
  }
  return sanitize_radiance(
    max(globals.cloud_scattering_albedo_extinction.xyz, vec3<f32>(0.0))
      * (ambient + sun_radiance * phase * sun_transmittance)
  );
}

fn integrate_cloud_sheet(ray: Ray, maximum_distance: f32) -> vec4<f32> {
  if (abs(ray.direction.y) < 0.000001) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let bounds_minimum = globals.cloud_bounds_min_mode.xyz;
  let bounds_maximum = globals.cloud_bounds_max_steps.xyz;
  let plane_height = (bounds_minimum.y + bounds_maximum.y) * 0.5;
  let distance = (plane_height - ray.origin.y) / ray.direction.y;
  if (distance <= INTERSECTION_EPSILON || distance >= maximum_distance) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let position = ray.origin + ray.direction * distance;
  let density = cloud_sheet_density(position);
  if (density <= 0.0) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let vertical_thickness = max(bounds_maximum.y - bounds_minimum.y, 0.0001);
  let path_length = min(
    vertical_thickness / max(abs(ray.direction.y), 0.08),
    vertical_thickness * 8.0
  );
  let step_transmittance = exp(
    -density * max(globals.cloud_scattering_albedo_extinction.w, 0.0) * path_length
  );
  let source = cloud_incident_radiance(position, ray.direction);
  return vec4<f32>(sanitize_radiance(source * (1.0 - step_transmittance)), step_transmittance);
}

fn integrate_cloud_volume(
  ray: Ray,
  maximum_distance: f32,
  initial_state: u32
) -> vec4<f32> {
  let interval = cloud_ray_interval(ray, maximum_distance);
  if (interval.y <= interval.x) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let requested_steps = clamp(u32(globals.cloud_bounds_max_steps.w), 1u, MAX_CLOUD_STEPS);
  let step_length = (interval.y - interval.x) / f32(requested_steps);
  var state = initial_state;
  let jitter = random(&state);
  var transmittance = 1.0;
  var scattering = vec3<f32>(0.0);
  let extinction = max(globals.cloud_scattering_albedo_extinction.w, 0.0);
  for (var step = 0u; step < MAX_CLOUD_STEPS; step += 1u) {
    if (step >= requested_steps || transmittance < 0.002) {
      break;
    }
    let distance = interval.x + (f32(step) + jitter) * step_length;
    let position = ray.origin + ray.direction * distance;
    let density = cloud_density_at(position);
    if (density <= 0.0) {
      continue;
    }
    let step_transmittance = exp(-density * extinction * step_length);
    let source = cloud_incident_radiance(position, ray.direction);
    scattering += transmittance * source * (1.0 - step_transmittance);
    transmittance *= step_transmittance;
  }
  return vec4<f32>(sanitize_radiance(scattering), transmittance);
}

fn integrate_cloud(
  ray: Ray,
  maximum_distance: f32,
  initial_state: u32
) -> vec4<f32> {
  if (
    globals.cloud_density_anisotropy.w <= 0.5
    || globals.cloud_bounds_min_mode.w <= 0.5
    || globals.cloud_scattering_albedo_extinction.w <= 0.0
    || globals.cloud_density_anisotropy.x <= 0.0
  ) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  if (globals.cloud_bounds_min_mode.w < 1.5) {
    return integrate_cloud_sheet(ray, maximum_distance);
  }
  return integrate_cloud_volume(ray, maximum_distance, initial_state);
}

fn integrate_volume(
  ray: Ray,
  maximum_distance: f32,
  initial_state: u32,
  local_invocation_index: u32
) -> vec4<f32> {
  let base_density = max(globals.camera_right_fog_density.w, 0.0);
  let requested_step_count = min(u32(globals.light_volume.y), MAX_VOLUME_STEPS);
  let volume_weight = sanitized_ray_weight(globals.ray_type_weights.w);
  if (base_density <= 0.0 || requested_step_count == 0u || volume_weight <= 0.0) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  var state = initial_state;
  let capped_distance = min(maximum_distance, max(globals.light_volume.z, 1.0));
  // The configured step count remains the hard quality ceiling. In clear air,
  // the upper-bound optical depth is tiny and a uniformly jittered estimate
  // converges long before 64 shadow queries. Dense media still consume the
  // full requested march, preserving shafts and visibility transitions.
  let optical_depth_upper_bound = base_density * volume_weight * capped_distance;
  let optical_depth_steps = u32(ceil(clamp(
    optical_depth_upper_bound * f32(MAX_VOLUME_STEPS),
    1.0,
    f32(MAX_VOLUME_STEPS)
  )));
  let adaptive_step_count = max(MIN_ADAPTIVE_VOLUME_STEPS, optical_depth_steps);
  let step_count = min(requested_step_count, adaptive_step_count);
  let step_length = capped_distance / f32(step_count);
  let jitter = random(&state);
  var transmittance = 1.0;
  var scattering = vec3<f32>(0.0);
  let directional_index = primary_directional_light_index();
  var sun_direction = default_sun_direction();
  var sun_radiance = vec3<f32>(5.4, 4.35, 3.15);
  if (directional_index != NO_LIGHT_INDEX) {
    let directional = lights[directional_index];
    sun_direction = normalize(-directional.vector);
    sun_radiance = max(directional.color, vec3<f32>(0.0)) * max(directional.intensity, 0.0);
    let captured_sun_irradiance = captured_environment_sun_irradiance();
    if (max(captured_sun_irradiance.r, max(captured_sun_irradiance.g, captured_sun_irradiance.b)) > 0.0) {
      sun_radiance = captured_sun_irradiance
        * max(directional.color, vec3<f32>(0.0))
        * max(directional.intensity, 0.0);
    }
  }
  let sun_phase = henyey_greenstein(dot(ray.direction, sun_direction), globals.fog_color_anisotropy.w);
  let light_count = min(min(u32(globals.light_volume.x), arrayLength(&lights)), MAX_LIGHTS);
  for (var step = 0u; step < MAX_VOLUME_STEPS; step += 1u) {
    if (step >= step_count || transmittance < 0.005) {
      break;
    }
    let distance = (f32(step) + jitter) * step_length;
    let position = ray.origin + ray.direction * distance;
    let density = base_density
      * volume_weight
      * exp(-max(position.y, 0.0) * max(globals.camera_up_fog_height.w, 0.0));
    let optical_depth = density * step_length;
    let step_transmittance = exp(-optical_depth);
    let sun_visibility = shadow_transmittance(
      position,
      sun_direction,
      1e29,
      local_invocation_index
    );
    var ambient = environment(vec3<f32>(0.0, 1.0, 0.0)) * 0.025;
    if (captured_sun_matching_active()) {
      ambient = diffuse_environment_irradiance(vec3<f32>(0.0, 1.0, 0.0)) * (0.025 / PI);
    }
    let sunlight = sun_radiance * sun_phase * sun_visibility;
    var scene_light_scattering = vec3<f32>(0.0);
    if (light_count > 0u) {
      let sampled_light_index = min(u32(random(&state) * f32(light_count)), light_count - 1u);
      let light = lights[sampled_light_index];
      var direction_to_light = normalize(-light.vector);
      var maximum_distance = 1e29;
      var attenuation = 1.0;
      if (light.kind == 0u) {
        let sampled_position = light.vector + random_unit_vector(&state) * max(light.radius, 0.0);
        let delta = sampled_position - position;
        let distance_squared = max(dot(delta, delta), 0.0001);
        maximum_distance = sqrt(distance_squared);
        direction_to_light = delta / maximum_distance;
        let range_value = light.range_or_angular_radius;
        let range_fade = select(
          1.0,
          clamp(1.0 - distance_squared / (range_value * range_value), 0.0, 1.0),
          range_value > 0.0
        );
        attenuation = range_fade * range_fade / distance_squared;
        let casts_shadow = (light.flags & 1u) != 0u;
        var light_visibility = vec3<f32>(1.0);
        if (casts_shadow) {
          light_visibility = shadow_transmittance(
            position + direction_to_light * RAY_ORIGIN_BIAS,
            direction_to_light,
            max(maximum_distance - RAY_ORIGIN_BIAS, INTERSECTION_EPSILON),
            local_invocation_index
          );
        }
        if (max(light_visibility.r, max(light_visibility.g, light_visibility.b)) > 0.0) {
          let phase = henyey_greenstein(
            dot(ray.direction, direction_to_light),
            globals.fog_color_anisotropy.w
          );
          scene_light_scattering = sanitize_radiance(
            max(light.color, vec3<f32>(0.0))
            * max(light.intensity, 0.0)
            * attenuation
            * phase
            * f32(light_count)
            * light_visibility
          );
        }
      }
    }
    let source = globals.fog_color_anisotropy.xyz * (ambient + sunlight + scene_light_scattering);
    scattering += transmittance * source * (1.0 - step_transmittance);
    transmittance *= step_transmittance;
  }
  return vec4<f32>(sanitize_radiance(scattering), transmittance);
}

fn concentric_lens_sample(sample: vec2<f32>) -> vec2<f32> {
  let offset = sample * 2.0 - vec2<f32>(1.0);
  if (offset.x == 0.0 && offset.y == 0.0) {
    return vec2<f32>(0.0);
  }
  var radius: f32;
  var theta: f32;
  if (abs(offset.x) > abs(offset.y)) {
    radius = offset.x;
    theta = (PI * 0.25) * (offset.y / offset.x);
  } else {
    radius = offset.y;
    theta = (PI * 0.5) - (PI * 0.25) * (offset.x / offset.y);
  }
  return radius * vec2<f32>(cos(theta), sin(theta));
}

fn camera_ray(
  pixel: vec2<u32>,
  jitter: vec2<f32>,
  lens_sample: vec2<f32>
) -> Ray {
  let resolution = globals.resolution_samples.xy;
  let screen = ((vec2<f32>(pixel) + vec2<f32>(0.5) + jitter) / resolution) * 2.0 - vec2<f32>(1.0);
  let aspect = resolution.x / resolution.y;
  let pinhole_direction = normalize(
    globals.camera_forward_exposure.xyz
    + globals.camera_right_fog_density.xyz * (screen.x * aspect * globals.camera_position_tan_fov.w)
    - globals.camera_up_fog_height.xyz * (screen.y * globals.camera_position_tan_fov.w)
  );
  let f_stop = clamp(globals.camera_lens_params.x, 1.0, 25.0);
  let focus_distance = max(globals.camera_lens_params.y, 0.0001);
  let focal_length = max(globals.camera_lens_params.z, 0.000001);
  let aperture_radius = focal_length / (2.0 * f_stop);
  let disk = concentric_lens_sample(lens_sample) * aperture_radius;
  let origin = globals.camera_position_tan_fov.xyz
    + globals.camera_right_fog_density.xyz * disk.x
    + globals.camera_up_fog_height.xyz * disk.y;
  let focal_plane_t = focus_distance / max(
    dot(pinhole_direction, globals.camera_forward_exposure.xyz),
    0.000001
  );
  let focal_point = globals.camera_position_tan_fov.xyz + pinhole_direction * focal_plane_t;
  let direction = normalize(focal_point - origin);
  return Ray(origin, direction, 0.0, 1u);
}

fn project_previous_uv(position: vec3<f32>) -> vec3<f32> {
  let relative = position - globals.previous_position_tan_fov.xyz;
  let view_depth = dot(relative, globals.previous_forward.xyz);
  let safe_depth = max(view_depth, 0.0001);
  let aspect = globals.output_mode.x / max(globals.output_mode.y, 1.0);
  let ndc = vec2<f32>(
    dot(relative, globals.previous_right.xyz) / (safe_depth * globals.previous_position_tan_fov.w * aspect),
    -dot(relative, globals.previous_up.xyz) / (safe_depth * globals.previous_position_tan_fov.w)
  );
  return vec3<f32>(ndc * 0.5 + vec2<f32>(0.5), view_depth);
}

fn project_previous_direction_uv(direction: vec3<f32>) -> vec3<f32> {
  let view_depth = dot(direction, globals.previous_forward.xyz);
  let safe_depth = max(view_depth, 0.0001);
  let aspect = globals.output_mode.x / max(globals.output_mode.y, 1.0);
  let ndc = vec2<f32>(
    dot(direction, globals.previous_right.xyz) / (safe_depth * globals.previous_position_tan_fov.w * aspect),
    -dot(direction, globals.previous_up.xyz) / (safe_depth * globals.previous_position_tan_fov.w)
  );
  return vec3<f32>(ndc * 0.5 + vec2<f32>(0.5), view_depth);
}

fn reconstruct_previous_surface_position(
  pixel: vec2<u32>,
  view_depth: f32,
  actual_dimensions: vec2<u32>
) -> vec3<f32> {
  let previous_jitter = vec2<f32>(globals.previous_forward.w, globals.previous_right.w);
  let screen = (
    (vec2<f32>(pixel) + vec2<f32>(0.5) + previous_jitter)
      / vec2<f32>(actual_dimensions)
  ) * 2.0 - vec2<f32>(1.0);
  let aspect = f32(actual_dimensions.x) / max(f32(actual_dimensions.y), 1.0);
  let previous_ray = globals.previous_forward.xyz
    + globals.previous_right.xyz * (screen.x * aspect * globals.previous_position_tan_fov.w)
    - globals.previous_up.xyz * (screen.y * globals.previous_position_tan_fov.w);
  return globals.previous_position_tan_fov.xyz + previous_ray * view_depth;
}

fn disabled_sun_visibility_cache(enabled: bool) -> SunVisibilityCache {
  return SunVisibilityCache(
    vec3<f32>(1.0),
    vec3<f32>(0.0, 1.0, 0.0),
    0u,
    select(0u, 1u, enabled),
    0u
  );
}

fn encode_octahedral_direction(direction: vec3<f32>) -> u32 {
  let normalized = safe_normalize(direction, vec3<f32>(0.0, 1.0, 0.0));
  var projected = normalized / max(abs(normalized.x) + abs(normalized.y) + abs(normalized.z), 0.000001);
  if (projected.z < 0.0) {
    let signs = select(vec2<f32>(-1.0), vec2<f32>(1.0), projected.xy >= vec2<f32>(0.0));
    projected = vec3<f32>((vec2<f32>(1.0) - abs(projected.yx)) * signs, projected.z);
  }
  return pack2x16snorm(clamp(projected.xy, vec2<f32>(-1.0), vec2<f32>(1.0)));
}

fn decode_octahedral_direction(encoded: u32) -> vec3<f32> {
  let projected = unpack2x16snorm(encoded);
  var direction = vec3<f32>(
    projected,
    1.0 - abs(projected.x) - abs(projected.y)
  );
  let fold = clamp(-direction.z, 0.0, 1.0);
  direction.x += select(fold, -fold, direction.x >= 0.0);
  direction.y += select(fold, -fold, direction.y >= 0.0);
  return safe_normalize(direction, vec3<f32>(0.0, 1.0, 0.0));
}

fn sun_visibility_cache_enabled(actual_dimensions: vec2<u32>) -> bool {
  let required_entries = actual_dimensions.x * actual_dimensions.y;
  return globals.output_mode.z <= 0.5
    && u32(clamp(globals.render_params.z, 1.0, f32(MAX_SAMPLES_PER_FRAME))) == 1u
    && u32(clamp(globals.render_params.w, 1.0, f32(MAX_BOUNCES))) == 1u
    && u32(clamp(globals.path_controls.w, 1.0, 8.0)) == 1u
    && arrayLength(&previous_sun_visibility) >= required_entries
    && arrayLength(&output_sun_visibility) >= required_entries
    && textureDimensions(previous_normal_depth).x == actual_dimensions.x
    && textureDimensions(previous_normal_depth).y == actual_dimensions.y;
}

fn sun_visibility_cache_index(pixel: vec2<u32>, actual_dimensions: vec2<u32>) -> u32 {
  return pixel.y * actual_dimensions.x + pixel.x;
}

fn sun_visibility_refresh_period() -> u32 {
  return SUN_VISIBILITY_STATIC_REFRESH_PERIOD;
}

fn clear_current_sun_visibility(
  pixel: vec2<u32>,
  actual_dimensions: vec2<u32>,
  enabled: bool
) {
  if (enabled) {
    output_sun_visibility[sun_visibility_cache_index(pixel, actual_dimensions)] = PackedSunVisibility(0u, 0u);
  }
}

fn reprojected_sun_visibility(
  hit: Hit,
  pixel: vec2<u32>,
  actual_dimensions: vec2<u32>,
  enabled: bool
) -> SunVisibilityCache {
  var result = disabled_sun_visibility_cache(enabled);
  if (!enabled
      || hit.hit == 0u
      || hit.reactive > 0.5
      || u32(globals.resolution_samples.w) == 0u) {
    return result;
  }

  let previous_projection = project_previous_uv(hit.position);
  if (previous_projection.z <= 0.0) {
    return result;
  }
  let previous_jitter = vec2<f32>(globals.previous_forward.w, globals.previous_right.w);
  let previous_raster = previous_projection.xy * vec2<f32>(actual_dimensions) - previous_jitter;
  if (any(previous_raster < vec2<f32>(0.0))
      || any(previous_raster >= vec2<f32>(actual_dimensions))) {
    return result;
  }
  let previous_pixel = vec2<u32>(floor(previous_raster));
  let previous_surface = textureLoad(previous_normal_depth, vec2<i32>(previous_pixel), 0);
  let previous_normal_length_squared = dot(previous_surface.xyz, previous_surface.xyz);
  let depth_tolerance = max(0.01, previous_projection.z * 0.004);
  if (previous_surface.w >= 65503.0
      || abs(previous_surface.w - previous_projection.z) > depth_tolerance
      || previous_normal_length_squared < 0.25
      || dot(
        safe_normalize(previous_surface.xyz, hit.normal),
        safe_normalize(hit.normal, previous_surface.xyz)
      ) < SUN_VISIBILITY_NORMAL_COSINE) {
    return result;
  }
  let previous_world_position = reconstruct_previous_surface_position(
    previous_pixel,
    previous_surface.w,
    actual_dimensions
  );
  let pixel_world_footprint = previous_projection.z
    * globals.previous_position_tan_fov.w
    * 2.0
    / max(f32(actual_dimensions.y), 1.0);
  let position_tolerance = max(
    0.01,
    pixel_world_footprint * 2.5 + previous_projection.z * 0.004
  );
  if (distance(previous_world_position, hit.position) > position_tolerance) {
    return result;
  }

  let packed = previous_sun_visibility[
    sun_visibility_cache_index(previous_pixel, actual_dimensions)
  ];
  let decoded = unpack4x8unorm(packed.visibility_age);
  let age_marker = u32(round(decoded.w * 255.0));
  if (age_marker == 0u) {
    return result;
  }
  let age = age_marker - 1u;
  let refresh_period = sun_visibility_refresh_period();
  let refresh_phase = hash_u32(pixel.x * 1973u + pixel.y * 9277u) % refresh_period;
  let frame_phase = u32(globals.resolution_samples.w) % refresh_period;
  if (age + 1u >= refresh_period || refresh_phase == frame_phase) {
    return result;
  }

  result.visibility = decoded.xyz;
  result.direction = decode_octahedral_direction(packed.direction);
  result.age = age;
  result.valid = 1u;
  return result;
}

fn store_sun_visibility(
  pixel: vec2<u32>,
  actual_dimensions: vec2<u32>,
  cache: SunVisibilityCache,
  visibility: vec3<f32>,
  direction: vec3<f32>
) {
  if (cache.enabled == 0u) {
    return;
  }
  let next_age = select(0u, min(cache.age + 1u, 254u), cache.valid != 0u);
  output_sun_visibility[sun_visibility_cache_index(pixel, actual_dimensions)] = PackedSunVisibility(
    pack4x8unorm(
      vec4<f32>(clamp(visibility, vec3<f32>(0.0), vec3<f32>(1.0)), f32(next_age + 1u) / 255.0)
    ),
    encode_octahedral_direction(direction)
  );
}

fn encoded_previous_depth(view_depth: f32, sky: bool) -> f32 {
  if (view_depth <= 0.0) {
    return -1.0;
  }
  return select(min(view_depth, 65504.0), 65504.0, sky);
}

fn bounded_noise_metric(value: f32, fallback: f32) -> f32 {
  if (value != value) {
    return fallback;
  }
  return clamp(value, 0.0, 1.0);
}

fn relative_radiance_change(current: vec3<f32>, previous: vec3<f32>) -> f32 {
  let safe_current = sanitize_radiance(current);
  let safe_previous = sanitize_radiance(previous);
  let luminance_weights = vec3<f32>(0.2126, 0.7152, 0.0722);
  let current_luminance = max(dot(safe_current, luminance_weights), 0.0);
  let previous_luminance = max(dot(safe_previous, luminance_weights), 0.0);
  let luminance_reference = max(max(current_luminance, previous_luminance), 0.0001);
  let luminance_change = abs(current_luminance - previous_luminance) / luminance_reference;
  let color_reference = max(length(max(safe_current, safe_previous)), 0.0001);
  let color_change = length(safe_current - safe_previous) / color_reference;
  return clamp(max(luminance_change, color_change), 0.0, 1.0);
}

fn store_gbuffer(
  pixel: vec2<u32>,
  actual_dimensions: vec2<u32>,
  ray: Ray,
  hit: Hit
) {
  let current_uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(actual_dimensions);
  if (hit.hit == 0u) {
    let previous_projection = project_previous_direction_uv(ray.direction);
    let motion = previous_projection.xy - current_uv;
    textureStore(normal_depth, vec2<i32>(pixel), vec4<f32>(0.0, 0.0, 0.0, 65504.0));
    textureStore(albedo_roughness, vec2<i32>(pixel), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    textureStore(
      motion_reactive,
      vec2<i32>(pixel),
      vec4<f32>(motion, 0.0, encoded_previous_depth(previous_projection.z, true))
    );
    return;
  }

  let previous_projection = project_previous_uv(hit.position);
  let motion = previous_projection.xy - current_uv;
  let reactive = clamp(
    max(
      max(max(hit.emission.r, hit.emission.g), hit.emission.b) * 0.05
        + hit.transmission,
      hit.reactive
    ),
    0.0,
    1.0
  );
  let current_view_depth = dot(
    hit.position - globals.camera_position_tan_fov.xyz,
    globals.camera_forward_exposure.xyz
  );
  textureStore(
    normal_depth,
    vec2<i32>(pixel),
    vec4<f32>(hit.normal, clamp(current_view_depth, 0.0, 65504.0))
  );
  var gbuffer_roughness = select(hit.roughness, 0.0, hit_is_unlit(hit));
  if (!hit_is_unlit(hit)
      && ray.background_visibility != 0u
      && globals.backplate_texture_params.x > 0.5
      && backplate_transmission_preserves_visibility(hit)) {
    // Stored in RGBA8 as 2/255: distinct from the zero-valued unlit marker,
    // while remaining inside the denoiser's deterministic-radiance bypass.
    gbuffer_roughness = BACKPLATE_GBUFFER_ROUGHNESS_MARKER;
  }
  textureStore(albedo_roughness, vec2<i32>(pixel), vec4<f32>(hit.albedo, gbuffer_roughness));
  textureStore(
    motion_reactive,
    vec2<i32>(pixel),
    vec4<f32>(motion, reactive, encoded_previous_depth(previous_projection.z, false))
  );
}

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_index) local_invocation_index: u32
) {
  let actual_dimensions = textureDimensions(output_frame);
  let dispatch_offset = vec2<u32>(max(globals.trace_dispatch_tile.xy, vec2<f32>(0.0)));
  let dispatch_extent = vec2<u32>(max(globals.trace_dispatch_tile.zw, vec2<f32>(0.0)));
  if (global_id.x >= dispatch_extent.x || global_id.y >= dispatch_extent.y) {
    return;
  }
  let pixel = global_id.xy + dispatch_offset;
  if (pixel.x >= actual_dimensions.x || pixel.y >= actual_dimensions.y) {
    return;
  }

  let frame_index = u32(globals.resolution_samples.w);
  let sample_limit = clamp(u32(globals.render_params.z), 1u, MAX_SAMPLES_PER_FRAME);
  let reuse_primary_gbuffer = globals.output_mode.z <= 0.5 && sample_limit == 1u;
  let sun_cache_enabled = sun_visibility_cache_enabled(actual_dimensions);
  let jitter_enabled = globals.output_mode.w > 0.5;
  let previous_samples = max(globals.resolution_samples.z, 0.0);
  let previous_texel = textureLoad(previous_frame, vec2<i32>(pixel), 0);
  let previous = sanitize_radiance(previous_texel.rgb);
  let previous_noise = bounded_noise_metric(previous_texel.a, 1.0);
  var convergence_threshold = globals.path_controls.y;
  if (convergence_threshold != convergence_threshold) {
    convergence_threshold = 0.0;
  }
  convergence_threshold = clamp(convergence_threshold, 0.0, 1.0);
  var requested_minimum_samples = globals.path_controls.z;
  if (requested_minimum_samples != requested_minimum_samples) {
    requested_minimum_samples = 1.0;
  }
  // Both normal/depth ping-pong targets need one traced frame before a pixel
  // can retain its prior G-buffer without another scene intersection.
  let minimum_convergence_samples = max(
    max(requested_minimum_samples, 1.0),
    f32(sample_limit) * 2.0
  );
  let pixel_converged = convergence_threshold > 0.0
    && previous_samples > 0.0
    && previous_samples >= minimum_convergence_samples
    && previous_noise / sqrt(max(previous_samples, 1.0)) <= convergence_threshold;
  if (pixel_converged && sun_cache_enabled) {
    let cache_index = sun_visibility_cache_index(pixel, actual_dimensions);
    output_sun_visibility[cache_index] = previous_sun_visibility[cache_index];
  }

  var accumulated = previous;
  var noise_metric = previous_noise;
  if (!pixel_converged) {
    var summed_radiance = vec3<f32>(0.0);
    for (var sample_index = 0u; sample_index < MAX_SAMPLES_PER_FRAME; sample_index += 1u) {
      if (sample_index >= sample_limit) {
        break;
      }
      var state = hash_u32(
        pixel.x * 1973u
        + pixel.y * 9277u
        + frame_index * 26699u
        + sample_index * 31847u
      );
      var jitter = vec2<f32>(0.0);
      if (jitter_enabled) {
        jitter = globals.render_params.xy;
        if (sample_limit > 1u) {
          jitter += vec2<f32>(random(&state) - 0.5, random(&state) - 0.5) * 0.7;
        }
      }
      let lens_sample = vec2<f32>(random(&state), random(&state));
      let ray = camera_ray(pixel, jitter, lens_sample);
      let primary_hit = intersect_scene(ray, local_invocation_index);
      let primary_sun_cache = reprojected_sun_visibility(
        primary_hit,
        pixel,
        actual_dimensions,
        sun_cache_enabled
      );
      if (reuse_primary_gbuffer) {
        store_gbuffer(pixel, actual_dimensions, ray, primary_hit);
      }
      var sample_radiance = vec3<f32>(0.0);
      if (globals.output_mode.z > 0.5) {
        sample_radiance = trace_path(ray, primary_hit, state, local_invocation_index);
      } else {
        sample_radiance = trace_ray(
          ray,
          primary_hit,
          state,
          primary_sun_cache,
          pixel,
          actual_dimensions,
          local_invocation_index
        );
      }
      let fog_distance = select(
        max(globals.light_volume.z, 1.0),
        min(primary_hit.t, max(globals.light_volume.z, 1.0)),
        primary_hit.hit != 0u
      );
      let volume = integrate_volume(
        ray,
        fog_distance,
        state ^ 0x68bc21ebu,
        local_invocation_index
      );
      sample_radiance = volume.xyz + sample_radiance * volume.w;
      summed_radiance += sanitize_radiance(sample_radiance);
    }

    let current_samples = f32(sample_limit);
    accumulated = summed_radiance / max(current_samples, 1.0);
    if (previous_samples > 0.0) {
      let batch_mean = accumulated;
      accumulated = (previous * previous_samples + summed_radiance)
        / max(previous_samples + current_samples, 1.0);
      // Running-mean changes shrink as 1/N even while the estimator remains
      // noisy. A batch residual estimates its coefficient of variation, and
      // sqrt(batch spp) keeps that estimate comparable across batch sizes.
      let instantaneous_noise = bounded_noise_metric(
        relative_radiance_change(batch_mean, previous) * sqrt(current_samples),
        1.0
      );
      noise_metric = bounded_noise_metric(mix(previous_noise, instantaneous_noise, 0.2), 1.0);
    } else {
      noise_metric = 1.0;
    }
  }
  textureStore(
    output_frame,
    vec2<i32>(pixel),
    vec4<f32>(sanitize_radiance(accumulated), bounded_noise_metric(noise_metric, 1.0))
  );

  // Scene/camera/wind edits reset the sample count, so converged pixels can
  // safely retain the already-populated G-buffer history in their ping-pong target.
  if (pixel_converged) {
    return;
  }
  if (reuse_primary_gbuffer) {
    return;
  }

  let center_ray = camera_ray(pixel, vec2<f32>(0.0), vec2<f32>(0.5));
  let center_hit = intersect_scene(center_ray, local_invocation_index);
  store_gbuffer(pixel, actual_dimensions, center_ray, center_hit);
}
`
  );
  var TRACE_MODE_CONSTANT_ANCHOR = "const PI: f32 = 3.141592653589793;";
  var RUNTIME_RAY_TRACE_MODE_TEST = "globals.output_mode.z <= 0.5";
  var RUNTIME_PATH_TRACE_MODE_TEST = "globals.output_mode.z > 0.5";
  function modeSpecializedPathTracerShader(pathTracingEnabled) {
    const compileTimeMode = `const PIPELINE_PATH_TRACING_ENABLED: bool = ${pathTracingEnabled};`;
    const withCompileTimeMode = ADVANCED_PATH_TRACER_SHADER.replace(
      TRACE_MODE_CONSTANT_ANCHOR,
      `${compileTimeMode}
${TRACE_MODE_CONSTANT_ANCHOR}`
    );
    if (withCompileTimeMode === ADVANCED_PATH_TRACER_SHADER) {
      throw new Error("The advanced tracing shader compile-time mode anchor is missing.");
    }
    let specialized = withCompileTimeMode.replaceAll(RUNTIME_RAY_TRACE_MODE_TEST, "!PIPELINE_PATH_TRACING_ENABLED").replaceAll(RUNTIME_PATH_TRACE_MODE_TEST, "PIPELINE_PATH_TRACING_ENABLED");
    if (specialized.includes("globals.output_mode.z")) {
      throw new Error("The advanced tracing shader contains an unspecialized render-mode branch.");
    }
    if (pathTracingEnabled) {
      const cinematicStackWords = CINEMATIC_TRACE_WORKGROUP_SIZE * CINEMATIC_TRACE_WORKGROUP_SIZE * BVH_TRAVERSAL_STACK_CAPACITY;
      const replacements = [
        [
          `@compute @workgroup_size(${RAY_TRACE_WORKGROUP_SIZE}, ${RAY_TRACE_WORKGROUP_SIZE}, 1)`,
          `@compute @workgroup_size(${CINEMATIC_TRACE_WORKGROUP_SIZE}, ${CINEMATIC_TRACE_WORKGROUP_SIZE}, 1)`
        ],
        [
          `var<workgroup> bvh_traversal_stack: array<u32, ${BVH_WORKGROUP_STACK_WORDS}>;`,
          `var<workgroup> bvh_traversal_stack: array<u32, ${cinematicStackWords}>;`
        ]
      ];
      for (const [anchor, replacement] of replacements) {
        if (specialized.split(anchor).length !== 2) {
          throw new Error("The cinematic tracing shader workgroup specialization anchor is missing or ambiguous.");
        }
        specialized = specialized.replace(anchor, replacement);
      }
    }
    return specialized;
  }
  var ADVANCED_RAY_TRACER_SHADER = modeSpecializedPathTracerShader(false);
  var ADVANCED_CINEMATIC_PATH_TRACER_SHADER = modeSpecializedPathTracerShader(true);

  // src/renderer/display-shaders.ts
  var AUTO_EXPOSURE_SHADER = (
    /* wgsl */
    `
struct ExposureParameters {
  resolution_delta: vec4<f32>,
  controls: vec4<f32>,
}

struct ExposureState {
  exposure: f32,
  average_luminance: f32,
  target_exposure: f32,
  padding: f32,
}

@group(0) @binding(0) var<uniform> parameters: ExposureParameters;
@group(0) @binding(1) var hdr_source: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> state: ExposureState;

var<workgroup> log_luminance_samples: array<f32, 256>;

fn luminance(color: vec3<f32>) -> f32 {
  return dot(max(color, vec3<f32>(0.0)), vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_index) local_index: u32) {
  let dimensions = textureDimensions(hdr_source);
  let grid = vec2<u32>(local_index & 15u, local_index >> 4u);
  let pixel = min(
    vec2<u32>(
      (grid.x * dimensions.x + dimensions.x / 2u) / 16u,
      (grid.y * dimensions.y + dimensions.y / 2u) / 16u
    ),
    dimensions - vec2<u32>(1u)
  );
  let measured = clamp(luminance(textureLoad(hdr_source, vec2<i32>(pixel), 0).rgb), 0.0001, 65504.0);
  log_luminance_samples[local_index] = log(measured);
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (local_index < stride) {
      log_luminance_samples[local_index] += log_luminance_samples[local_index + stride];
    }
    workgroupBarrier();
    if (stride == 1u) {
      break;
    }
    stride >>= 1u;
  }

  if (local_index == 0u) {
    let average_luminance = exp(log_luminance_samples[0] / 256.0);
    let target_value = clamp(parameters.controls.x / max(average_luminance, 0.0001), parameters.controls.z, parameters.controls.w);
    let previous = select(target_value, state.exposure, state.exposure > 0.0 && state.exposure <= 65504.0);
    let adaptation = 1.0 - exp(-max(parameters.resolution_delta.z, 0.0) * max(parameters.controls.y, 0.0));
    state.exposure = mix(previous, target_value, adaptation);
    state.average_luminance = average_luminance;
    state.target_exposure = target_value;
  }
}
`
  );
  var DISPLAY_POST_PROCESS_SHADER = (
    /* wgsl */
    `
struct DisplayPostParameters {
  sizes: vec4<f32>,
  controls: vec4<f32>,
  flags_balance_rg: vec4<f32>,
  bloom_balance_b: vec4<f32>,
  camera_latitude: vec4<f32>,
}

struct ExposureState {
  exposure: f32,
  average_luminance: f32,
  target_exposure: f32,
  padding: f32,
}

@group(0) @binding(0) var<uniform> parameters: DisplayPostParameters;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var hdr_source: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> exposure_state: ExposureState;
@group(0) @binding(4) var display_output: texture_storage_2d<rgba16float, write>;

const ACES_INPUT_MATRIX: mat3x3<f32> = mat3x3<f32>(
  vec3<f32>(0.59719, 0.07600, 0.02840),
  vec3<f32>(0.35458, 0.90834, 0.13383),
  vec3<f32>(0.04823, 0.01566, 0.83777)
);

const ACES_OUTPUT_MATRIX: mat3x3<f32> = mat3x3<f32>(
  vec3<f32>( 1.60475, -0.10208, -0.00327),
  vec3<f32>(-0.53108,  1.10813, -0.07276),
  vec3<f32>(-0.07367, -0.00605,  1.07602)
);

const LINEAR_SRGB_TO_LINEAR_DISPLAY_P3_MATRIX: mat3x3<f32> = mat3x3<f32>(
  vec3<f32>(0.82246197, 0.03319420, 0.01708263),
  vec3<f32>(0.17753803, 0.96680580, 0.07239744),
  vec3<f32>(0.00000000, 0.00000000, 0.91051993)
);

fn luminance(color: vec3<f32>) -> f32 {
  return dot(max(color, vec3<f32>(0.0)), vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn bloom_sample(uv: vec2<f32>) -> vec3<f32> {
  let color = max(textureSampleLevel(hdr_source, linear_sampler, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb, vec3<f32>(0.0));
  let brightness = luminance(color);
  let threshold = parameters.controls.w;
  let knee = max(parameters.bloom_balance_b.x, 0.0001);
  let soft = clamp((brightness - threshold + knee) / (2.0 * knee), 0.0, 1.0);
  let contribution = max(brightness - threshold, 0.0) + soft * soft * knee;
  return color * (contribution / max(brightness, 0.0001));
}

fn bloom_filter(uv: vec2<f32>) -> vec3<f32> {
  if (parameters.controls.z <= 0.0) {
    return vec3<f32>(0.0);
  }
  let texel = vec2<f32>(1.0) / max(parameters.sizes.xy, vec2<f32>(1.0));
  let radius = max(parameters.bloom_balance_b.y, 1.0);
  var result = bloom_sample(uv) * 0.20;
  result += (bloom_sample(uv + vec2<f32>( texel.x, 0.0) * radius)
    + bloom_sample(uv + vec2<f32>(-texel.x, 0.0) * radius)
    + bloom_sample(uv + vec2<f32>(0.0,  texel.y) * radius)
    + bloom_sample(uv + vec2<f32>(0.0, -texel.y) * radius)) * 0.10;
  result += (bloom_sample(uv + vec2<f32>( texel.x,  texel.y) * radius * 2.0)
    + bloom_sample(uv + vec2<f32>(-texel.x,  texel.y) * radius * 2.0)
    + bloom_sample(uv + vec2<f32>( texel.x, -texel.y) * radius * 2.0)
    + bloom_sample(uv + vec2<f32>(-texel.x, -texel.y) * radius * 2.0)) * 0.05;
  return result;
}

fn sanitize_linear_hdr(color: vec3<f32>) -> vec3<f32> {
  let clamped = clamp(color, vec3<f32>(0.0), vec3<f32>(65504.0));
  return select(vec3<f32>(0.0), clamped, color == color);
}

fn aces_rrt_odt_fit(color: vec3<f32>) -> vec3<f32> {
  let numerator = color * (color + vec3<f32>(0.0245786)) - vec3<f32>(0.000090537);
  let denominator = color * (vec3<f32>(0.983729) * color + vec3<f32>(0.4329510))
    + vec3<f32>(0.238081);
  return numerator / denominator;
}

fn aces_tone_map(linear_srgb: vec3<f32>) -> vec3<f32> {
  let aces_working = ACES_INPUT_MATRIX * linear_srgb;
  let fitted = aces_rrt_odt_fit(aces_working);
  let linear_srgb_output = ACES_OUTPUT_MATRIX * fitted;
  return clamp(linear_srgb_output, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_srgb_to_linear_display_p3(color: vec3<f32>) -> vec3<f32> {
  return LINEAR_SRGB_TO_LINEAR_DISPLAY_P3_MATRIX * color;
}

fn tone_map(color: vec3<f32>, mode: f32) -> vec3<f32> {
  if (mode < 0.5) {
    return aces_tone_map(color);
  }
  if (mode < 1.5) {
    return color / (vec3<f32>(1.0) + color);
  }
  return clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
  let low = color * 12.92;
  let high = 1.055 * pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
  return select(high, low, color <= vec3<f32>(0.0031308));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let output_dimensions = textureDimensions(display_output);
  if (any(global_id.xy >= output_dimensions)) {
    return;
  }
  let uv = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) / vec2<f32>(output_dimensions);
  var hdr = max(textureSampleLevel(hdr_source, linear_sampler, uv, 0.0).rgb, vec3<f32>(0.0));
  hdr += bloom_filter(uv) * parameters.controls.z;
  let white_balance = vec3<f32>(parameters.flags_balance_rg.zw, parameters.bloom_balance_b.z);
  hdr *= white_balance;
  let automatic = select(1.0, exposure_state.exposure, parameters.flags_balance_rg.x > 0.5);
  let exposed = sanitize_linear_hdr(hdr * parameters.controls.x * automatic);
  let mapped = tone_map(exposed, parameters.controls.y);
  let extended_display_p3 = sanitize_linear_hdr(linear_srgb_to_linear_display_p3(exposed));
  let display_color = select(
    linear_to_srgb(mapped),
    extended_display_p3,
    parameters.flags_balance_rg.y > 0.5
  );
  textureStore(display_output, vec2<i32>(global_id.xy), vec4<f32>(display_color, 1.0));
}
`
  );
  var PHOTOGRAPHIC_EFFECTS_SHADER = (
    /* wgsl */
    `
struct PhotographicParameters {
  sizes: vec4<f32>,
  grain: vec4<f32>,
  aberration: vec4<f32>,
  frame_seed: u32,
  output_is_hdr: u32,
  padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> parameters: PhotographicParameters;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var display_source: texture_2d<f32>;
@group(0) @binding(3) var display_output: texture_storage_2d<rgba16float, write>;

fn sanitize_display(color: vec3<f32>) -> vec3<f32> {
  let clamped = clamp(color, vec3<f32>(0.0), vec3<f32>(65504.0));
  return select(vec3<f32>(0.0), clamped, color == color);
}

fn bounded_sample_uv(uv: vec2<f32>) -> vec2<f32> {
  let half_texel = parameters.sizes.zw * 0.5;
  return clamp(uv, half_texel, vec2<f32>(1.0) - half_texel);
}

fn hash_u32(value: u32) -> u32 {
  var state = value;
  state ^= state >> 16u;
  state *= 0x7feb352du;
  state ^= state >> 15u;
  state *= 0x846ca68bu;
  state ^= state >> 16u;
  return state;
}

fn triangular_lattice_noise(cell: vec2<i32>, seed: u32) -> f32 {
  let x = bitcast<u32>(cell.x);
  let y = bitcast<u32>(cell.y);
  let keyed = hash_u32(
    (x * 0x8da6b343u) ^ (y * 0xd8163841u) ^ (seed * 0xcb1ab31fu)
  );
  let first = f32(keyed & 0x00ffffffu) * (1.0 / 16777216.0);
  let second_hash = hash_u32(keyed ^ 0xa511e9b3u);
  let second = f32(second_hash & 0x00ffffffu) * (1.0 / 16777216.0);
  return first - second;
}

fn smooth_grain_noise(pixel: vec2<f32>, scale: f32, seed: u32) -> f32 {
  let lattice_position = pixel / max(scale, 0.5) - vec2<f32>(0.5);
  let cell = vec2<i32>(floor(lattice_position));
  let fractional = fract(lattice_position);
  let weight = fractional * fractional * (vec2<f32>(3.0) - 2.0 * fractional);
  let n00 = triangular_lattice_noise(cell, seed);
  let n10 = triangular_lattice_noise(cell + vec2<i32>(1, 0), seed);
  let n01 = triangular_lattice_noise(cell + vec2<i32>(0, 1), seed);
  let n11 = triangular_lattice_noise(cell + vec2<i32>(1, 1), seed);
  return mix(mix(n00, n10, weight.x), mix(n01, n11, weight.x), weight.y);
}

fn grain_luminance_envelope(color: vec3<f32>) -> f32 {
  let luminance = dot(max(color, vec3<f32>(0.0)), vec3<f32>(0.2126, 0.7152, 0.0722));
  let perceptual_luminance = select(
    clamp(luminance, 0.0, 1.0),
    luminance / (1.0 + luminance),
    parameters.output_is_hdr != 0u
  );
  let shadow_gate = smoothstep(0.008, 0.10, perceptual_luminance);
  let highlight_gate = 1.0 - smoothstep(0.72, 0.985, perceptual_luminance);
  return shadow_gate * highlight_gate;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let output_dimensions = textureDimensions(display_output);
  if (any(global_id.xy >= output_dimensions)) {
    return;
  }

  let pixel = vec2<i32>(global_id.xy);
  let pixel_center = vec2<f32>(global_id.xy) + vec2<f32>(0.5);
  let uv = pixel_center * parameters.sizes.zw;
  var color = sanitize_display(textureLoad(display_source, pixel, 0).rgb);

  if (parameters.aberration.x > 0.0) {
    // Pixel space is aspect-correct by construction. Normalizing by the half
    // diagonal gives the same radial falloff at all four image corners.
    let from_center_pixels = pixel_center - parameters.sizes.xy * 0.5;
    let center_distance = length(from_center_pixels);
    let half_diagonal = max(length(parameters.sizes.xy) * 0.5, 1.0);
    let normalized_radius = clamp(center_distance / half_diagonal, 0.0, 1.0);
    let edge_start = clamp(parameters.aberration.y, 0.0, 1.0);
    let falloff_width = max(parameters.aberration.z, 0.0001);
    let radial_weight = smoothstep(edge_start, edge_start + falloff_width, normalized_radius);
    if (center_distance > 0.0001 && radial_weight > 0.0) {
      let radial_direction = from_center_pixels / center_distance;
      let offset_uv = radial_direction * parameters.aberration.x * radial_weight * parameters.sizes.zw;
      let red = textureSampleLevel(
        display_source,
        linear_sampler,
        bounded_sample_uv(uv + offset_uv),
        0.0
      ).r;
      let blue = textureSampleLevel(
        display_source,
        linear_sampler,
        bounded_sample_uv(uv - offset_uv),
        0.0
      ).b;
      color = sanitize_display(vec3<f32>(red, color.g, blue));
    }
  }

  if (parameters.grain.x > 0.0) {
    let noise = smooth_grain_noise(pixel_center, parameters.grain.y, parameters.frame_seed);
    let envelope = grain_luminance_envelope(color);
    // A scalar gain preserves chromaticity. The symmetric triangular hash has
    // zero expected mean, while the luminance envelope protects black and
    // highlight detail from clipping bias.
    color *= max(1.0 + noise * parameters.grain.x * envelope, 0.0);
  }

  color = sanitize_display(color);
  if (parameters.output_is_hdr == 0u) {
    color = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
  }
  textureStore(display_output, pixel, vec4<f32>(color, 1.0));
}
`
  );

  // src/renderer/post/denoise.ts
  var QUALITY_DEGRID_ITERATION = 4;
  var QUALITY_FIREFLY_COLOR_SIGMA_BOOST = 4;
  var QUALITY_FIREFLY_CONFIDENCE_SCALE = 8;
  var QUALITY_FIREFLY_PREFILTER_FULL_SAMPLES_MAX = 16;
  var QUALITY_FIREFLY_PREFILTER_FADE_SAMPLES_MAX = 64;
  var QUALITY_BACKPLATE_GLASS_FIREFLY_FADE_SAMPLES_MAX = 256;
  var QUALITY_BACKPLATE_GLASS_FIREFLY_RESIDUAL_STRENGTH = 0.15;
  var QUALITY_BACKPLATE_GLASS_BILATERAL_STRENGTH = 0.82;
  var QUALITY_BACKPLATE_GLASS_PRESERVATION_THRESHOLD = 0.02;
  var QUALITY_BACKPLATE_GLASS_INSTABILITY_THRESHOLD = 0.12;
  var QUALITY_FIREFLY_LOCAL_HEADROOM_STOPS = 0.5;
  var QUALITY_FIREFLY_LOCAL_DEVIATION_SCALE = 1.5;
  var QUALITY_FIREFLY_MIN_GUIDE_SUPPORT = 3;
  var QUALITY_FIREFLY_FULL_GUIDE_SUPPORT = 8;
  var QUALITY_FIREFLY_TEMPORAL_MIN_GUIDE_SUPPORT = 0.25;
  var QUALITY_FIREFLY_TEMPORAL_FULL_GUIDE_SUPPORT = 1;
  var QUALITY_FIREFLY_TEMPORAL_NOISE_MIN = 0.18;
  var QUALITY_FIREFLY_TEMPORAL_NOISE_FULL = 0.45;
  var QUALITY_FIREFLY_TEMPORAL_RESIDUAL_MIN = 0.04;
  var QUALITY_FIREFLY_TEMPORAL_RESIDUAL_FULL = 0.18;
  var QUALITY_FIREFLY_TEMPORAL_HEADROOM_STOPS = 0.125;
  var QUALITY_FIREFLY_TEMPORAL_DEVIATION_SCALE = 0.75;
  var QUALITY_FIREFLY_ISOLATION_RAMP_STOPS = 0.75;
  var QUALITY_FIREFLY_PERSISTENT_NOISE_MIN = 0.35;
  var QUALITY_FIREFLY_PERSISTENT_NOISE_FULL = 0.75;
  var QUALITY_FIREFLY_PERSISTENT_RESIDUAL_MIN = 0.12;
  var QUALITY_FIREFLY_PERSISTENT_RESIDUAL_FULL = 0.4;
  var QUALITY_FIREFLY_PERSISTENT_UNCERTAINTY_MIN = 3e-3;
  var QUALITY_FIREFLY_PERSISTENT_UNCERTAINTY_FULL = 0.012;
  var QUALITY_NEAR_DELTA_ROUGHNESS_MAX = 0.03;
  var QUALITY_NEAR_DELTA_PRESERVATION_THRESHOLD = 0.04;
  var QUALITY_NEAR_DELTA_INSTABILITY_THRESHOLD = 0.12;
  var QUALITY_ROUGH_PRESERVATION_THRESHOLD = 0.015;
  var QUALITY_ROUGH_INSTABILITY_THRESHOLD = 0.08;
  var QUALITY_ROUGH_NORMAL_RELAXATION_START = 0.35;
  var QUALITY_ROUGH_NORMAL_RELAXATION_END = 0.8;
  var QUALITY_ROUGH_NORMAL_EXPONENT_SCALE = 0.25;
  var QUALITY_NORMAL_DISCONTINUITY_MIN = 0.45;
  var QUALITY_NORMAL_DISCONTINUITY_FULL = 0.75;
  var ATROUS_ROUGHNESS_SIGMA = 0.08;
  var UNLIT_GBUFFER_ROUGHNESS_MAX = 0.01;
  var BACKPLATE_GBUFFER_ROUGHNESS_MARKER = 2 / 255;
  var BACKPLATE_GBUFFER_ROUGHNESS_TOLERANCE = 0.5 / 255;
  var QUALITY_FIREFLY_DEPTH_SIGMA_MAX = 0.015;
  var ATROUS_DENOISE_SHADER = (
    /* wgsl */
    `
struct AtrousParameters {
  image: vec4<f32>,
  sigmas: vec4<f32>,
  estimator: vec4<f32>,
}

@group(0) @binding(0) var<uniform> parameters: AtrousParameters;
@group(0) @binding(1) var input_hdr: texture_2d<f32>;
@group(0) @binding(2) var scene_depth: texture_2d<f32>;
@group(0) @binding(3) var normal_roughness: texture_2d<f32>;
@group(0) @binding(4) var surface_albedo: texture_2d<f32>;
@group(0) @binding(5) var output_hdr: texture_storage_2d<rgba16float, write>;

fn clamp_pixel(pixel: vec2<i32>) -> vec2<i32> {
  let dimensions = vec2<i32>(textureDimensions(input_hdr));
  return clamp(pixel, vec2<i32>(0), dimensions - vec2<i32>(1));
}

fn safe_normal(value: vec3<f32>) -> vec3<f32> {
  let length_squared = dot(value, value);
  if (length_squared <= 1e-8) {
    return vec3<f32>(0.0, 0.0, 1.0);
  }
  return value * inverseSqrt(length_squared);
}

fn luminance(color: vec3<f32>) -> f32 {
  return dot(max(color, vec3<f32>(0.0)), vec3<f32>(0.2126, 0.7152, 0.0722));
}

const MAX_FILTER_RADIANCE: f32 = 65504.0;
const ALBEDO_DIVISOR_FLOOR: f32 = 0.04;
const MIN_DEMODULATION_ALBEDO_LUMINANCE: f32 = 0.04;
const MIN_DEMODULATION_ROUGHNESS: f32 = 0.45;
const QUALITY_FIREFLY_DEPTH_SIGMA_MAX: f32 = ${QUALITY_FIREFLY_DEPTH_SIGMA_MAX};
const QUALITY_FIREFLY_COLOR_SIGMA_BOOST: f32 = ${QUALITY_FIREFLY_COLOR_SIGMA_BOOST}.0;
const QUALITY_FIREFLY_CONFIDENCE_SCALE: f32 = ${QUALITY_FIREFLY_CONFIDENCE_SCALE}.0;
const QUALITY_FIREFLY_PREFILTER_FULL_SAMPLES_MAX: f32 = ${QUALITY_FIREFLY_PREFILTER_FULL_SAMPLES_MAX}.0;
const QUALITY_FIREFLY_PREFILTER_FADE_SAMPLES_MAX: f32 = ${QUALITY_FIREFLY_PREFILTER_FADE_SAMPLES_MAX}.0;
const QUALITY_BACKPLATE_GLASS_FIREFLY_FADE_SAMPLES_MAX: f32 = ${QUALITY_BACKPLATE_GLASS_FIREFLY_FADE_SAMPLES_MAX}.0;
const QUALITY_BACKPLATE_GLASS_FIREFLY_RESIDUAL_STRENGTH: f32 = ${QUALITY_BACKPLATE_GLASS_FIREFLY_RESIDUAL_STRENGTH};
const QUALITY_BACKPLATE_GLASS_BILATERAL_STRENGTH: f32 = ${QUALITY_BACKPLATE_GLASS_BILATERAL_STRENGTH};
const QUALITY_BACKPLATE_GLASS_PRESERVATION_THRESHOLD: f32 = ${QUALITY_BACKPLATE_GLASS_PRESERVATION_THRESHOLD};
const QUALITY_BACKPLATE_GLASS_INSTABILITY_THRESHOLD: f32 = ${QUALITY_BACKPLATE_GLASS_INSTABILITY_THRESHOLD};
const QUALITY_FIREFLY_LOCAL_HEADROOM_STOPS: f32 = ${QUALITY_FIREFLY_LOCAL_HEADROOM_STOPS};
const QUALITY_FIREFLY_LOCAL_DEVIATION_SCALE: f32 = ${QUALITY_FIREFLY_LOCAL_DEVIATION_SCALE};
const QUALITY_FIREFLY_MIN_GUIDE_SUPPORT: f32 = ${QUALITY_FIREFLY_MIN_GUIDE_SUPPORT}.0;
const QUALITY_FIREFLY_FULL_GUIDE_SUPPORT: f32 = ${QUALITY_FIREFLY_FULL_GUIDE_SUPPORT}.0;
const QUALITY_FIREFLY_TEMPORAL_MIN_GUIDE_SUPPORT: f32 = ${QUALITY_FIREFLY_TEMPORAL_MIN_GUIDE_SUPPORT};
const QUALITY_FIREFLY_TEMPORAL_FULL_GUIDE_SUPPORT: f32 = ${QUALITY_FIREFLY_TEMPORAL_FULL_GUIDE_SUPPORT}.0;
const QUALITY_FIREFLY_TEMPORAL_NOISE_MIN: f32 = ${QUALITY_FIREFLY_TEMPORAL_NOISE_MIN};
const QUALITY_FIREFLY_TEMPORAL_NOISE_FULL: f32 = ${QUALITY_FIREFLY_TEMPORAL_NOISE_FULL};
const QUALITY_FIREFLY_TEMPORAL_RESIDUAL_MIN: f32 = ${QUALITY_FIREFLY_TEMPORAL_RESIDUAL_MIN};
const QUALITY_FIREFLY_TEMPORAL_RESIDUAL_FULL: f32 = ${QUALITY_FIREFLY_TEMPORAL_RESIDUAL_FULL};
const QUALITY_FIREFLY_TEMPORAL_HEADROOM_STOPS: f32 = ${QUALITY_FIREFLY_TEMPORAL_HEADROOM_STOPS};
const QUALITY_FIREFLY_TEMPORAL_DEVIATION_SCALE: f32 = ${QUALITY_FIREFLY_TEMPORAL_DEVIATION_SCALE};
const QUALITY_FIREFLY_ISOLATION_RAMP_STOPS: f32 = ${QUALITY_FIREFLY_ISOLATION_RAMP_STOPS};
const QUALITY_FIREFLY_PERSISTENT_NOISE_MIN: f32 = ${QUALITY_FIREFLY_PERSISTENT_NOISE_MIN};
const QUALITY_FIREFLY_PERSISTENT_NOISE_FULL: f32 = ${QUALITY_FIREFLY_PERSISTENT_NOISE_FULL};
const QUALITY_FIREFLY_PERSISTENT_RESIDUAL_MIN: f32 = ${QUALITY_FIREFLY_PERSISTENT_RESIDUAL_MIN};
const QUALITY_FIREFLY_PERSISTENT_RESIDUAL_FULL: f32 = ${QUALITY_FIREFLY_PERSISTENT_RESIDUAL_FULL};
const QUALITY_FIREFLY_PERSISTENT_UNCERTAINTY_MIN: f32 = ${QUALITY_FIREFLY_PERSISTENT_UNCERTAINTY_MIN};
const QUALITY_FIREFLY_PERSISTENT_UNCERTAINTY_FULL: f32 = ${QUALITY_FIREFLY_PERSISTENT_UNCERTAINTY_FULL};
const QUALITY_NEAR_DELTA_ROUGHNESS_MAX: f32 = ${QUALITY_NEAR_DELTA_ROUGHNESS_MAX};
const QUALITY_NEAR_DELTA_PRESERVATION_THRESHOLD: f32 = ${QUALITY_NEAR_DELTA_PRESERVATION_THRESHOLD};
const QUALITY_NEAR_DELTA_INSTABILITY_THRESHOLD: f32 = ${QUALITY_NEAR_DELTA_INSTABILITY_THRESHOLD};
const QUALITY_ROUGH_PRESERVATION_THRESHOLD: f32 = ${QUALITY_ROUGH_PRESERVATION_THRESHOLD};
const QUALITY_ROUGH_INSTABILITY_THRESHOLD: f32 = ${QUALITY_ROUGH_INSTABILITY_THRESHOLD};
const QUALITY_ROUGH_NORMAL_RELAXATION_START: f32 = ${QUALITY_ROUGH_NORMAL_RELAXATION_START};
const QUALITY_ROUGH_NORMAL_RELAXATION_END: f32 = ${QUALITY_ROUGH_NORMAL_RELAXATION_END};
const QUALITY_ROUGH_NORMAL_EXPONENT_SCALE: f32 = ${QUALITY_ROUGH_NORMAL_EXPONENT_SCALE};
const QUALITY_NORMAL_DISCONTINUITY_MIN: f32 = ${QUALITY_NORMAL_DISCONTINUITY_MIN};
const QUALITY_NORMAL_DISCONTINUITY_FULL: f32 = ${QUALITY_NORMAL_DISCONTINUITY_FULL};
const ATROUS_ROUGHNESS_SIGMA: f32 = ${ATROUS_ROUGHNESS_SIGMA};
const UNLIT_GBUFFER_ROUGHNESS_MAX: f32 = ${UNLIT_GBUFFER_ROUGHNESS_MAX};
const BACKPLATE_GBUFFER_ROUGHNESS_MARKER: f32 = ${BACKPLATE_GBUFFER_ROUGHNESS_MARKER};
const BACKPLATE_GBUFFER_ROUGHNESS_TOLERANCE: f32 = ${BACKPLATE_GBUFFER_ROUGHNESS_TOLERANCE};
const QUALITY_DEGRID_ITERATION: f32 = ${QUALITY_DEGRID_ITERATION}.0;

fn is_backplate_glass_marker(roughness: f32) -> bool {
  return abs(roughness - BACKPLATE_GBUFFER_ROUGHNESS_MARKER)
    <= BACKPLATE_GBUFFER_ROUGHNESS_TOLERANCE;
}

fn is_filterable_radiance(value: vec3<f32>) -> bool {
  return all(value == value)
    && all(value >= vec3<f32>(0.0))
    && all(value <= vec3<f32>(MAX_FILTER_RADIANCE));
}

fn sanitize_radiance(value: vec3<f32>) -> vec3<f32> {
  if (!is_filterable_radiance(value)) {
    return vec3<f32>(0.0);
  }
  return value;
}

fn sanitize_noise(value: f32) -> f32 {
  if (value != value) {
    return 1.0;
  }
  return clamp(value, 0.0, 1.0);
}

fn can_filter_illumination(radiance: vec3<f32>, albedo: vec3<f32>, roughness: f32) -> bool {
  return roughness >= MIN_DEMODULATION_ROUGHNESS
    && luminance(albedo) >= MIN_DEMODULATION_ALBEDO_LUMINANCE
    && is_filterable_radiance(radiance);
}

fn demodulate_radiance(
  radiance: vec3<f32>,
  albedo: vec3<f32>,
  maximum_illumination: vec3<f32>
) -> vec3<f32> {
  let clamped_albedo = max(albedo, vec3<f32>(ALBEDO_DIVISOR_FLOOR));
  return min(radiance / clamped_albedo, maximum_illumination);
}

fn winsorize_luminance(color: vec3<f32>, ceiling: f32, strength: f32) -> vec3<f32> {
  let color_luminance = luminance(color);
  let bounded_ceiling = max(ceiling, 0.0);
  let target_scale = select(
    1.0,
    bounded_ceiling / max(color_luminance, 1e-6),
    color_luminance > bounded_ceiling
  );
  return color * mix(1.0, target_scale, clamp(strength, 0.0, 1.0));
}

fn quality_normal_weight(
  alignment: f32,
  center_roughness: f32,
  sample_roughness: f32,
  base_exponent: f32
) -> f32 {
  let shared_roughness = min(center_roughness, sample_roughness);
  let relaxation = smoothstep(
    QUALITY_ROUGH_NORMAL_RELAXATION_START,
    QUALITY_ROUGH_NORMAL_RELAXATION_END,
    shared_roughness
  );
  let relaxed_exponent = max(base_exponent * QUALITY_ROUGH_NORMAL_EXPONENT_SCALE, 1.0);
  let effective_exponent = mix(base_exponent, relaxed_exponent, relaxation);
  let discontinuity_gate = smoothstep(
    QUALITY_NORMAL_DISCONTINUITY_MIN,
    QUALITY_NORMAL_DISCONTINUITY_FULL,
    alignment
  );
  return pow(max(alignment, 0.0), effective_exponent) * discontinuity_gate;
}

fn quality_guides_are_compatible(
  center_depth: f32,
  center_normal: vec3<f32>,
  center_albedo: vec3<f32>,
  center_roughness: f32,
  sample_depth: f32,
  sample_normal: vec3<f32>,
  sample_albedo: vec3<f32>,
  sample_roughness: f32,
  depth_sigma: f32,
  albedo_sigma: f32,
  allow_backplate_glass_marker: bool
) -> bool {
  let relative_depth_delta = abs(sample_depth - center_depth) / max(abs(center_depth), 1e-3);
  let normal_alignment = max(dot(center_normal, sample_normal), 0.0);
  let albedo_delta = length(sample_albedo - center_albedo);
  let roughness_delta = abs(sample_roughness - center_roughness);
  return sample_depth < 65000.0
    && (sample_roughness > UNLIT_GBUFFER_ROUGHNESS_MAX
      || (allow_backplate_glass_marker
        && is_backplate_glass_marker(sample_roughness)))
    && relative_depth_delta <= max(depth_sigma * 3.0, 0.002)
    && normal_alignment >= 0.75
    && albedo_delta <= max(albedo_sigma * 2.5, 0.12)
    && roughness_delta <= 0.12;
}

fn firefly_guide_weight(
  center_depth: f32,
  center_normal: vec3<f32>,
  center_albedo: vec3<f32>,
  center_roughness: f32,
  sample_depth: f32,
  sample_normal: vec3<f32>,
  sample_albedo: vec3<f32>,
  sample_roughness: f32,
  depth_sigma: f32,
  normal_exponent: f32,
  albedo_sigma: f32,
  allow_backplate_glass_marker: bool
) -> f32 {
  if (!quality_guides_are_compatible(
    center_depth,
    center_normal,
    center_albedo,
    center_roughness,
    sample_depth,
    sample_normal,
    sample_albedo,
    sample_roughness,
    depth_sigma,
    albedo_sigma,
    allow_backplate_glass_marker
  )) {
    return 0.0;
  }
  let relative_depth_delta = abs(sample_depth - center_depth) / max(abs(center_depth), 1e-3);
  let normal_alignment = max(dot(center_normal, sample_normal), 0.0);
  let albedo_delta = length(sample_albedo - center_albedo);
  let roughness_delta = abs(sample_roughness - center_roughness);
  let depth_weight = exp(-relative_depth_delta / max(depth_sigma, 1e-5));
  let normal_weight = quality_normal_weight(
    normal_alignment,
    center_roughness,
    sample_roughness,
    max(normal_exponent * 0.25, 1.0)
  );
  let albedo_weight = exp(-albedo_delta / max(albedo_sigma, 1e-5));
  let roughness_weight = exp(-roughness_delta / ATROUS_ROUGHNESS_SIGMA);
  return depth_weight * normal_weight * albedo_weight * roughness_weight;
}

fn quality_firefly_sample_fade(accumulated_samples: f32) -> f32 {
  return 1.0 - smoothstep(
    QUALITY_FIREFLY_PREFILTER_FULL_SAMPLES_MAX,
    QUALITY_FIREFLY_PREFILTER_FADE_SAMPLES_MAX,
    accumulated_samples
  );
}

fn quality_backplate_glass_firefly_sample_fade(accumulated_samples: f32) -> f32 {
  let primary_fade = quality_firefly_sample_fade(accumulated_samples);
  let residual_fade = QUALITY_BACKPLATE_GLASS_FIREFLY_RESIDUAL_STRENGTH * (
    1.0 - smoothstep(
      QUALITY_FIREFLY_PREFILTER_FADE_SAMPLES_MAX,
      QUALITY_BACKPLATE_GLASS_FIREFLY_FADE_SAMPLES_MAX,
      accumulated_samples
    )
  );
  return primary_fade + (1.0 - primary_fade) * residual_fade;
}

fn quality_firefly_temporal_innovation(
  noise: f32,
  local_noise: f32
) -> f32 {
  let absolute_innovation = smoothstep(
    QUALITY_FIREFLY_TEMPORAL_NOISE_MIN,
    QUALITY_FIREFLY_TEMPORAL_NOISE_FULL,
    noise
  );
  let residual_innovation = smoothstep(
    QUALITY_FIREFLY_TEMPORAL_RESIDUAL_MIN,
    QUALITY_FIREFLY_TEMPORAL_RESIDUAL_FULL,
    max(noise - local_noise, 0.0)
  );
  return max(absolute_innovation, residual_innovation);
}

fn quality_firefly_persistent_innovation(
  noise: f32,
  local_noise: f32,
  estimator_uncertainty: f32
) -> f32 {
  return smoothstep(
    QUALITY_FIREFLY_PERSISTENT_NOISE_MIN,
    QUALITY_FIREFLY_PERSISTENT_NOISE_FULL,
    noise
  ) * smoothstep(
    QUALITY_FIREFLY_PERSISTENT_RESIDUAL_MIN,
    QUALITY_FIREFLY_PERSISTENT_RESIDUAL_FULL,
    max(noise - local_noise, 0.0)
  ) * smoothstep(
    QUALITY_FIREFLY_PERSISTENT_UNCERTAINTY_MIN,
    QUALITY_FIREFLY_PERSISTENT_UNCERTAINTY_FULL,
    estimator_uncertainty
  );
}

fn quality_firefly_spatial_isolation(
  log_luminance: f32,
  local_log_mean: f32,
  local_log_headroom: f32
) -> f32 {
  let log_excess = log_luminance - local_log_mean;
  return smoothstep(
    local_log_headroom,
    local_log_headroom + QUALITY_FIREFLY_ISOLATION_RAMP_STOPS,
    log_excess
  );
}

fn kernel_weight(offset: i32) -> f32 {
  let distance = abs(offset);
  if (distance == 0) {
    return 6.0;
  }
  if (distance == 1) {
    return 4.0;
  }
  return 1.0;
}

fn atrous_sample_offset(
  kernel_offset: vec2<i32>,
  step_width: i32,
  iteration: f32,
  quality_filter: bool
) -> vec2<i32> {
  if (quality_filter && iteration >= QUALITY_DEGRID_ITERATION) {
    // Rotate the final footprint off the pixel axes without expanding its
    // sample-adaptive radius. This avoids repeatedly spreading sparse early
    // estimators along the same rows and columns.
    let degrid_axis_step = max(i32(parameters.estimator.y), 1);
    let degrid_cross_step = max(i32(parameters.estimator.z), 1);
    return vec2<i32>(
      kernel_offset.x * degrid_axis_step
        + kernel_offset.y * degrid_cross_step,
      kernel_offset.y * degrid_axis_step
        - kernel_offset.x * degrid_cross_step
    );
  }
  return kernel_offset * step_width;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let output_dimensions = textureDimensions(output_hdr);
  if (any(global_id.xy >= output_dimensions)) {
    return;
  }

  let center_pixel = vec2<i32>(global_id.xy);
  let center_color = textureLoad(input_hdr, center_pixel, 0);
  let center_depth = textureLoad(scene_depth, center_pixel, 0).w;
  let center_normal = safe_normal(textureLoad(normal_roughness, center_pixel, 0).xyz);
  let center_surface = textureLoad(surface_albedo, center_pixel, 0);
  let center_albedo = center_surface.rgb;
  let center_roughness = center_surface.a;
  let step_width = max(i32(parameters.image.z), 1);
  let quality_filter = parameters.sigmas.y <= QUALITY_FIREFLY_DEPTH_SIGMA_MAX;
  let quality_firefly_pass = parameters.image.w < 0.5
    && quality_filter;
  let center_noise = sanitize_noise(center_color.a);
  let accumulated_samples = max(parameters.estimator.x, 1.0);
  let inverse_sqrt_samples = inverseSqrt(accumulated_samples);
  let center_estimator_uncertainty = center_noise * inverse_sqrt_samples;
  let center_backplate_glass_prefilter = quality_firefly_pass
    && is_backplate_glass_marker(center_roughness);
  let backplate_glass_firefly_fade = select(
    0.0,
    quality_backplate_glass_firefly_sample_fade(accumulated_samples),
    center_backplate_glass_prefilter
  );
  let center_unlit = center_roughness <= UNLIT_GBUFFER_ROUGHNESS_MAX
    && !center_backplate_glass_prefilter;
  let center_near_delta = center_roughness <= QUALITY_NEAR_DELTA_ROUGHNESS_MAX;
  let rough_filter_strength = select(
    1.0,
    smoothstep(
      QUALITY_ROUGH_PRESERVATION_THRESHOLD,
      QUALITY_ROUGH_INSTABILITY_THRESHOLD,
      center_estimator_uncertainty
    ),
    quality_filter
  );
  let near_delta_filter_strength = select(
    0.0,
    smoothstep(
      QUALITY_NEAR_DELTA_PRESERVATION_THRESHOLD,
      QUALITY_NEAR_DELTA_INSTABILITY_THRESHOLD,
      center_estimator_uncertainty
    ),
    center_near_delta && quality_filter
  );
  let requested_center_filter_strength = select(
    rough_filter_strength,
    near_delta_filter_strength,
    center_near_delta
  );
  let backplate_glass_instability = smoothstep(
    QUALITY_BACKPLATE_GLASS_PRESERVATION_THRESHOLD,
    QUALITY_BACKPLATE_GLASS_INSTABILITY_THRESHOLD,
    center_estimator_uncertainty
  );
  let center_filter_strength = select(
    requested_center_filter_strength,
    QUALITY_BACKPLATE_GLASS_BILATERAL_STRENGTH
      * backplate_glass_firefly_fade
      * backplate_glass_instability,
    center_backplate_glass_prefilter
  );
  let effective_color_sigma = parameters.sigmas.x * select(
    1.0,
    1.0 + QUALITY_FIREFLY_COLOR_SIGMA_BOOST
      * center_estimator_uncertainty * center_estimator_uncertainty,
    quality_firefly_pass
  );

  // Direct environment and unlit radiance is deterministic. Marked backplate
  // glass gets a tightly bounded first-pass cleanup; later passes copy that
  // protected result. Other stable near-delta surfaces retain transmitted or
  // reflected detail.
  if (center_depth >= 65000.0
      || center_unlit) {
    textureStore(output_hdr, center_pixel, center_color);
    return;
  }

  let center_radiance = sanitize_radiance(center_color.rgb);
  let filter_illumination = can_filter_illumination(
    center_color.rgb,
    center_albedo,
    center_roughness
  );
  let center_albedo_divisor = max(center_albedo, vec3<f32>(ALBEDO_DIVISOR_FLOOR));
  let maximum_illumination = vec3<f32>(MAX_FILTER_RADIANCE) / center_albedo_divisor;
  var center_filter_value = center_radiance;
  if (filter_illumination) {
    center_filter_value = demodulate_radiance(
      center_radiance,
      center_albedo,
      maximum_illumination
    );
  }
  var firefly_sample_fade = 0.0;
  if (quality_firefly_pass) {
    firefly_sample_fade = quality_firefly_sample_fade(accumulated_samples);
    if (center_backplate_glass_prefilter) {
      firefly_sample_fade = backplate_glass_firefly_fade;
    }
  }
  var firefly_local_ceiling = MAX_FILTER_RADIANCE;
  var firefly_temporal_ceiling = MAX_FILTER_RADIANCE;
  var firefly_local_log_mean = 0.0;
  var firefly_local_log_headroom = MAX_FILTER_RADIANCE;
  var firefly_local_noise_mean = center_noise;
  var firefly_local_support_strength = 0.0;
  var firefly_temporal_support_strength = 0.0;
  let persistent_firefly_candidate = quality_firefly_pass
    && center_noise > QUALITY_FIREFLY_PERSISTENT_NOISE_MIN
    && center_estimator_uncertainty > QUALITY_FIREFLY_PERSISTENT_UNCERTAINTY_MIN;
  let inspect_firefly_neighborhood = firefly_sample_fade > 0.0 || persistent_firefly_candidate;
  if (inspect_firefly_neighborhood) {
    var local_log_luminance_sum = 0.0;
    var local_log_luminance_squared_sum = 0.0;
    var local_statistical_weight = 0.0;
    var local_guide_support = 0.0;
    var local_noise_sum = 0.0;
    for (var local_offset_y = -2; local_offset_y <= 2; local_offset_y += 1) {
      for (var local_offset_x = -2; local_offset_x <= 2; local_offset_x += 1) {
        if (local_offset_x == 0 && local_offset_y == 0) {
          continue;
        }
        let local_pixel = clamp_pixel(
          center_pixel + vec2<i32>(local_offset_x, local_offset_y)
        );
        let local_color = textureLoad(input_hdr, local_pixel, 0);
        if (!is_filterable_radiance(local_color.rgb)) {
          continue;
        }
        let local_depth = textureLoad(scene_depth, local_pixel, 0).w;
        let local_normal = safe_normal(textureLoad(normal_roughness, local_pixel, 0).xyz);
        let local_surface = textureLoad(surface_albedo, local_pixel, 0);
        let local_albedo = local_surface.rgb;
        let local_roughness = local_surface.a;
        let local_backplate_glass = quality_firefly_pass
          && is_backplate_glass_marker(local_roughness);
        if (center_backplate_glass_prefilter
            && !local_backplate_glass) {
          continue;
        }
        if (local_roughness <= UNLIT_GBUFFER_ROUGHNESS_MAX
            && !local_backplate_glass) {
          continue;
        }

        var local_filter_value = sanitize_radiance(local_color.rgb);
        if (filter_illumination) {
          if (!can_filter_illumination(local_color.rgb, local_albedo, local_roughness)) {
            continue;
          }
          local_filter_value = demodulate_radiance(
            local_filter_value,
            local_albedo,
            maximum_illumination
          );
        }
        let guide_weight = firefly_guide_weight(
          center_depth,
          center_normal,
          center_albedo,
          center_roughness,
          local_depth,
          local_normal,
          local_albedo,
          local_roughness,
          parameters.sigmas.y,
          parameters.sigmas.z,
          parameters.sigmas.w,
          quality_firefly_pass
        );
        if (guide_weight <= 0.0) {
          continue;
        }
        let local_noise = sanitize_noise(local_color.a);
        let local_uncertainty = local_noise * inverse_sqrt_samples;
        let local_confidence = 1.0 / (
          1.0 + QUALITY_FIREFLY_CONFIDENCE_SCALE * local_uncertainty * local_uncertainty
        );
        let statistical_weight = guide_weight * local_confidence;
        let log_luminance = log2(max(luminance(local_filter_value), 1e-4));
        local_log_luminance_sum += log_luminance * statistical_weight;
        local_log_luminance_squared_sum += log_luminance * log_luminance * statistical_weight;
        local_statistical_weight += statistical_weight;
        local_guide_support += guide_weight;
        local_noise_sum += local_noise * guide_weight;
      }
    }

    if (local_statistical_weight > 1e-5
        && local_guide_support >= QUALITY_FIREFLY_TEMPORAL_MIN_GUIDE_SUPPORT) {
      let local_log_mean = local_log_luminance_sum / local_statistical_weight;
      let local_log_variance = max(
        local_log_luminance_squared_sum / local_statistical_weight
          - local_log_mean * local_log_mean,
        0.0
      );
      let local_log_deviation = sqrt(local_log_variance);
      let local_log_headroom = max(
        QUALITY_FIREFLY_LOCAL_HEADROOM_STOPS,
        QUALITY_FIREFLY_LOCAL_DEVIATION_SCALE * local_log_deviation
      );
      firefly_local_log_mean = local_log_mean;
      firefly_local_log_headroom = local_log_headroom;
      firefly_local_noise_mean = local_noise_sum / max(local_guide_support, 1e-5);
      firefly_local_ceiling = min(
        exp2(local_log_mean + local_log_headroom),
        MAX_FILTER_RADIANCE
      );
      firefly_temporal_ceiling = min(
        exp2(local_log_mean + max(
          QUALITY_FIREFLY_TEMPORAL_HEADROOM_STOPS,
          QUALITY_FIREFLY_TEMPORAL_DEVIATION_SCALE * local_log_deviation
        )),
        MAX_FILTER_RADIANCE
      );
      firefly_local_support_strength = smoothstep(
        QUALITY_FIREFLY_MIN_GUIDE_SUPPORT,
        QUALITY_FIREFLY_FULL_GUIDE_SUPPORT,
        local_guide_support
      );
      firefly_temporal_support_strength = smoothstep(
        QUALITY_FIREFLY_TEMPORAL_MIN_GUIDE_SUPPORT,
        QUALITY_FIREFLY_TEMPORAL_FULL_GUIDE_SUPPORT,
        local_guide_support
      );
    }
  }
  var prefiltered_center_radiance = center_radiance;
  if (inspect_firefly_neighborhood
      && (firefly_local_support_strength > 0.0
        || firefly_temporal_support_strength > 0.0)) {
    let center_firefly_instability = smoothstep(
      QUALITY_ROUGH_PRESERVATION_THRESHOLD,
      QUALITY_ROUGH_INSTABILITY_THRESHOLD,
      center_estimator_uncertainty
    );
    let center_firefly_prefilter_strength = firefly_sample_fade
      * firefly_local_support_strength
      * center_firefly_instability;
    let center_temporal_innovation = quality_firefly_temporal_innovation(
      center_noise,
      firefly_local_noise_mean
    );
    let center_spatial_isolation = quality_firefly_spatial_isolation(
      log2(max(luminance(center_filter_value), 1e-4)),
      firefly_local_log_mean,
      firefly_local_log_headroom
    );
    let center_persistent_innovation = quality_firefly_persistent_innovation(
      center_noise,
      firefly_local_noise_mean,
      center_estimator_uncertainty
    );
    let center_temporal_prefilter_strength = (
      firefly_sample_fade * firefly_temporal_support_strength * center_temporal_innovation
      + (1.0 - firefly_sample_fade) * firefly_local_support_strength * center_persistent_innovation
    ) * center_spatial_isolation;
    center_filter_value = winsorize_luminance(
      center_filter_value,
      firefly_local_ceiling,
      center_firefly_prefilter_strength
    );
    center_filter_value = winsorize_luminance(
      center_filter_value,
      firefly_temporal_ceiling,
      center_temporal_prefilter_strength
    );
    if (firefly_sample_fade > 0.0 || center_temporal_prefilter_strength > 0.0) {
      prefiltered_center_radiance = center_filter_value;
      if (filter_illumination) {
        prefiltered_center_radiance = center_filter_value * center_albedo_divisor;
      }
    }
  }
  if (center_filter_strength <= 0.0) {
    textureStore(
      output_hdr,
      center_pixel,
      vec4<f32>(prefiltered_center_radiance, center_color.a)
    );
    return;
  }
  let center_filter_luminance = luminance(center_filter_value);

  var accumulated_filter_value = vec3<f32>(0.0);
  var accumulated_weight = 0.0;
  for (var offset_y = -2; offset_y <= 2; offset_y += 1) {
    for (var offset_x = -2; offset_x <= 2; offset_x += 1) {
      let kernel_offset = vec2<i32>(offset_x, offset_y);
      let sample_offset = atrous_sample_offset(
        kernel_offset,
        step_width,
        parameters.image.w,
        quality_filter
      );
      let sample_pixel = clamp_pixel(center_pixel + sample_offset);
      let sample_color = textureLoad(input_hdr, sample_pixel, 0);
      let sample_radiance = sanitize_radiance(sample_color.rgb);
      let sample_noise = sanitize_noise(sample_color.a);
      let sample_estimator_uncertainty = sample_noise * inverse_sqrt_samples;
      let sample_depth = textureLoad(scene_depth, sample_pixel, 0).w;
      let sample_normal = safe_normal(textureLoad(normal_roughness, sample_pixel, 0).xyz);
      let sample_surface = textureLoad(surface_albedo, sample_pixel, 0);
      let sample_albedo = sample_surface.rgb;
      let sample_roughness = sample_surface.a;
      let sample_backplate_glass = center_backplate_glass_prefilter
        && is_backplate_glass_marker(sample_roughness);
      let sample_unlit = sample_roughness <= UNLIT_GBUFFER_ROUGHNESS_MAX
        && !sample_backplate_glass;

      var sample_filter_value = sample_radiance;
      var filter_compatibility = select(1.0, 0.0, sample_unlit);
      if (center_backplate_glass_prefilter
          && !sample_backplate_glass) {
        filter_compatibility = 0.0;
      }
      if (filter_illumination) {
        if (can_filter_illumination(sample_color.rgb, sample_albedo, sample_roughness)) {
          sample_filter_value = demodulate_radiance(
            sample_radiance,
            sample_albedo,
            maximum_illumination
          );
        } else {
          // Mixing raw radiance into an illumination-domain average has invalid units.
          filter_compatibility = 0.0;
        }
      }
      if (inspect_firefly_neighborhood
          && (firefly_local_support_strength > 0.0
            || firefly_temporal_support_strength > 0.0)) {
        let sample_firefly_instability = smoothstep(
          QUALITY_ROUGH_PRESERVATION_THRESHOLD,
          QUALITY_ROUGH_INSTABILITY_THRESHOLD,
          sample_estimator_uncertainty
        );
        let sample_firefly_prefilter_strength = firefly_sample_fade
          * firefly_local_support_strength
          * sample_firefly_instability;
        sample_filter_value = winsorize_luminance(
          sample_filter_value,
          firefly_local_ceiling,
          sample_firefly_prefilter_strength
        );
        let sample_temporal_innovation = quality_firefly_temporal_innovation(
          sample_noise,
          firefly_local_noise_mean
        );
        let sample_spatial_isolation = quality_firefly_spatial_isolation(
          log2(max(luminance(sample_filter_value), 1e-4)),
          firefly_local_log_mean,
          firefly_local_log_headroom
        );
        let sample_persistent_innovation = quality_firefly_persistent_innovation(
          sample_noise,
          firefly_local_noise_mean,
          sample_estimator_uncertainty
        );
        let sample_temporal_prefilter_strength = (
          firefly_sample_fade * firefly_temporal_support_strength * sample_temporal_innovation
          + (1.0 - firefly_sample_fade) * firefly_local_support_strength * sample_persistent_innovation
        ) * sample_spatial_isolation;
        sample_filter_value = winsorize_luminance(
          sample_filter_value,
          firefly_temporal_ceiling,
          sample_temporal_prefilter_strength
        );
      }

      let spatial_weight = kernel_weight(offset_x) * kernel_weight(offset_y);
      let sample_filter_luminance = luminance(sample_filter_value);
      let luminance_scale = max(
        sqrt(max(center_filter_luminance, sample_filter_luminance)),
        0.05
      );
      let color_delta = abs(sample_filter_luminance - center_filter_luminance) / luminance_scale;
      let color_weight = exp(-color_delta / effective_color_sigma);
      let relative_depth_delta = abs(sample_depth - center_depth) / max(abs(center_depth), 1e-3);
      let depth_weight = exp(-relative_depth_delta / parameters.sigmas.y);
      let normal_alignment = max(dot(center_normal, sample_normal), 0.0);
      let standard_normal_weight = pow(normal_alignment, parameters.sigmas.z);
      let relaxed_normal_weight = quality_normal_weight(
        normal_alignment,
        center_roughness,
        sample_roughness,
        parameters.sigmas.z
      );
      let normal_weight = select(standard_normal_weight, relaxed_normal_weight, quality_filter);
      let albedo_delta = length(sample_albedo - center_albedo);
      let albedo_weight = exp(-albedo_delta / parameters.sigmas.w);
      let roughness_delta = abs(sample_roughness - center_roughness);
      let roughness_weight = exp(-roughness_delta / ATROUS_ROUGHNESS_SIGMA);
      let sample_confidence = select(
        1.0,
        1.0 / (1.0 + QUALITY_FIREFLY_CONFIDENCE_SCALE
          * sample_estimator_uncertainty * sample_estimator_uncertainty),
        quality_firefly_pass
      );
      let weight = spatial_weight
        * color_weight
        * depth_weight
        * normal_weight
        * albedo_weight
        * roughness_weight
        * filter_compatibility
        * sample_confidence;

      accumulated_filter_value += sample_filter_value * weight;
      accumulated_weight += weight;
    }
  }

  let filtered_domain_value = accumulated_filter_value / max(accumulated_weight, 1e-6);
  var filtered_radiance = filtered_domain_value;
  if (filter_illumination) {
    filtered_radiance = filtered_domain_value * center_albedo_divisor;
  }
  let bounded_radiance = clamp(
    filtered_radiance,
    vec3<f32>(0.0),
    vec3<f32>(MAX_FILTER_RADIANCE)
  );
  let output_radiance = mix(
    prefiltered_center_radiance,
    bounded_radiance,
    center_filter_strength
  );
  textureStore(output_hdr, center_pixel, vec4<f32>(output_radiance, center_color.a));
}
`
  );

  // src/renderer/post/learned-enhancement.ts
  var LEARNED_RESIDUAL_NETWORK = Object.freeze({
    architecture: Object.freeze([11, 32, 24, 5]),
    activation: "relu",
    domain: "normalized-log2-HDR-residual",
    heads: Object.freeze(["denoise", "2x-even-even", "2x-odd-even", "2x-even-odd", "2x-odd-odd"]),
    layers: Object.freeze([
      Object.freeze({
        inputs: 11,
        size: 32,
        weights: Object.freeze([
          -0.49707683,
          0.11808412,
          -0.14675151,
          0.07992672,
          0.41051334,
          -0.29892153,
          -0.07930201,
          -0.15412357,
          -0.33395458,
          0.40595685,
          0.07882618,
          -0.14712224,
          0.3118929,
          0.30198212,
          -0.25328624,
          0.13321421,
          0.60211124,
          0.02336738,
          0.10686044,
          0.57716713,
          0.31429931,
          -0.04087136,
          -0.03524444,
          0.27308811,
          -0.70263495,
          0.23880885,
          -0.06830598,
          0.64980107,
          -0.4184798,
          0.24425199,
          0.03448149,
          -0.05558424,
          -0.05748726,
          0.09338448,
          0.04011554,
          -0.62698601,
          -0.37488125,
          0.04615935,
          -0.48297671,
          0.29665997,
          0.07899383,
          -0.03008746,
          0.20366045,
          -0.28010866,
          0.05533663,
          0.02271312,
          0.56239585,
          0.12451602,
          -0.49357488,
          -0.27415639,
          0.03534925,
          -0.33413125,
          0.34888409,
          0.17903208,
          -0.14721822,
          -0.31327677,
          0.91360222,
          -0.08849705,
          1.00615335,
          -0.09815152,
          702955e-8,
          -0.12046709,
          -0.43647349,
          -480551e-8,
          -0.70502433,
          -0.0176977,
          -0.03531481,
          -0.15938318,
          -0.07240116,
          -987246e-8,
          0.16035112,
          0.46243169,
          0.11996854,
          -0.14065831,
          -0.14899149,
          0.59783897,
          -0.06090956,
          0.36365532,
          -0.205695,
          0.04006749,
          0.91075502,
          -0.37476119,
          -0.1231864,
          0.25505756,
          -0.32075977,
          -0.21613014,
          -0.25228593,
          -0.14959517,
          0.10180797,
          -0.48271735,
          -0.09409158,
          0.01968439,
          0.1599462,
          0.18713455,
          -0.41849345,
          -0.45327718,
          -0.2832683,
          0.23219365,
          0.06189918,
          0.10210212,
          -0.43468397,
          -0.21691982,
          0.19287526,
          -0.04989171,
          -0.3491987,
          0.34313618,
          0.45847217,
          0.37789298,
          0.3408761,
          -0.10139198,
          0.90344032,
          -803553e-8,
          -0.28944635,
          -0.18001004,
          -0.13042709,
          -0.49498846,
          -0.06565966,
          -0.28390402,
          -0.05204627,
          -0.05737113,
          0.07868102,
          -0.49629945,
          0.45216804,
          0.16003141,
          0.43478341,
          -0.34818357,
          0.18557823,
          0.15433956,
          0.54984263,
          -0.20863926,
          -0.0982863,
          0.38245965,
          -0.18622863,
          -0.06904749,
          -0.06278021,
          -0.09290129,
          0.78201665,
          0.06994503,
          -0.0870237,
          -0.33560912,
          -0.50397743,
          -0.30709328,
          -0.26015794,
          -0.37439849,
          0.17293591,
          0.44227298,
          -0.56284118,
          -0.52560386,
          -0.05017162,
          0.35608408,
          0.20249246,
          -0.22074179,
          -0.46443948,
          -0.04937844,
          -0.28807312,
          -0.15046099,
          -0.23821185,
          -0.51584488,
          -0.15441259,
          0.09632077,
          -0.19170927,
          0.84924214,
          -4361e-7,
          -0.48989358,
          0.1573366,
          0.315946,
          -0.65682512,
          0.37677833,
          -0.22387252,
          -0.31917335,
          0.17775647,
          0.42320763,
          0.03651095,
          -0.21530907,
          -0.45903614,
          0.55526791,
          -364179e-8,
          0.03847555,
          0.04180981,
          0.09183871,
          -0.73470469,
          -0.1812905,
          0.02654105,
          0.03368053,
          0.67385689,
          -0.49744927,
          0.32231242,
          -846754e-8,
          -0.25602871,
          0.11835499,
          -0.09337037,
          0.2272915,
          -0.38314932,
          0.24103536,
          -0.11660389,
          -0.70256519,
          0.58863242,
          0.28077852,
          0.16057952,
          0.31365478,
          0.02099077,
          -0.10657597,
          -0.01884369,
          -0.38125719,
          0.02181443,
          -0.46390922,
          -0.15638435,
          0.20421832,
          0.17818674,
          -0.36238284,
          0.03567175,
          -0.1145104,
          -0.50377114,
          0.64319116,
          -0.24804682,
          -0.04836095,
          0.0941876,
          -0.01266443,
          -0.19663065,
          -0.23906918,
          0.05956653,
          -0.1352486,
          -0.07640531,
          -0.15853787,
          -0.16846582,
          0.93923104,
          0.21645389,
          -0.69013058,
          -0.09385785,
          -0.7211674,
          -0.11421037,
          -0.16195336,
          -0.69471515,
          0.18602128,
          -0.27562121,
          0.23254886,
          -0.22511536,
          -0.28679329,
          -0.41885169,
          0.54150241,
          -0.25156334,
          0.18175985,
          0.03923183,
          -0.28236352,
          -0.1878072,
          0.72542687,
          -0.27690412,
          -0.56053081,
          -0.2858278,
          0.73056335,
          -0.1044485,
          -0.24377551,
          0.2306088,
          0.12756678,
          -0.06975066,
          -0.38268957,
          -0.06945088,
          0.54371733,
          -0.09944572,
          0.06153397,
          -0.39243987,
          0.28995774,
          -0.07809456,
          0.22630821,
          0.63411913,
          0.35826948,
          -0.09031086,
          0.24491559,
          -0.59331635,
          -0.04304418,
          -0.11224328,
          -0.02107353,
          0.34167265,
          0.36843318,
          0.20501159,
          -0.2335216,
          0.82879041,
          -0.11654458,
          -0.59634911,
          -0.2271589,
          0.36399052,
          -0.55985773,
          -0.17617372,
          -0.50512846,
          -0.16565827,
          0.04005685,
          0.5079594,
          0.74993971,
          0.01628233,
          0.33039159,
          -0.63324527,
          -0.18653111,
          0.22375873,
          -0.19979135,
          -0.3414916,
          -0.1954292,
          -0.18565385,
          -0.28629085,
          -0.26534193,
          0.6052288,
          -0.28630723,
          0.32611094,
          -0.10455553,
          -0.06930422,
          -0.22925035,
          -0.28779162,
          -0.04471535,
          -0.20507069,
          -0.09955856,
          -0.10355304,
          -0.24697857,
          -0.50599096,
          0.2838354,
          -0.324823,
          -0.70513687,
          -0.05328611,
          -0.04716329,
          0.3290884,
          -0.15797167,
          -0.10970241,
          0.53872933,
          -0.0566374,
          0.02336519,
          -0.12247779,
          0.1556122,
          0.43868649,
          -0.55497507,
          0.55166287,
          -0.59216646,
          0.33312286,
          -0.26981337,
          -0.35032135,
          0.86809721,
          -0.09082777,
          -0.3244553,
          -0.68107276,
          -0.06318601,
          -0.16878088,
          -0.23623216,
          -0.58766452,
          0.14382819,
          -0.1385256,
          0.06725673,
          0.21834242,
          0.28394594,
          -0.84049587,
          0.41917429,
          0.10116335,
          0.68696906,
          -0.02499826,
          0.293764,
          0.33517215
        ]),
        biases: Object.freeze([
          -0.1663864,
          -0.20273559,
          -0.55214674,
          -0.39848801,
          -0.48098131,
          -0.58604074,
          -0.4238218,
          -0.9187509,
          0.12658756,
          -0.30183716,
          -0.18020708,
          0.29046557,
          -0.66963108,
          -0.05684034,
          -0.21832831,
          -0.1304397,
          -0.49699518,
          -0.07718027,
          0.09665793,
          -0.49370898,
          -0.31596203,
          -0.35070039,
          -0.30696812,
          0.0150069,
          0.4724836,
          -0.40465483,
          -0.41627172,
          -0.25386005,
          0.47452979,
          -0.41348004,
          -0.37004069,
          0.17496589
        ])
      }),
      Object.freeze({
        inputs: 32,
        size: 24,
        weights: Object.freeze([
          0.08426129,
          0.67797825,
          -0.34854893,
          -0.61283061,
          0.0299848,
          -0.34363431,
          0.16417065,
          0.47695167,
          -0.19884537,
          0.4550357,
          0.01136327,
          0.36866516,
          -0.27119602,
          -0.06401871,
          0.01084299,
          -0.32599671,
          -0.24324159,
          0.4861054,
          0.28827039,
          0.3521857,
          0.22066113,
          -0.28476381,
          -0.16137044,
          0.16497485,
          -0.22916017,
          -0.33186828,
          0.33640455,
          0.22901045,
          0.29998168,
          -0.19460324,
          0.11579038,
          -0.39103427,
          -0.34530866,
          0.16819927,
          0.10896834,
          -0.03443751,
          -0.19233386,
          -0.43879355,
          0.31083491,
          0.47771513,
          -0.36670931,
          0.15743958,
          -0.66972596,
          0.35035102,
          0.37102373,
          0.26571133,
          0.197447,
          -0.04154522,
          0.42648747,
          0.1049777,
          -0.14208739,
          -0.07144824,
          -1.20955609,
          -0.08228616,
          -0.29660832,
          0.51286065,
          0.03003554,
          -0.44641289,
          -0.56488167,
          -0.2221872,
          0.57925913,
          0.09820275,
          -0.79741519,
          0.07274109,
          171828e-8,
          -0.12717229,
          -0.58280735,
          -0.10573979,
          0.09667582,
          -0.52182139,
          -0.23461583,
          -0.65374392,
          0.16610346,
          0.08390097,
          0.08209849,
          -0.8392311,
          0.75421121,
          -0.89857938,
          -0.25715748,
          -2.26237635,
          0.25532543,
          0.12928912,
          -0.14307629,
          0.29065905,
          -1.55408531,
          -0.4293097,
          -0.15028914,
          0.35117498,
          -0.15442076,
          -0.17713683,
          -0.40986367,
          0.22241627,
          0.1186505,
          -0.98165449,
          -0.80799354,
          0.29932657,
          -0.38962856,
          0.28591496,
          -0.03293636,
          -0.25837716,
          0.15206048,
          -503221e-8,
          -0.19034941,
          -0.74043157,
          0.11340648,
          -0.2413225,
          -0.0265513,
          0.22935589,
          0.78972052,
          -0.40129244,
          0.91716442,
          0.28999774,
          -0.68405349,
          0.56802048,
          -0.03518629,
          0.21240037,
          0.28494458,
          -0.58657581,
          -0.28238026,
          -0.09675046,
          0.10110458,
          -0.21588291,
          0.1138923,
          0.03121465,
          0.08913682,
          0.35507304,
          0.18596436,
          -0.18037434,
          0.28415766,
          -0.28405434,
          0.4767459,
          0.09258039,
          0.5092476,
          -0.9132094,
          -0.21354867,
          -0.67147527,
          0.36217614,
          0.17809199,
          0.55400888,
          0.18371725,
          -0.85711616,
          -0.39462317,
          -0.05624301,
          0.09226942,
          -0.16710391,
          -0.59980366,
          -0.06005413,
          0.35584818,
          -0.2051284,
          0.29901647,
          0.34196352,
          0.37403552,
          -0.56244243,
          0.53125013,
          0.22227397,
          -0.37764974,
          -0.31674623,
          -0.08130023,
          0.71547424,
          0.02738496,
          -0.14423945,
          0.53675998,
          0.30708392,
          0.5277951,
          -0.11611136,
          -0.13295296,
          -0.42801397,
          -0.35584383,
          0.50649385,
          -0.0876357,
          0.24534347,
          -0.27801746,
          -0.97139455,
          -0.36483305,
          -0.3642017,
          -0.25118341,
          0.15593962,
          -0.03740377,
          -0.09775827,
          -0.99696799,
          0.61856944,
          -0.66174541,
          0.63466051,
          -0.28650571,
          -0.62817665,
          0.21232557,
          -0.04830129,
          -0.01164944,
          -0.65718197,
          -0.67871725,
          0.58834079,
          0.38102408,
          0.46664815,
          0.17800496,
          -0.02320416,
          0.39367711,
          0.12367764,
          0.85794306,
          -0.34748592,
          0.64170502,
          0.84075197,
          0.23361012,
          0.36318862,
          -0.04659724,
          0.08705532,
          0.47566066,
          -0.29281988,
          -0.02817248,
          0.04685752,
          -0.31551419,
          0.03396214,
          0.29366252,
          -834464e-8,
          -0.3949338,
          0.05847397,
          -0.28615215,
          -0.37635966,
          -0.32273386,
          0.38974844,
          -126608e-8,
          -0.12899435,
          -0.03149779,
          0.25288553,
          0.24966415,
          0.43789474,
          -0.83788108,
          -1.11712235,
          0.57523516,
          -0.46634374,
          -0.28043747,
          0.03326044,
          -0.22169551,
          -0.55503957,
          -0.71563349,
          -0.32614587,
          0.02571696,
          -0.8512535,
          0.36947509,
          -0.15127121,
          -0.53238776,
          -0.01161571,
          -0.32804254,
          0.33980093,
          0.73582738,
          0.16427193,
          0.10719026,
          0.04608921,
          0.26045636,
          0.08748611,
          0.06081887,
          0.17550132,
          0.35871779,
          -1.12530696,
          0.09720732,
          0.12876123,
          -1.48918084,
          0.66224009,
          -0.35457558,
          -0.14954697,
          -0.54946915,
          -0.07512548,
          0.52648299,
          0.04622166,
          0.52312604,
          0.44237001,
          -0.50084825,
          -0.0783978,
          -0.01507706,
          0.12439205,
          -0.2871483,
          -0.4130948,
          0.12173836,
          -0.48025029,
          -0.62170514,
          -0.21172061,
          -0.61580261,
          -26087e-7,
          0.02490007,
          0.45589028,
          0.04106588,
          -0.10464088,
          0.2128426,
          0.58958858,
          -0.3673695,
          -0.2418731,
          -0.71185352,
          -0.8371136,
          0.02852362,
          0.37818363,
          0.12616743,
          -0.18461504,
          0.42718304,
          0.18290915,
          -0.51950981,
          -0.2145689,
          0.36209497,
          0.31529994,
          -0.20482788,
          -0.03793723,
          -0.93562358,
          -0.24528949,
          -0.03996591,
          0.28836222,
          -0.02595348,
          0.1292413,
          -0.2401646,
          0.52854607,
          -0.31729534,
          -0.31313925,
          -0.02677857,
          -0.28789198,
          0.25233777,
          -0.13088209,
          -0.54023275,
          -0.33439537,
          0.22700825,
          0.28930721,
          -0.57605174,
          -0.35160763,
          0.40090962,
          -0.71146636,
          -9745e-8,
          0.38505506,
          0.1715573,
          0.78637979,
          -0.49364391,
          -0.20535925,
          0.37759845,
          0.24568163,
          -0.6438735,
          0.59491587,
          0.08755647,
          0.29430009,
          0.41665681,
          0.14703697,
          0.18873508,
          -0.12876316,
          -0.02663855,
          0.25513218,
          -0.19533224,
          0.23050082,
          -0.38729651,
          0.50872956,
          -0.3959238,
          -0.7366263,
          0.71703762,
          -0.28406735,
          -0.68200791,
          0.2809114,
          -0.30145218,
          -0.18020368,
          0.29965706,
          0.06231017,
          -0.65802014,
          -0.0130288,
          0.18375648,
          -0.39545382,
          0.02883612,
          -0.02784894,
          -0.02879533,
          -0.2040886,
          -0.72231452,
          0.29190427,
          0.42095397,
          -0.2237289,
          -0.10801522,
          -0.01255394,
          0.36295808,
          -0.49672205,
          0.27030479,
          0.07749147,
          0.41874005,
          0.06259889,
          0.39173237,
          0.55973936,
          -0.14213002,
          -0.6375431,
          -0.10291423,
          0.16591039,
          -0.25738607,
          0.34495765,
          0.38470863,
          0.40972711,
          -0.78796704,
          -0.10138531,
          0.16147565,
          -0.34833745,
          0.29622436,
          -0.02224728,
          -0.37628571,
          0.04293597,
          0.23908605,
          -0.452382,
          0.06765518,
          -0.14287772,
          0.02386261,
          -0.26933678,
          0.06037102,
          -0.41486546,
          -0.28035397,
          -0.46576726,
          0.08259866,
          0.2865031,
          -0.13507073,
          -0.93263431,
          0.6897966,
          0.11108931,
          -0.19474054,
          0.54650463,
          -0.26607512,
          -0.26467871,
          0.0902788,
          0.37212006,
          0.22153507,
          -0.72788856,
          0.36677039,
          0.16723875,
          -0.09952713,
          0.06399308,
          -338703e-8,
          -0.11972028,
          0.53767744,
          0.07309924,
          0.26872246,
          -0.13741324,
          -0.15370242,
          0.43703981,
          -0.38924548,
          0.11640357,
          -0.44823101,
          -0.3120654,
          -0.37672499,
          0.40508913,
          0.17808441,
          0.12678696,
          -0.54664905,
          0.73968013,
          -0.35920229,
          -0.26368259,
          -0.08868177,
          -0.28679669,
          -0.09268006,
          -0.18045426,
          0.4247273,
          0.37123618,
          0.56449049,
          0.42165317,
          0.65683978,
          -0.06951557,
          0.32336859,
          -0.33631605,
          -0.19704325,
          -0.07459825,
          -0.2059225,
          0.29635831,
          0.28720737,
          0.06215006,
          0.31017931,
          -0.54908018,
          -0.1621779,
          0.32823013,
          -0.38644138,
          -0.16578758,
          -0.57254644,
          -0.34638097,
          0.08748564,
          0.50898768,
          0.42929254,
          172896e-8,
          -0.40130269,
          -0.58937907,
          0.07704925,
          0.41434579,
          -0.30611722,
          0.0330133,
          0.24978929,
          -0.65958398,
          -0.14242781,
          0.19677909,
          0.48930999,
          -0.19835702,
          0.15770112,
          -0.07924788,
          0.09383182,
          -0.91357125,
          0.09720811,
          -0.08954753,
          0.51238271,
          -0.18938018,
          0.15449248,
          -0.91622569,
          -0.01810119,
          -0.2814499,
          -0.34683005,
          0.36105699,
          -0.48914334,
          -0.17436901,
          0.78205273,
          -0.31583735,
          0.21522763,
          -0.17532666,
          0.14717934,
          -0.35922194,
          -0.074386,
          0.24137947,
          -0.36340902,
          -0.42532813,
          -0.546334,
          0.23450567,
          -0.23032911,
          -0.75219063,
          -0.11084795,
          0.08121366,
          0.15433656,
          -0.33382659,
          -0.18737944,
          -0.16788801,
          -0.58760202,
          -0.18572934,
          -0.09784276,
          -0.03784108,
          0.29609562,
          -0.25989536,
          -0.1974996,
          0.51791293,
          -0.04694541,
          -0.5131216,
          -0.09961301,
          0.61394144,
          0.18470458,
          -0.06570302,
          0.39269402,
          0.11670836,
          0.1011127,
          -0.50393232,
          0.5347561,
          0.17678584,
          -0.10891958,
          0.6674657,
          0.16307447,
          0.01822041,
          -0.13428837,
          -0.01210965,
          -0.60846922,
          -0.35745865,
          -0.31353335,
          0.24163228,
          0.07948617,
          0.92519141,
          0.07310835,
          -0.19129148,
          -0.49196142,
          0.0444174,
          0.18629848,
          -0.92006919,
          0.3307998,
          -0.91462891,
          0.2415627,
          0.41983321,
          -0.03474669,
          -0.52788894,
          0.05895806,
          -0.29564023,
          -1.04834355,
          -0.94817566,
          -0.87298415,
          0.07113648,
          -1.02350642,
          -0.23844839,
          -0.47044314,
          -0.33698129,
          -1.44302492,
          0.71864996,
          0.01736237,
          -0.94875895,
          -0.2640966,
          -0.18076646,
          -0.02583688,
          0.31857112,
          0.29295011,
          -0.03477388,
          0.45364391,
          -0.21738587,
          -0.17881867,
          0.22917915,
          -0.25653647,
          -0.04420273,
          0.09828817,
          -1.23768438,
          -0.05524404,
          0.43819736,
          0.08358292,
          0.81655425,
          0.14071888,
          0.25815596,
          -0.18857667,
          0.27178427,
          -0.46538849,
          -758761e-8,
          -0.12846031,
          0.03918293,
          -0.21489266,
          -0.10027873,
          0.17097086,
          -0.09251924,
          0.13810374,
          0.0160701,
          0.15828273,
          -0.11590098,
          0.21530975,
          -0.31799653,
          -0.01376101,
          -0.46919285,
          0.12326956,
          0.69524037,
          0.58645157,
          -0.03367066,
          -0.46532797,
          0.25036323,
          -0.10883913,
          -0.18878693,
          0.23851234,
          0.11742348,
          -0.25320421,
          0.37887343,
          0.03343794,
          0.17834399,
          0.02621229,
          -0.66509622,
          0.35382558,
          0.30238225,
          0.06591033,
          -0.05946904,
          0.19798718,
          0.0206594,
          -0.13758523,
          0.32510127,
          -0.24385731,
          0.10138752,
          0.2224476,
          -0.35090308,
          0.50956354,
          -0.41918036,
          -0.26632075,
          -0.16860754,
          -0.0643226,
          -0.21882664,
          0.36463538,
          -0.30527108,
          -0.02159075,
          -0.10995424,
          0.18712246,
          -0.01177692,
          -0.16476003,
          172917e-8,
          -0.24290693,
          -0.08924853,
          0.59326671,
          0.11710589,
          0.7269842,
          0.27638102,
          -0.03186435,
          -0.02380855,
          -0.08841901,
          0.37612593,
          -0.05421732,
          0.31965915,
          -182544e-8,
          0.08304567,
          -0.30197536,
          -0.13782156,
          0.37184799,
          -0.45146681,
          -0.1055582,
          -0.39458745,
          0.18725328,
          0.22946415,
          0.02660909,
          -0.30107486,
          -0.35466765,
          0.30409674,
          -0.67506985,
          0.24983543,
          -0.27875486,
          -0.29734327,
          0.01760466,
          0.36522504,
          0.09145638,
          0.25670319,
          -0.37556799,
          -0.34379688,
          -0.20549282,
          0.08543104,
          -0.30067539,
          -0.09929312,
          0.47321725,
          -0.30923786,
          0.74474018,
          -1.12497394,
          -0.37221844,
          0.29137101,
          0.37257725,
          0.21697212,
          0.49431618,
          0.02772408,
          0.53337837,
          -0.70795559,
          -0.28283261,
          0.32948052,
          0.52985039,
          0.6286799,
          -1.04195701,
          -0.11850956,
          -0.08807354,
          -0.18078514,
          -0.54610119,
          0.07825705,
          -0.31250843,
          0.34560867,
          -0.07439516,
          0.32003705,
          -0.13463072,
          -0.16449498,
          -0.28581321,
          -0.20872651,
          -0.08412974,
          -0.42135833,
          -0.11620463,
          0.5110913,
          -0.55437094,
          -0.28815911,
          0.21407866,
          0.41718305,
          -0.64100856,
          0.29871611,
          -0.40269246,
          0.09452079,
          0.16883852,
          -0.44875329,
          -0.1465242,
          -0.04784405,
          -2.3630946,
          -0.24206206,
          0.88224397,
          0.1355829,
          -0.17080884,
          -0.33348868,
          -1.65216791,
          0.17012567,
          0.87740995,
          -0.26683565,
          -0.43681926,
          0.46668954,
          0.27867792,
          -0.54080718,
          0.5863747,
          -0.27757903,
          0.23863227,
          -0.65205588,
          0.38798058,
          -0.4335524,
          -0.26837348,
          0.48236712,
          -0.2096637,
          64791e-7,
          -0.40535987,
          -0.7224497,
          -0.46496972
        ]),
        biases: Object.freeze([
          0.21889094,
          -0.40091215,
          -0.52488895,
          0.1838162,
          -23284e-8,
          -0.34736266,
          -0.62205257,
          -0.1310965,
          226138e-8,
          0.23789823,
          -0.40335511,
          -0.6131147,
          -0.42958079,
          -0.09568613,
          0.26827456,
          0.21072144,
          -0.15769971,
          -0.51610512,
          -0.29010926,
          0.07512199,
          -0.28448405,
          0.07917707,
          0.18116725,
          -0.42361612
        ])
      }),
      Object.freeze({
        inputs: 24,
        size: 5,
        weights: Object.freeze([
          -0.26036698,
          0.13647664,
          -0.69640856,
          0.09274119,
          0.11960939,
          -0.05210632,
          0.04015139,
          0.1002313,
          -446754e-8,
          -0.19262903,
          -0.11247833,
          0.18739999,
          0.15234154,
          -0.09550112,
          0.25520136,
          -0.24358797,
          -0.13401444,
          -0.19907485,
          -0.0335523,
          -0.06869695,
          0.24249337,
          0.1741927,
          0.0234971,
          -0.52536051,
          0.18568496,
          -0.34064057,
          0.48757543,
          0.0856389,
          0.29320939,
          -0.28867325,
          -0.38348616,
          -0.70237017,
          0.2505292,
          0.2048088,
          -0.43201787,
          -0.39902465,
          -0.03329655,
          -0.05972653,
          0.04453575,
          0.12165941,
          -0.40283552,
          -0.4119982,
          0.32851005,
          0.13591022,
          -0.01307439,
          0.27928146,
          -0.29567109,
          -0.10424556,
          0.19985894,
          -0.21236356,
          -0.0300879,
          0.20033858,
          0.22791591,
          0.1575027,
          -0.27782486,
          -0.88664571,
          0.33161446,
          0.16113456,
          -0.33142201,
          0.01503653,
          0.06119119,
          -0.43629482,
          0.27483511,
          -0.15571744,
          -0.13307565,
          -1.1788986,
          0.28412622,
          0.1734231,
          -0.25077111,
          0.22664724,
          -0.15863001,
          -0.34332061,
          0.20612426,
          -0.16420551,
          -0.18493588,
          0.17824919,
          0.19928904,
          -0.82786585,
          -0.30654567,
          -0.69388302,
          0.42450555,
          0.25075509,
          -0.27250845,
          -0.29128062,
          0.2212783,
          -0.04143527,
          -0.10758427,
          -0.02429627,
          -0.15416271,
          -0.12342556,
          0.07282197,
          -0.26392081,
          0.08282732,
          0.34658284,
          -0.25941414,
          -0.08965551,
          0.06597172,
          391763e-8,
          -0.66022325,
          0.33150349,
          0.26984755,
          -0.19588335,
          -0.19345383,
          -0.41281946,
          0.40923148,
          0.01857108,
          -0.3909965,
          0.20960376,
          0.29679185,
          -0.24166033,
          0.22061087,
          -0.24510921,
          -0.08574154,
          -0.39435122,
          -0.15705285,
          -0.03037707,
          -0.13778722,
          0.38936597,
          -0.18670057,
          -0.36101839
        ]),
        biases: Object.freeze([
          -0.04518343,
          -0.21771699,
          -0.22318852,
          -0.16030896,
          -0.21667241
        ])
      })
    ])
  });
  var LEARNED_RESIDUAL_NETWORK_EVIDENCE = Object.freeze({ "training": { "seed": 1380274994, "patches": 65536, "epochs": 80, "batchSize": 64, "optimizer": "Adam", "cleanTarget": "4x4 area-integrated radiance: source footprint 1, denoise footprint 1, subpixel footprint 0.5" }, "heldOut": { "seed": -204680817, "patchesPerClass": 2048, "metrics": { "smooth": [{ "baselineMse": 0.04171142827508997, "learnedMse": 0.010590163264235973 }, { "baselineMse": 0.016709972646655014, "learnedMse": 0.012196996194597326 }, { "baselineMse": 0.015755599478619334, "learnedMse": 0.011569696702930855 }, { "baselineMse": 0.016892165379550197, "learnedMse": 0.01147178970240307 }, { "baselineMse": 0.016283274551179956, "learnedMse": 0.012329120773427805 }], "edge": [{ "baselineMse": 0.039978745534485244, "learnedMse": 0.025336168701284946 }, { "baselineMse": 0.15024819722089267, "learnedMse": 0.08965167914352022 }, { "baselineMse": 0.18031844848695625, "learnedMse": 0.09462176126618357 }, { "baselineMse": 0.17627331830843798, "learnedMse": 0.09349653977489156 }, { "baselineMse": 0.20833860403242546, "learnedMse": 0.09222866267035015 }], "textile": [{ "baselineMse": 0.03921429817894771, "learnedMse": 0.023758514759197425 }, { "baselineMse": 0.10962071314765369, "learnedMse": 0.05517873517328411 }, { "baselineMse": 0.11453364693577243, "learnedMse": 0.05517306286391389 }, { "baselineMse": 0.10847342752156958, "learnedMse": 0.048488141819489516 }, { "baselineMse": 0.10849824381314079, "learnedMse": 0.04887365047211635 }], "outlier": [{ "baselineMse": 7.163468380468217, "learnedMse": 0.015095370155855964 }, { "baselineMse": 0.18361817546133216, "learnedMse": 0.013722821528475293 }, { "baselineMse": 0.35905943711151517, "learnedMse": 0.011262793530225986 }, { "baselineMse": 0.3888897899589984, "learnedMse": 0.014153951772140435 }, { "baselineMse": 2.2931409827103164, "learnedMse": 0.014974091723661164 }], "curve": [{ "baselineMse": 0.039052989884609306, "learnedMse": 0.011940394218804367 }, { "baselineMse": 0.01538910626780259, "learnedMse": 0.015869717552404104 }, { "baselineMse": 0.014817787739348508, "learnedMse": 0.014587157911553853 }, { "baselineMse": 0.015597007841276555, "learnedMse": 0.015182362374962831 }, { "baselineMse": 0.015512809211963399, "learnedMse": 0.01468229939756715 }], "detail": [{ "baselineMse": 0.040372590811741306, "learnedMse": 0.03584022944165596 }, { "baselineMse": 0.11841306582586027, "learnedMse": 0.06351438131274477 }, { "baselineMse": 0.12458490294726221, "learnedMse": 0.060240093381030806 }, { "baselineMse": 0.1332102778879683, "learnedMse": 0.06415424479482636 }, { "baselineMse": 0.1509114316459736, "learnedMse": 0.0693403275526497 }], "woven": [{ "baselineMse": 0.04466000455788437, "learnedMse": 0.05293859802012552 }, { "baselineMse": 0.10034856011356197, "learnedMse": 0.0853048029548386 }, { "baselineMse": 0.10321515750322624, "learnedMse": 0.08875682009539247 }, { "baselineMse": 0.10740063817585338, "learnedMse": 0.09487287175425181 }, { "baselineMse": 0.129525342834754, "learnedMse": 0.11706538295706866 }] } } });
  var LEARNED_DENOISE_RESIDUAL_NETWORK = Object.freeze({
    architecture: Object.freeze([11, 32, 24, 5]),
    activation: "relu",
    domain: "normalized-log2-HDR-residual",
    heads: Object.freeze(["denoise", "2x-even-even", "2x-odd-even", "2x-even-odd", "2x-odd-odd"]),
    layers: Object.freeze([
      Object.freeze({
        inputs: 11,
        size: 32,
        weights: Object.freeze([
          -0.12991092,
          0.01536196,
          -0.05679628,
          0.01361683,
          0.28614255,
          -0.07927125,
          -0.05225388,
          -0.1173794,
          -0.50385407,
          0.78767055,
          0.10775243,
          -0.03872992,
          -0.38517406,
          0.20778431,
          0.10556129,
          0.40324703,
          0.54555409,
          0.0360434,
          0.21021319,
          0.18858457,
          0.36792283,
          -0.04133649,
          -0.5098215,
          0.28472902,
          -0.13028882,
          0.14509546,
          -0.16089508,
          0.4394024,
          0.05468274,
          0.270764,
          -0.20382887,
          -0.97408304,
          0.17520365,
          0.14293459,
          0.04320968,
          -0.29676874,
          -0.51342237,
          0.23349052,
          -0.3824739,
          -0.04476635,
          -0.5696072,
          0.29574017,
          0.09137026,
          0.31451829,
          -0.16440321,
          -0.08550534,
          0.52386594,
          0.04581011,
          -0.29124685,
          0.01384498,
          -0.17292489,
          -0.0641862,
          0.22272623,
          0.20246887,
          -0.01350181,
          -0.16822777,
          0.56339393,
          0.04157052,
          0.23558401,
          0.19876032,
          -0.52161032,
          0.0746484,
          0.14526002,
          0.04749644,
          -0.79508148,
          -0.17026497,
          0.12515294,
          0.20411123,
          -0.26125404,
          -0.35554715,
          0.04399987,
          0.31462639,
          0.17735258,
          -0.09028428,
          -0.01659849,
          0.56686179,
          -0.15276185,
          0.02702716,
          -0.11668295,
          0.0941346,
          0.16575881,
          -0.4228599,
          -0.03185439,
          0.60790316,
          -0.03412392,
          0.02506742,
          -1.25396275,
          0.1088689,
          -0.09803487,
          -0.41068619,
          -0.21077781,
          -0.1567964,
          0.03260227,
          0.22323387,
          -0.45288138,
          -0.55821211,
          0.2729881,
          0.49228274,
          -0.35891616,
          0.02168177,
          -0.16651839,
          -0.23456981,
          0.14475465,
          -0.35935984,
          0.07085114,
          0.22180015,
          0.43326332,
          0.35032126,
          0.3861275,
          -0.1241652,
          0.61732325,
          -0.1140349,
          -0.16760157,
          -81357e-7,
          -0.2191466,
          -0.14776596,
          -0.21393149,
          -0.20970988,
          0.08219972,
          -1.08495233,
          0.07780225,
          -0.01456705,
          0.01463552,
          -0.05508344,
          0.53345009,
          -0.26642052,
          0.49999718,
          0.02248809,
          0.25365523,
          -0.33292926,
          0.10454899,
          -0.02269999,
          0.17568215,
          -0.32877891,
          -0.1240177,
          0.16124081,
          0.45003764,
          0.0585048,
          0.08692901,
          0.02434513,
          -0.39302349,
          0.02985578,
          -0.11690318,
          -0.24180791,
          0.15613315,
          0.16695314,
          0.09084977,
          -0.49987088,
          -0.12462747,
          0.03957086,
          0.35096263,
          -0.3467028,
          1.08682675,
          -0.16569956,
          -0.28701727,
          -0.02904641,
          -0.14753832,
          -0.50550634,
          0.19030754,
          -0.05383189,
          -0.09175354,
          0.20400249,
          -358107e-8,
          0.68479985,
          300791e-8,
          -0.13151229,
          0.32341591,
          -0.01268808,
          -0.68845297,
          -0.1823912,
          -0.03064658,
          0.06741563,
          0.04469081,
          0.10885597,
          -0.61622712,
          -0.0760567,
          0.05619816,
          0.04252948,
          0.12585866,
          -0.09123497,
          -0.37268025,
          -0.35542964,
          0.08856926,
          0.14103439,
          0.40110155,
          -1.24999237,
          -0.08594103,
          -0.01289281,
          -0.26191522,
          0.0439182,
          -0.24740759,
          -15315e-7,
          -0.16107046,
          958797e-8,
          -0.13917343,
          0.01311416,
          1.25084889,
          0.02866495,
          0.33033813,
          0.15184608,
          -0.18548686,
          -0.24127947,
          -0.06843575,
          -0.40683595,
          0.15719484,
          0.05458959,
          -0.36815092,
          0.52864152,
          0.33105685,
          -0.30759789,
          -0.05301841,
          0.07668603,
          -0.04102367,
          0.49847693,
          -0.06193758,
          -0.27887906,
          -0.22011409,
          0.12574788,
          -0.10466865,
          -0.30006676,
          -0.0467458,
          -0.0816433,
          0.11018682,
          -0.09359484,
          -0.08042089,
          0.72514619,
          0.13364977,
          -0.24081936,
          -0.59399769,
          -1.2621205,
          -0.38394247,
          -0.26339158,
          -0.21354202,
          -0.341009,
          -0.22496121,
          0.31414417,
          -0.4363597,
          -0.3204716,
          -0.11729904,
          0.50118602,
          0.06769785,
          -0.26835526,
          -0.04273684,
          -0.01644911,
          -0.05238089,
          0.1064625,
          -0.27379554,
          -0.21390003,
          -0.13548254,
          0.7870634,
          -0.26898518,
          -1.14956674,
          0.04526459,
          0.02548516,
          -0.24785169,
          -0.27416341,
          0.16998264,
          0.46860663,
          -0.1745009,
          -0.12843418,
          0.13253184,
          -0.16125282,
          -0.59560343,
          0.11982812,
          0.01430604,
          0.21884292,
          -605375e-8,
          0.42679831,
          -0.27292251,
          0.08134364,
          0.10019994,
          0.01677494,
          0.09787088,
          0.7031046,
          0.28288277,
          0.01336858,
          0.3213782,
          -0.49479604,
          -0.33589425,
          -0.17319761,
          140041e-8,
          -0.64378303,
          -0.05230556,
          0.04102221,
          -0.67032822,
          0.01721453,
          -0.26939634,
          0.27976999,
          -0.19648538,
          0.27125323,
          -0.10439922,
          0.02228731,
          0.31722017,
          -0.14865791,
          0.09984192,
          -1.09841831,
          0.1110093,
          -0.29851197,
          -0.34215465,
          -0.04659532,
          -0.34441536,
          0.49382714,
          0.05586472,
          -0.05882951,
          0.11013027,
          -0.32554128,
          -0.60651209,
          0.16177814,
          0.04260162,
          -0.03851417,
          -0.16777812,
          -0.25321146,
          0.05177208,
          -0.1517216,
          -0.39684847,
          -0.2330386,
          -0.30293413,
          0.89480884,
          0.12652471,
          0.50998437,
          0.31587547,
          -0.17414056,
          0.12846488,
          -0.32510015,
          0.10304585,
          -0.22686364,
          -0.08801619,
          0.61248401,
          -0.92604009,
          0.03932558,
          -0.42813884,
          0.1136625,
          0.35225165,
          -0.23701641,
          -0.16998811,
          -0.33704448,
          0.08265207,
          -0.35796119,
          -0.43817716,
          -0.15538133,
          0.12826484,
          0.11166351,
          0.25883584,
          0.30345343,
          0.07039091,
          -0.67145054,
          0.29057421,
          0.32161285,
          0.12506982,
          53937e-8,
          0.59719186,
          0.40820691
        ]),
        biases: Object.freeze([
          -0.08714704,
          -0.31682367,
          -0.05668421,
          -9433e-6,
          -0.16413837,
          -0.0712687,
          0.09821608,
          -0.11567065,
          -0.35018862,
          0.12597993,
          0.22483431,
          0.11008891,
          -0.345363,
          -0.0758154,
          -0.09396849,
          0.13389901,
          -0.30595422,
          -0.34214656,
          0.08491624,
          -0.22881544,
          -0.04751955,
          -0.53130115,
          0.35123024,
          -0.10572029,
          -0.02319042,
          -0.38671842,
          0.21731401,
          -0.519627,
          -0.22012409,
          157447e-8,
          -0.15824165,
          0.17721481
        ])
      }),
      Object.freeze({
        inputs: 32,
        size: 24,
        weights: Object.freeze([
          0.01997998,
          0.18816741,
          -0.41759127,
          -0.10756864,
          -0.18915068,
          0.2217616,
          -0.16931075,
          -0.08147875,
          -48827e-8,
          -0.04786486,
          -0.31348824,
          0.12666466,
          -0.25087857,
          0.68867964,
          0.24047607,
          0.09993794,
          -0.23024558,
          0.35996693,
          0.35747212,
          0.29253952,
          0.42775797,
          0.246806,
          -0.21337751,
          -0.02514387,
          -0.12313133,
          -0.20913311,
          0.11842541,
          0.33977419,
          0.19827842,
          0.25327209,
          0.10628032,
          -0.54567934,
          -0.34028831,
          0.11903368,
          0.21854951,
          0.28878616,
          0.17415132,
          -0.1063428,
          0.17661014,
          -0.19912845,
          -0.12247439,
          0.28351771,
          -0.53960269,
          -0.06618485,
          0.08605561,
          0.36523085,
          0.09685789,
          -0.14329593,
          0.06484659,
          0.25510797,
          -0.4777897,
          -0.24225949,
          -0.41511156,
          0.26640074,
          -0.15372039,
          0.206396,
          0.22466394,
          -0.46849645,
          -0.6047757,
          97745e-8,
          0.53623537,
          0.29382977,
          -0.34154701,
          0.10521371,
          0.01063384,
          -0.11262768,
          -0.76525789,
          0.16348053,
          -0.54923726,
          0.08519868,
          0.04358878,
          0.05283066,
          -0.12681136,
          -0.27334242,
          -0.83152732,
          -0.67617346,
          0.18767616,
          -0.0490286,
          0.17536131,
          -0.29660232,
          -0.05103317,
          -0.16107073,
          -0.29232407,
          0.23145562,
          -1.35162164,
          -0.02518433,
          -1.21677378,
          0.41082435,
          -0.53520588,
          -1.14573952,
          -1.03278865,
          0.40337692,
          0.40605184,
          -0.31711688,
          -0.24499836,
          -0.37823091,
          -0.07169326,
          -0.48336604,
          0.04287785,
          0.28480442,
          -0.23493095,
          -0.46557894,
          -0.14053871,
          -0.45256084,
          0.44912684,
          -0.67487641,
          -0.35420057,
          0.21974276,
          0.11477347,
          0.04150297,
          0.1134047,
          -0.291212,
          -0.78405882,
          0.48537352,
          -0.02890771,
          0.13257577,
          0.10076783,
          -0.58061948,
          -0.22797498,
          -0.15865305,
          0.55089743,
          -0.28836311,
          -0.6693676,
          0.21883525,
          0.21934029,
          0.20362001,
          0.10202461,
          0.31536635,
          0.01920064,
          -0.15492175,
          596294e-8,
          0.232741,
          0.32495801,
          0.04841972,
          0.25630351,
          -0.62649247,
          -28717e-7,
          0.43043405,
          0.46721011,
          0.14235171,
          -0.54701942,
          -0.32167595,
          -0.20838938,
          0.1256614,
          -0.2251501,
          -0.45553638,
          -0.21657251,
          -0.13563934,
          0.04583469,
          0.51619529,
          0.19741292,
          0.09171472,
          -0.08917016,
          -0.12266601,
          -0.09313798,
          -0.25877028,
          0.0290913,
          -0.2374266,
          -0.27154927,
          -0.22796941,
          -0.01444349,
          -0.03670258,
          0.04473766,
          0.12391569,
          -0.20722339,
          0.0654117,
          -0.29320168,
          -0.21630769,
          0.32084303,
          0.09981442,
          -0.08109947,
          0.45737943,
          -0.06974192,
          -0.31051962,
          -0.28158915,
          -0.18765854,
          -0.33007085,
          -0.10843234,
          0.27370724,
          0.16125129,
          0.61791234,
          -0.38846759,
          0.59204417,
          -0.55310747,
          0.16440802,
          0.26778821,
          -0.29531886,
          0.41914563,
          -0.05868771,
          -0.47072479,
          0.13607314,
          -0.22777812,
          0.41419746,
          0.14062242,
          0.23147435,
          0.35902145,
          0.40571966,
          0.30209316,
          0.19255964,
          0.46031406,
          0.24786499,
          -0.3589442,
          -0.08780448,
          0.16836447,
          0.02085717,
          0.02800624,
          -0.04804761,
          -0.2323719,
          0.34885444,
          0.36132124,
          0.11961543,
          0.02558032,
          -0.42347627,
          -0.76566146,
          -0.03908459,
          0.41802653,
          -0.18657586,
          -0.35524615,
          -0.10970119,
          -0.49884207,
          -0.20835882,
          0.3372697,
          -0.14378856,
          0.03241944,
          -0.05614771,
          -0.0388763,
          -1.0074511,
          0.05616003,
          -0.05765863,
          -0.19057114,
          0.24481526,
          -0.42502654,
          0.31582655,
          -0.75529729,
          0.37791944,
          -0.14085361,
          -0.06336264,
          0.26924472,
          -0.29629709,
          0.17911868,
          -0.5232351,
          -0.42725942,
          0.06355597,
          -0.37880522,
          -0.73477913,
          0.51440896,
          -0.80535039,
          0.18222155,
          0.15734697,
          0.15143147,
          0.04819965,
          -0.22434907,
          -0.10889392,
          -1.12245057,
          0.17093865,
          -0.36852375,
          0.03613871,
          -0.08266061,
          0.45371054,
          -0.2147257,
          -0.13663722,
          0.0177354,
          0.27064906,
          -0.24383408,
          0.58543003,
          -0.30104164,
          -0.10113335,
          0.16018296,
          0.18217169,
          -0.12825141,
          -0.17982067,
          0.09994431,
          0.08331487,
          -1.61482278,
          -0.03937475,
          -0.12719014,
          -0.39319136,
          0.02320284,
          -0.36086802,
          0.15423538,
          -0.05727252,
          -0.01365663,
          -0.15304271,
          -0.05864548,
          0.0317058,
          -0.22875215,
          0.05621626,
          -1.01496947,
          0.22898374,
          0.08930355,
          0.34470904,
          0.78880274,
          0.22436196,
          -0.28562389,
          -0.29771907,
          0.39163369,
          0.41076146,
          -0.09666733,
          -0.21368598,
          -0.18666305,
          0.23454726,
          0.0874715,
          -0.12026325,
          -0.07222848,
          0.17712626,
          0.33549673,
          0.17060132,
          0.2347001,
          -0.04533485,
          -0.08289408,
          -0.12683499,
          -0.26460257,
          0.02392313,
          0.01764548,
          -0.08136037,
          -0.5306441,
          -0.25197392,
          -0.51494314,
          -0.05699512,
          0.19560024,
          -0.30404297,
          -0.05169119,
          0.16408915,
          -0.01955175,
          -0.29174398,
          -0.23735777,
          -0.06309446,
          0.37789113,
          0.56944396,
          -0.22475952,
          0.13649052,
          0.42578002,
          881579e-8,
          0.59511626,
          0.14579556,
          -956513e-8,
          -0.01375407,
          -0.07236463,
          0.09825388,
          -0.41626827,
          0.54222284,
          -0.05040446,
          0.07179094,
          -0.12954691,
          -0.15953465,
          632102e-8,
          -0.81634954,
          -0.7769546,
          0.34312184,
          -0.61550464,
          -0.10710544,
          0.18545492,
          0.21377784,
          0.19482999,
          -0.07826983,
          -0.15560621,
          0.01151447,
          0.11367678,
          -0.08703966,
          -0.38147867,
          0.26604226,
          -0.47042736,
          -0.10872288,
          0.17703917,
          0.13155095,
          0.2173761,
          0.3440982,
          0.18338527,
          -0.94581345,
          -0.26646176,
          0.44899222,
          0.24359135,
          -0.26819503,
          -0.1806002,
          0.25360641,
          -0.1344682,
          -0.11666638,
          0.29469123,
          0.05187406,
          -0.31926205,
          -0.68039154,
          -0.21024913,
          0.252126,
          -0.44696328,
          0.14992221,
          0.24314199,
          -0.27334252,
          0.01388542,
          -0.38542004,
          -0.1128135,
          -0.18087676,
          -0.4698825,
          -0.46903209,
          0.25257053,
          -0.28059373,
          0.14179481,
          -0.38151134,
          0.2301885,
          -0.24933751,
          -0.50182803,
          0.19190623,
          0.18819767,
          0.27360592,
          0.39562981,
          -460265e-8,
          0.25189469,
          -0.20796708,
          0.04973086,
          0.23115192,
          -0.10916135,
          -0.17354713,
          -0.21928662,
          0.37164351,
          -0.23624117,
          -0.53660396,
          0.43051975,
          0.41678007,
          0.23570894,
          0.31224805,
          -0.24172268,
          0.39505025,
          0.04058502,
          0.13885818,
          -0.57527318,
          -0.16242133,
          0.08902813,
          -0.17531886,
          0.17992044,
          0.08166886,
          -0.28317784,
          0.10818627,
          0.08021402,
          0.13101172,
          -0.14082175,
          -0.05297014,
          -102745e-8,
          -0.51581547,
          -0.3356001,
          0.26441518,
          -0.63088556,
          -0.01260836,
          -0.39484982,
          -0.29241362,
          0.26139992,
          -0.14659325,
          0.53937721,
          -0.41286022,
          0.23563544,
          -0.29540636,
          0.58942041,
          -0.26754957,
          -0.06624582,
          -0.1055255,
          -0.12381215,
          0.25414493,
          -0.09645432,
          -0.44437119,
          0.11577397,
          -0.06280371,
          -0.07782232,
          0.26272135,
          -0.09197726,
          -0.27637424,
          -0.1479725,
          -0.53378985,
          0.26634273,
          0.32934091,
          0.01196528,
          -0.05591821,
          -0.07563373,
          -0.36677624,
          -0.45893588,
          0.47807108,
          -0.27602691,
          0.27196063,
          0.43476593,
          -0.46158262,
          87985e-7,
          0.30102799,
          0.21571393,
          0.10258121,
          -0.28119774,
          0.05201882,
          0.35625973,
          -0.43933321,
          0.04208366,
          -0.39575195,
          1.24269103,
          -0.11819186,
          0.17606348,
          -1.44162015,
          -0.95422591,
          0.06987348,
          -0.62164907,
          0.25082541,
          -0.34217758,
          0.39710413,
          -0.03930138,
          -0.39139445,
          0.2773661,
          -0.39834672,
          0.17035253,
          -0.94958687,
          0.28052228,
          -0.18958063,
          -0.62311663,
          -0.8058793,
          0.09076679,
          0.15512667,
          -0.1767092,
          -0.10724579,
          -0.93460916,
          0.4533203,
          -0.21320698,
          -0.60529445,
          0.10333503,
          -0.32694157,
          -0.48893803,
          -0.03513907,
          -0.09902909,
          -0.44815388,
          0.15348206,
          -0.01379187,
          -0.178736,
          0.37654233,
          0.25693623,
          0.34787506,
          -0.50876071,
          0.08166121,
          0.19678238,
          -0.22826341,
          0.09376846,
          -0.76526289,
          0.1292963,
          -0.39289678,
          -0.31037508,
          0.32488857,
          -0.17690768,
          0.23728103,
          -0.13845418,
          0.17658791,
          0.17557113,
          -0.62186751,
          -0.17855231,
          0.0865385,
          -0.21163759,
          -0.15236686,
          0.05646112,
          -0.09060996,
          -0.42557957,
          -0.24086036,
          -0.6665491,
          0.4877727,
          0.23994924,
          0.05645362,
          0.51908846,
          -0.31739209,
          0.0725709,
          0.25003313,
          0.19313102,
          -0.1434098,
          -0.36827425,
          0.09929735,
          -0.40986944,
          -1.32567811,
          0.25077811,
          0.467423,
          -0.17286191,
          -0.41130404,
          0.19055099,
          -0.14776854,
          -0.17838362,
          -0.68950566,
          -0.09922183,
          -0.2608158,
          -0.32337637,
          488078e-8,
          0.06503998,
          -0.32528932,
          -0.19388393,
          535973e-8,
          -0.35220238,
          -0.12147716,
          0.02931649,
          0.16071442,
          -0.11302517,
          -0.20868852,
          -0.15373929,
          0.25922027,
          0.57697805,
          0.16383242,
          -0.08329055,
          0.4409174,
          -0.32224801,
          -0.03577057,
          -0.21023611,
          -0.2933886,
          0.34398699,
          -0.03883677,
          -1.35613668,
          -0.19332927,
          -0.03398818,
          0.18593608,
          0.39548856,
          0.25238977,
          -0.14103971,
          -0.2497223,
          0.51220249,
          0.07192757,
          0.25257016,
          0.07699506,
          -0.2916442,
          -0.03973963,
          0.56305866,
          0.51046134,
          -0.02064094,
          -0.16512593,
          0.17965633,
          0.07894075,
          0.11976079,
          -0.16374558,
          -0.34137971,
          -0.03765016,
          -0.07332642,
          0.73836133,
          0.43978381,
          -0.18612097,
          0.11887125,
          0.05032754,
          0.28918299,
          0.13106681,
          -0.47691489,
          0.31973263,
          0.40543755,
          -0.02341674,
          0.13360202,
          -0.18846933,
          -0.29584437,
          0.06041927,
          0.25820883,
          -0.04876197,
          -0.20879754,
          0.16106387,
          0.03443516,
          0.23991498,
          -0.0375394,
          -0.34497648,
          0.1396477,
          -0.05994996,
          0.168552,
          -0.12977997,
          -0.26432092,
          -0.32541268,
          0.16124401,
          0.11797224,
          0.07610418,
          0.13982046,
          0.4228868,
          0.19272488,
          -0.07564521,
          0.2207737,
          0.16054713,
          204249e-8,
          -0.19390119,
          0.29419626,
          -0.33318996,
          0.30454973,
          0.35483456,
          0.2813755,
          0.42544317,
          -0.23420237,
          0.29731602,
          -0.5097201,
          -0.49986924,
          0.21689414,
          0.32282237,
          0.15452618,
          0.21760634,
          -0.11383612,
          -0.74333797,
          0.40420215,
          -0.09206251,
          -0.42057435,
          -0.05210712,
          0.04616621,
          0.10649328,
          0.62432202,
          -0.32924145,
          -0.80045903,
          -0.31195618,
          0.0117966,
          -0.1390143,
          -0.03918383,
          -0.37791139,
          -0.18663086,
          0.01967431,
          7685e-7,
          0.32438491,
          -0.13933917,
          -0.56394433,
          0.13809899,
          -0.15838013,
          -0.27136994,
          0.42884071,
          -0.14862651,
          0.42661558,
          0.31258274,
          -0.04966014,
          -0.13592567,
          -0.03408196,
          0.22018166,
          -1.16061853,
          0.09420257,
          -0.12600086,
          -0.48893727,
          0.50722486,
          -0.30338093,
          -0.26878891,
          -66143e-7,
          -0.14869788,
          1.31078586,
          0.1228164,
          0.09163835,
          0.32903928,
          0.01093072,
          0.25207751,
          -0.53986204,
          -0.4708597,
          0.2018772,
          -0.32359082,
          -0.49597719,
          0.29810848,
          0.24758211,
          0.58229617,
          0.12607358,
          0.15148403,
          0.15498052,
          -0.02115283,
          -0.46375118,
          -0.04766846,
          0.29356788,
          0.68335203,
          0.60426036,
          0.14333979,
          -0.09526268,
          -0.24862683,
          0.31257963,
          -0.01211344,
          -0.50494933,
          -0.25506836,
          -0.74573337,
          0.37406247,
          0.87298023,
          -0.41299479,
          -0.09200556,
          0.55303067,
          0.2364578,
          -1.00364959,
          -0.58451597,
          -190255e-8,
          0.13950387,
          -0.64045065,
          0.22208312,
          0.0645186,
          -0.10996083,
          -0.39299025,
          -443855e-8,
          -0.82492013
        ]),
        biases: Object.freeze([
          0.24660602,
          -0.14581893,
          -0.11355164,
          -0.27609989,
          -0.08725654,
          0.16806106,
          -0.02109079,
          -0.11272633,
          0.11037888,
          -701043e-8,
          -0.13716663,
          -0.11296432,
          -0.23547887,
          -0.09866067,
          -0.06843386,
          -0.0628514,
          0.2360324,
          -908164e-8,
          -0.05806259,
          -0.19465631,
          0.01943896,
          -0.02299735,
          -0.03325563,
          0.190241
        ])
      }),
      Object.freeze({
        inputs: 24,
        size: 5,
        weights: Object.freeze([
          -0.33736142,
          0.21320433,
          -0.74518663,
          0.12307882,
          0.2623511,
          0.20793664,
          -0.19603178,
          0.88018551,
          0.40096033,
          -0.25817942,
          -0.06770318,
          0.37771318,
          0.37885478,
          56594e-8,
          0.30405547,
          -0.47358564,
          -0.40965762,
          -0.30830097,
          0.25081731,
          -0.15860387,
          0.20706131,
          0.34930333,
          0.0118117,
          0.23838441,
          0.22990636,
          -0.44257803,
          0.02633245,
          0.34534244,
          0.25706162,
          -0.21202622,
          0.01129558,
          0.73105426,
          0.32058702,
          0.02858839,
          -0.24753054,
          -0.29771661,
          0.31725308,
          0.43086701,
          -0.06508834,
          -0.26084197,
          -0.16733011,
          -0.16499692,
          0.1457418,
          0.19764725,
          -0.27646917,
          0.0935965,
          -0.30347457,
          -0.52091165,
          -0.04473655,
          -0.14554081,
          -0.25925788,
          0.67812687,
          0.46361698,
          0.13935919,
          -0.14542979,
          0.37502388,
          0.27856539,
          0.16238925,
          -0.4735582,
          0.0735277,
          0.16160992,
          0.20065278,
          0.03883791,
          -0.02915577,
          -0.20105068,
          -0.75511594,
          0.12026532,
          0.3327695,
          -0.31248142,
          0.01592982,
          -0.20168851,
          -0.65782802,
          0.37333283,
          -0.13416947,
          -0.32398883,
          0.28737545,
          0.31789131,
          0.01296121,
          -0.29873512,
          0.24323808,
          0.8567246,
          0.20605272,
          -0.31932648,
          -0.1655664,
          0.30655935,
          0.47758648,
          490075e-8,
          -0.12613708,
          -0.30703708,
          -0.3206248,
          -0.14368849,
          -0.14825821,
          0.0794955,
          0.20000181,
          -0.58446822,
          -0.43616349,
          0.03089151,
          0.11986326,
          -0.51859079,
          0.43106896,
          0.5216527,
          0.22163059,
          -0.33073635,
          0.20653455,
          0.75179098,
          0.04775324,
          -0.39068805,
          0.21466722,
          0.35790764,
          0.21387686,
          0.1835928,
          -0.39187202,
          -0.41076402,
          -0.65607954,
          0.10911773,
          -0.02481124,
          -0.10254557,
          0.27593212,
          -0.29093691,
          -0.36341088
        ]),
        biases: Object.freeze([
          -0.11979728,
          0.09371154,
          0.09079332,
          -0.06610403,
          -0.01657172
        ])
      })
    ])
  });
  var LEARNED_DENOISE_RESIDUAL_NETWORK_EVIDENCE = Object.freeze({ "training": { "seed": 1380274994, "patches": 36864, "epochs": 64, "batchSize": 64, "optimizer": "Adam", "cleanTarget": "analytic point-sampled radiance at native and subpixel positions" }, "heldOut": { "seed": -204680817, "patchesPerClass": 2048, "metrics": { "smooth": [{ "baselineMse": 0.04171142827508997, "learnedMse": 0.007795445234573354 }, { "baselineMse": 0.016709972646654997, "learnedMse": 0.008801734176527328 }, { "baselineMse": 0.015755599478619337, "learnedMse": 0.007876639207578475 }, { "baselineMse": 0.016892165379550187, "learnedMse": 0.007726645597198213 }, { "baselineMse": 0.016283274551179946, "learnedMse": 0.007249094131832241 }], "edge": [{ "baselineMse": 0.039097982681014797, "learnedMse": 0.02937090818243399 }, { "baselineMse": 0.42608698965543645, "learnedMse": 0.408035544980317 }, { "baselineMse": 0.5700312885244371, "learnedMse": 0.4837127742094384 }, { "baselineMse": 0.6579219456720005, "learnedMse": 0.5416111333555141 }, { "baselineMse": 0.6706288972192532, "learnedMse": 0.5363212470376765 }], "textile": [{ "baselineMse": 0.03886194997734891, "learnedMse": 0.028267329402121105 }, { "baselineMse": 0.13002670567791008, "learnedMse": 0.07634093514183253 }, { "baselineMse": 0.13628338646739663, "learnedMse": 0.08200390144012087 }, { "baselineMse": 0.12825029457968354, "learnedMse": 0.07660085050032518 }, { "baselineMse": 0.13093750027248907, "learnedMse": 0.06139592733247127 }], "outlier": [{ "baselineMse": 7.163468380468215, "learnedMse": 0.01097720038155117 }, { "baselineMse": 0.18361817546133224, "learnedMse": 0.037992812254389614 }, { "baselineMse": 0.35905943711151533, "learnedMse": 0.01367031442377618 }, { "baselineMse": 0.3888897899589984, "learnedMse": 0.014447564466957974 }, { "baselineMse": 2.2931409827103164, "learnedMse": 0.011600746766417825 }], "curve": [{ "baselineMse": 0.03886716394737378, "learnedMse": 0.011947195935310732 }, { "baselineMse": 0.01530807448317449, "learnedMse": 0.01615106804158797 }, { "baselineMse": 0.014684316000154922, "learnedMse": 0.01259467372923439 }, { "baselineMse": 0.01551177496051773, "learnedMse": 0.013273063493644096 }, { "baselineMse": 0.015359108932060167, "learnedMse": 0.010830616010844007 }], "detail": [{ "baselineMse": 0.039956582991123264, "learnedMse": 0.04956581920668081 }, { "baselineMse": 0.13698628472348298, "learnedMse": 0.08144314525130558 }, { "baselineMse": 0.14938851587694427, "learnedMse": 0.07879299041060857 }, { "baselineMse": 0.15491516607857378, "learnedMse": 0.09780276232450714 }, { "baselineMse": 0.1730841646566472, "learnedMse": 0.09063297984044764 }] } } });
  var LEARNED_RECONSTRUCTION_CALIBRATION = Object.freeze({ "kind": "supervised-bounded-ordinary-2x-residual-gain", "networkSha256": "1bde25b7fc8e0d1b1ca1965df997a49f40582c8022c183f83b4872cb0e71bcfe", "gain": 0.148383, "residualEnergyPenalty": 1, "trainingSeed": 1380537163, "trainingPatches": 8192, "heldOutSeed": 1707636726, "heldOutPatches": 4096, "scope": "Ordinary 2x residual only; confident isolated-outlier correction is separately bounded.", "training": { "baselineMse": 0.0025196479777989662, "fullResidualMse": 0.0026832237793096656, "calibratedMse": 0.002493066165956189 }, "heldOut": { "baselineMse": 0.0025175566164432174, "fullResidualMse": 0.0026783595474172005, "calibratedMse": 0.0024904115238443343 }, "sha256": "03081f1beff58af8dfb51fb555b28ee7ae67fea02b92006b221d9b080448178f" });
  var LEARNED_ENHANCEMENT_MAX_RADIANCE = 65504;
  var LEARNED_DETAIL_GATE_FEATURES = Object.freeze([
    "gradient-coherence",
    "center-isolation",
    "curvature-ratio",
    "relative-contrast"
  ]);
  var LEARNED_DETAIL_GATE_TRANSITION = Object.freeze({ relativeWidth: 0.04, minimumWidth: 2e-3 });
  var LEARNED_DETAIL_GATE_NODES = Object.freeze([
    { feature: 0, threshold: 1.22921, left: 1, right: 10, gain: 1 },
    { feature: 0, threshold: 1.077014, left: 2, right: 3, gain: 1 },
    { feature: -1, threshold: 0, left: 0, right: 0, gain: 1 },
    { feature: 3, threshold: 0.184417, left: 4, right: 7, gain: 1 },
    { feature: 3, threshold: 0.139105, left: 5, right: 6, gain: 1 },
    { feature: -1, threshold: 0, left: 0, right: 0, gain: 1 },
    { feature: -1, threshold: 0, left: 0, right: 0, gain: 0.09445 },
    { feature: 1, threshold: 0.898988, left: 8, right: 9, gain: 1 },
    { feature: -1, threshold: 0, left: 0, right: 0, gain: 0.017708 },
    { feature: -1, threshold: 0, left: 0, right: 0, gain: 1 },
    { feature: 3, threshold: 0.225801, left: 11, right: 16, gain: 1 },
    { feature: 3, threshold: 0.18924, left: 12, right: 15, gain: 1 },
    { feature: 3, threshold: 0.09962, left: 13, right: 14, gain: 1 },
    { feature: -1, threshold: 0, left: 0, right: 0, gain: 1 },
    { feature: -1, threshold: 0, left: 0, right: 0, gain: 0.840662 },
    { feature: -1, threshold: 0, left: 0, right: 0, gain: 0 },
    { feature: 1, threshold: 0.114351, left: 17, right: 20, gain: 1 },
    { feature: 1, threshold: 0.053675, left: 18, right: 19, gain: 1 },
    { feature: -1, threshold: 0, left: 0, right: 0, gain: 1 },
    { feature: -1, threshold: 0, left: 0, right: 0, gain: 0 },
    { feature: 1, threshold: 0.841165, left: 21, right: 22, gain: 1 },
    { feature: -1, threshold: 0, left: 0, right: 0, gain: 0 },
    { feature: -1, threshold: 0, left: 0, right: 0, gain: 0.171256 }
  ].map((node) => Object.freeze(node)));
  var LEARNED_DETAIL_GATE_TRAINING_EVIDENCE = Object.freeze({
    model: "bounded-supervised-detail-gate-tree-v1",
    scope: "same-resolution correction attenuation only",
    training: Object.freeze({
      seed: 1145394241,
      patches: 16e3,
      maximumDepth: 4,
      heldOutPatches: 4e4,
      smoothWeight: 1e3,
      outlierWeight: 5
    }),
    validation: "Synthetic radiance pairs with capped preset-strength corrections; not a room render or a guarantee of universal improvement.",
    heldOut: Object.freeze({
      smooth: Object.freeze({
        patches: 1e4,
        baselineMse: 0.029201061,
        ungatedMse: 8687454e-9,
        gatedMse: 871597e-8,
        improvementOverUngatedPercent: -0.328,
        changedPatches: 105
      }),
      edge: Object.freeze({
        patches: 1e4,
        baselineMse: 0.026814253,
        ungatedMse: 0.246046357,
        gatedMse: 0.067184692,
        improvementOverUngatedPercent: 72.694,
        changedPatches: 6835
      }),
      textile: Object.freeze({
        patches: 1e4,
        baselineMse: 0.027149761,
        ungatedMse: 0.221015551,
        gatedMse: 0.20596763,
        improvementOverUngatedPercent: 6.809,
        changedPatches: 1492
      }),
      outlier: Object.freeze({
        patches: 1e4,
        baselineMse: 5.613707849,
        ungatedMse: 2.810737217,
        gatedMse: 2.810754752,
        improvementOverUngatedPercent: -1e-3,
        changedPatches: 13
      })
    })
  });
  var LEARNED_DETAIL_GATE_RUNTIME_EVIDENCE = Object.freeze({
    seed: 1852143732,
    patches: 8e3,
    accumulatedSamples: 16,
    noiseEstimate: 0.3,
    validation: "Untouched synthetic holdout through the public presets. Columns are raw, previous-filter, and detail-gated MSE; lower error is better. Some signals remain worse than raw input.",
    presets: Object.freeze({
      balanced: Object.freeze({
        smooth: Object.freeze([0.031788013, 0.010293667, 0.010300206]),
        edge: Object.freeze([0.025949982, 0.164040061, 0.052905137]),
        textile: Object.freeze([0.027063835, 0.15989411, 0.147694942]),
        outlier: Object.freeze([5.364436944, 2.821793469, 2.821796112])
      }),
      "photoreal-detail": Object.freeze({
        smooth: Object.freeze([0.031788013, 0.02242397, 0.022427014]),
        edge: Object.freeze([0.025949982, 0.033143252, 0.026357476]),
        textile: Object.freeze([0.027063835, 0.03347729, 0.032696833]),
        outlier: Object.freeze([5.364436944, 4.721240506, 4.721241595])
      }),
      "artifact-cleanup": Object.freeze({
        smooth: Object.freeze([0.031788013, 5259747e-9, 5267843e-9]),
        edge: Object.freeze([0.025949982, 0.401851082, 0.102338482]),
        textile: Object.freeze([0.027063835, 0.394007848, 0.362235461]),
        outlier: Object.freeze([5.364436944, 1.572136998, 1.572140719])
      })
    })
  });
  var LEARNED_DENOISE_KERNEL = [
    0.079067102,
    0.128090031,
    0.083446914,
    0.118947505,
    0.182648536,
    0.123763115,
    0.078524823,
    0.121982558,
    0.083529416
  ];
  var LEARNED_RECONSTRUCTION_2X_KERNELS = [
    [
      -7116773e-9,
      -0.021109801,
      -0.037467358,
      -0.026827744,
      0.146065792,
      0.264471775,
      -0.029599734,
      0.260650539,
      0.450933304
    ],
    [
      -0.045131082,
      -8460088e-9,
      -0.067050286,
      0.033109427,
      0.258709969,
      0.154892766,
      0.075720946,
      0.368642874,
      0.229565472
    ],
    [
      -0.045459471,
      0.04122841,
      0.060686845,
      -0.018717654,
      0.246198137,
      0.393570526,
      -0.05059598,
      0.137560887,
      0.2355283
    ],
    [
      -0.065509269,
      0.086795122,
      -0.020670246,
      0.08549342,
      0.397243668,
      0.218568289,
      -0.011479907,
      0.214850608,
      0.094708315
    ]
  ];
  var LEARNED_ENHANCEMENT_TRAINING_EVIDENCE = Object.freeze({
    seed: 1279607122,
    trainingPatchesPerObjective: 24e3,
    heldOutPatchesPerObjective: 6e3,
    ridge: 5e-4,
    signalDomain: "log2(1+radiance)",
    modeledCorruption: "deterministic heteroscedastic zero-mean sampling noise",
    sameResolution: Object.freeze({
      baselineMse: 0.022516365,
      learnedMse: 4388102e-9,
      mseReductionPercent: 80.511,
      psnrGainDb: 7.102
    }),
    reconstruction2x: Object.freeze({
      baselineMse: 1055494e-9,
      learnedMse: 900559e-9,
      mseReductionPercent: 14.679,
      psnrGainDb: 0.688
    }),
    cleanBypassMaximumAbsoluteError: 0,
    unsupportedBypassMaximumAbsoluteError: 0
  });
  var DEFAULT_SETTINGS = Object.freeze({
    strength: 0.86,
    cleanNoiseThreshold: 0.012,
    fullNoiseThreshold: 0.08,
    depthSigma: 0.025,
    normalExponent: 24,
    albedoSigma: 0.24,
    roughnessSigma: 0.14,
    minimumConfidence: 0.5,
    fullConfidence: 0.86,
    minimumRoughness: 0.06,
    fullRoughness: 0.28,
    residualLimit: 0.38,
    minimumSupport: 0.32,
    maximumRadiance: LEARNED_ENHANCEMENT_MAX_RADIANCE,
    accumulatedSamples: 1
  });
  var LEARNED_ENHANCEMENT_PRESETS = Object.freeze({
    balanced: Object.freeze({
      strength: 0.68,
      cleanNoiseThreshold: 0.014,
      fullNoiseThreshold: 0.095,
      depthSigma: 0.02,
      normalExponent: 30,
      albedoSigma: 0.18,
      roughnessSigma: 0.11,
      minimumConfidence: 0.56,
      fullConfidence: 0.88,
      minimumRoughness: 0.08,
      fullRoughness: 0.32,
      residualLimit: 0.28,
      minimumSupport: 0.4
    }),
    "photoreal-detail": Object.freeze({
      strength: 0.46,
      cleanNoiseThreshold: 0.018,
      fullNoiseThreshold: 0.12,
      depthSigma: 0.014,
      normalExponent: 48,
      albedoSigma: 0.12,
      roughnessSigma: 0.075,
      minimumConfidence: 0.66,
      fullConfidence: 0.94,
      minimumRoughness: 0.12,
      fullRoughness: 0.42,
      residualLimit: 0.16,
      minimumSupport: 0.52
    }),
    "artifact-cleanup": Object.freeze({
      strength: 0.86,
      cleanNoiseThreshold: 0.012,
      fullNoiseThreshold: 0.08,
      depthSigma: 0.018,
      normalExponent: 32,
      albedoSigma: 0.16,
      roughnessSigma: 0.1,
      minimumConfidence: 0.55,
      fullConfidence: 0.86,
      minimumRoughness: 0.06,
      fullRoughness: 0.28,
      residualLimit: 0.32,
      minimumSupport: 0.4
    })
  });
  var MODE_CODE = Object.freeze({
    off: 0,
    "same-resolution": 1,
    "reconstruct-2x": 2
  });
  var wgslNumbers = (values) => values.map((value) => Number.isInteger(value) ? `${value}.0` : String(value)).join(", ");
  var reconstructionWeights = LEARNED_RECONSTRUCTION_2X_KERNELS.flatMap((kernel) => [...kernel]);
  var detailGateExpressions = LEARNED_DETAIL_GATE_NODES.map((node, index) => {
    if (node.feature < 0) return `  let detail_${index} = vec3<f32>(${wgslNumbers([node.gain])});`;
    const width = Math.max(
      LEARNED_DETAIL_GATE_TRANSITION.minimumWidth,
      Math.abs(node.threshold) * LEARNED_DETAIL_GATE_TRANSITION.relativeWidth
    );
    return `  let detail_${index} = mix(detail_${node.left}, detail_${node.right}, smoothstep(vec3<f32>(${wgslNumbers([node.threshold - width])}), vec3<f32>(${wgslNumbers([node.threshold + width])}), features[${node.feature}]));`;
  }).reverse().join("\n");
  var residualNetworkConstants = LEARNED_RESIDUAL_NETWORK.layers.map((layer, index) => `const RESIDUAL_WEIGHTS_${index}: array<f32, ${layer.weights.length}> = array<f32, ${layer.weights.length}>(${wgslNumbers(layer.weights)});
const RESIDUAL_BIASES_${index}: array<f32, ${layer.size}> = array<f32, ${layer.size}>(${wgslNumbers(layer.biases)});
const DENOISE_RESIDUAL_WEIGHTS_${index}: array<f32, ${layer.weights.length}> = array<f32, ${layer.weights.length}>(${wgslNumbers(LEARNED_DENOISE_RESIDUAL_NETWORK.layers[index].weights)});
const DENOISE_RESIDUAL_BIASES_${index}: array<f32, ${layer.size}> = array<f32, ${layer.size}>(${wgslNumbers(LEARNED_DENOISE_RESIDUAL_NETWORK.layers[index].biases)});`).join("\n");
  var residualNetworkLayers = LEARNED_RESIDUAL_NETWORK.layers.map((layer, index) => {
    const previous = index === 0 ? "features" : `residual_layer_${index - 1}`;
    if (index === LEARNED_RESIDUAL_NETWORK.layers.length - 1) return `
  var result = vec3<f32>(select(RESIDUAL_BIASES_${index}[head], DENOISE_RESIDUAL_BIASES_${index}[head], head == 0u));
  for (var input = 0u; input < ${layer.inputs}u; input += 1u) {
    result += ${previous}[input] * select(RESIDUAL_WEIGHTS_${index}[head * ${layer.inputs}u + input], DENOISE_RESIDUAL_WEIGHTS_${index}[head * ${layer.inputs}u + input], head == 0u);
  }
  return result * scale;`;
    return `
  var residual_layer_${index}: array<vec3<f32>, ${layer.size}>;
  for (var output = 0u; output < ${layer.size}u; output += 1u) {
    var value = vec3<f32>(select(RESIDUAL_BIASES_${index}[output], DENOISE_RESIDUAL_BIASES_${index}[output], head == 0u));
    for (var input = 0u; input < ${layer.inputs}u; input += 1u) {
      value += ${previous}[input] * select(RESIDUAL_WEIGHTS_${index}[output * ${layer.inputs}u + input], DENOISE_RESIDUAL_WEIGHTS_${index}[output * ${layer.inputs}u + input], head == 0u);
    }
    residual_layer_${index}[output] = max(value, vec3<f32>(0.0));
  }`;
  }).join("\n");
  var LEARNED_ENHANCEMENT_SHADER = (
    /* wgsl */
    `
struct LearnedEnhancementParameters {
  extents: vec4<f32>,
  mode_strength_noise: vec4<f32>,
  guide_gates: vec4<f32>,
  surface_gates: vec4<f32>,
  safety: vec4<f32>,
}

@group(0) @binding(0) var<uniform> parameters: LearnedEnhancementParameters;
@group(0) @binding(1) var input_hdr: texture_2d<f32>;
@group(0) @binding(2) var normal_depth: texture_2d<f32>;
@group(0) @binding(3) var albedo_roughness: texture_2d<f32>;
@group(0) @binding(4) var estimator_noise: texture_2d<f32>;
@group(0) @binding(5) var output_hdr: texture_storage_2d<rgba16float, write>;

const DENOISE_KERNEL: array<f32, 9> = array<f32, 9>(${wgslNumbers(LEARNED_DENOISE_KERNEL)});
const RECONSTRUCTION_KERNELS: array<f32, 36> = array<f32, 36>(${wgslNumbers(reconstructionWeights)});
${residualNetworkConstants}

fn infer_residual(values: array<vec3<f32>, 9>, noise: f32, head: u32) -> vec3<f32> {
  var mean = vec3<f32>(0.0);
  for (var index = 0u; index < 9u; index += 1u) { mean += values[index]; }
  mean /= 9.0;
  var variance = vec3<f32>(0.0);
  for (var index = 0u; index < 9u; index += 1u) {
    let delta = values[index] - mean;
    variance += delta * delta;
  }
  let scale = max(vec3<f32>(0.06), sqrt(variance / 9.0));
  var features: array<vec3<f32>, 11>;
  for (var index = 0u; index < 9u; index += 1u) { features[index] = (values[index] - mean) / scale; }
  let noise_reference = exp2(mean) - vec3<f32>(1.0);
  let shared_reference = max(noise_reference.r, max(noise_reference.g, noise_reference.b));
  let sigma = noise * select(noise_reference, vec3<f32>(shared_reference), head > 0u) * exp2(-mean) / 0.6931471805599453;
  features[9] = min(vec3<f32>(3.0), sigma / scale);
  features[10] = mean / (vec3<f32>(1.0) + mean);
${residualNetworkLayers}
}

struct StructureSafety {
  flat: vec3<f32>,
  outlier: vec3<f32>,
  detail: vec3<f32>,
}

fn structure_safety(values: array<vec3<f32>, 9>, noise: f32, reconstruct: bool, base: vec3<f32>) -> StructureSafety {
  var mean = vec3<f32>(0.0);
  for (var index = 0u; index < 9u; index += 1u) { mean += values[index]; }
  mean /= 9.0;
  let sigma = max(vec3<f32>(0.000001), noise * (vec3<f32>(1.0) - exp2(-mean)) / 0.6931471805599453);
  var variance = vec3<f32>(0.0);
  for (var index = 0u; index < 9u; index += 1u) {
    let delta = values[index] - mean;
    variance += delta * delta;
  }
  let deviation = sqrt(variance / 9.0);
  var neighbors: array<vec3<f32>, 9>;
  var isolated_value = values[4];
  for (var index = 0u; index < 9u; index += 1u) {
    neighbors[index] = values[index];
    if (reconstruct) { isolated_value = max(isolated_value, values[index]); }
  }
  if (!reconstruct) { neighbors[4] = vec3<f32>(65504.0); }
  for (var end = 8u; end > 0u; end -= 1u) {
    for (var index = 0u; index < end; index += 1u) {
      let lower = min(neighbors[index], neighbors[index + 1u]);
      let upper = max(neighbors[index], neighbors[index + 1u]);
      neighbors[index] = lower;
      neighbors[index + 1u] = upper;
    }
  }
  var neighbor_mean = vec3<f32>(0.0);
  for (var index = 1u; index < 7u; index += 1u) { neighbor_mean += neighbors[index]; }
  neighbor_mean /= 6.0;
  var neighbor_variance = vec3<f32>(0.0);
  for (var index = 1u; index < 7u; index += 1u) {
    let delta = neighbors[index] - neighbor_mean;
    neighbor_variance += delta * delta;
  }
  let neighbor_deviation = sqrt(neighbor_variance / 6.0);
  let threshold = max(max(vec3<f32>(0.5), sigma * 6.0), neighbor_deviation * 6.0);
  return StructureSafety(
    select(vec3<f32>(1.0) - smoothstep(sigma, sigma * 1.5, deviation), vec3<f32>(0.0), reconstruct),
    smoothstep(threshold, threshold * 1.5, isolated_value - neighbor_mean)
      * (vec3<f32>(1.0) - smoothstep(sigma * 2.0, sigma * 4.0, neighbor_deviation))
      * select(vec3<f32>(1.0), smoothstep(vec3<f32>(0.0), threshold, base - neighbor_mean), reconstruct),
    smoothstep(sigma * 1.25, sigma * 3.0, deviation)
  );
}

fn finite_scalar(value: f32) -> bool {
  return value == value && abs(value) <= ${LEARNED_ENHANCEMENT_MAX_RADIANCE}.0;
}

fn safe_radiance(value: vec3<f32>, maximum_radiance: f32) -> vec3<f32> {
  let finite = (value == value) & (abs(value) <= vec3<f32>(3.402823e38));
  return select(vec3<f32>(0.0), clamp(value, vec3<f32>(0.0), vec3<f32>(maximum_radiance)), finite);
}

fn safe_normal(value: vec3<f32>) -> vec3<f32> {
  let length_squared = dot(value, value);
  if (!all(value == value) || length_squared <= 0.25) {
    return vec3<f32>(0.0);
  }
  return value * inverseSqrt(length_squared);
}

fn compress_hdr(value: vec3<f32>) -> vec3<f32> {
  return log2(vec3<f32>(1.0) + max(value, vec3<f32>(0.0)));
}

fn expand_hdr(value: vec3<f32>, maximum_radiance: f32) -> vec3<f32> {
  return min(vec3<f32>(maximum_radiance), max(vec3<f32>(0.0), exp2(clamp(value, vec3<f32>(0.0), vec3<f32>(16.0))) - vec3<f32>(1.0)));
}

fn smooth_gate(minimum: f32, maximum: f32, value: f32) -> f32 {
  let unit = clamp((value - minimum) / max(maximum - minimum, 0.000001), 0.0, 1.0);
  return unit * unit * (3.0 - 2.0 * unit);
}

fn learned_detail_gate(values: array<vec3<f32>, 9>) -> vec3<f32> {
  var mean = vec3<f32>(0.0);
  for (var index = 0u; index < 9u; index += 1u) {
    mean += values[index];
  }
  mean /= 9.0;
  var variance = vec3<f32>(0.0);
  for (var index = 0u; index < 9u; index += 1u) {
    let delta = values[index] - mean;
    variance += delta * delta;
  }
  variance /= 9.0;
  let denominator = max(variance, vec3<f32>(0.00000001));
  let horizontal = (values[2] + 2.0 * values[5] + values[8] - values[0] - 2.0 * values[3] - values[6]) / 8.0;
  let vertical = (values[6] + 2.0 * values[7] + values[8] - values[0] - 2.0 * values[1] - values[2]) / 8.0;
  var curvature = vec3<f32>(0.0);
  for (var index = 0u; index < 3u; index += 1u) {
    let horizontal_curve = values[index * 3u] - 2.0 * values[index * 3u + 1u] + values[index * 3u + 2u];
    let vertical_curve = values[index] - 2.0 * values[index + 3u] + values[index + 6u];
    curvature += horizontal_curve * horizontal_curve + vertical_curve * vertical_curve;
  }
  let center_delta = values[4] - mean;
  let features = array<vec3<f32>, 4>(
    (horizontal * horizontal + vertical * vertical) / denominator,
    center_delta * center_delta / denominator,
    curvature / (6.0 * denominator),
    sqrt(variance) / max(vec3<f32>(0.02), abs(mean))
  );
${detailGateExpressions}
  return clamp(detail_0, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn clamp_source_pixel(pixel: vec2<i32>) -> vec2<i32> {
  return clamp(pixel, vec2<i32>(0), vec2<i32>(textureDimensions(estimator_noise)) - vec2<i32>(1));
}

fn load_radiance(pixel: vec2<i32>, maximum_radiance: f32) -> vec3<f32> {
  return safe_radiance(textureLoad(estimator_noise, clamp_source_pixel(pixel), 0).rgb, maximum_radiance);
}

fn load_fallback(pixel: vec2<i32>, maximum_radiance: f32) -> vec3<f32> {
  return safe_radiance(textureLoad(input_hdr, clamp_source_pixel(pixel), 0).rgb, maximum_radiance);
}

fn fallback_radiance(position: vec2<f32>, reconstruct: bool, maximum_radiance: f32) -> vec3<f32> {
  if (!reconstruct) { return load_fallback(vec2<i32>(round(position)), maximum_radiance); }
  let base = vec2<i32>(floor(position));
  let fraction = fract(position);
  let top = mix(load_fallback(base, maximum_radiance), load_fallback(base + vec2<i32>(1, 0), maximum_radiance), fraction.x);
  let bottom = mix(load_fallback(base + vec2<i32>(0, 1), maximum_radiance), load_fallback(base + vec2<i32>(1, 1), maximum_radiance), fraction.x);
  return mix(top, bottom, fraction.y);
}

fn bilinear_radiance(source_position: vec2<f32>, maximum_radiance: f32) -> vec3<f32> {
  let base = vec2<i32>(floor(source_position));
  let fraction = fract(source_position);
  let top = mix(load_radiance(base, maximum_radiance), load_radiance(base + vec2<i32>(1, 0), maximum_radiance), fraction.x);
  let bottom = mix(load_radiance(base + vec2<i32>(0, 1), maximum_radiance), load_radiance(base + vec2<i32>(1, 1), maximum_radiance), fraction.x);
  return mix(top, bottom, fraction.y);
}

fn kernel_weight(tap: u32, mode: u32, phase: u32) -> f32 {
  if (mode == 1u) {
    return DENOISE_KERNEL[tap];
  }
  return RECONSTRUCTION_KERNELS[phase * 9u + tap];
}

fn guide_supported(pixel: vec2<i32>) -> bool {
  let clamped_pixel = clamp_source_pixel(pixel);
  let normal_depth_sample = textureLoad(normal_depth, clamped_pixel, 0);
  let surface = textureLoad(albedo_roughness, clamped_pixel, 0);
  let depth = normal_depth_sample.w;
  let albedo = surface.rgb;
  return finite_scalar(depth)
    && depth >= 0.0
    && depth < 65000.0
    && all(albedo == albedo)
    && all(albedo >= vec3<f32>(0.0))
    && all(albedo <= vec3<f32>(1.0))
    && surface.w >= 0.0
    && surface.w <= 1.0
    && dot(safe_normal(normal_depth_sample.xyz), safe_normal(normal_depth_sample.xyz)) > 0.5;
}

fn guide_compatibility(reference_pixel: vec2<i32>, sample_pixel: vec2<i32>) -> f32 {
  if (!guide_supported(sample_pixel)) {
    return 0.0;
  }
  let reference_coordinate = clamp_source_pixel(reference_pixel);
  let sample_coordinate = clamp_source_pixel(sample_pixel);
  let reference_normal_depth = textureLoad(normal_depth, reference_coordinate, 0);
  let sample_normal_depth = textureLoad(normal_depth, sample_coordinate, 0);
  let reference_surface = textureLoad(albedo_roughness, reference_coordinate, 0);
  let sample_surface = textureLoad(albedo_roughness, sample_coordinate, 0);
  let reference_depth = reference_normal_depth.w;
  let sample_depth = sample_normal_depth.w;
  let reference_albedo = reference_surface.rgb;
  let sample_albedo = sample_surface.rgb;
  let depth_weight = exp(-abs(sample_depth - reference_depth) / max(abs(reference_depth), 0.001) / parameters.guide_gates.x);
  let normal_weight = pow(max(dot(safe_normal(reference_normal_depth.xyz), safe_normal(sample_normal_depth.xyz)), 0.0), parameters.guide_gates.y);
  let albedo_weight = exp(-length(sample_albedo - reference_albedo) / parameters.guide_gates.z);
  let roughness_weight = exp(-abs(sample_surface.w - reference_surface.w) / parameters.guide_gates.w);
  return depth_weight * normal_weight * albedo_weight * roughness_weight;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let output_size = vec2<u32>(parameters.extents.zw);
  if (any(invocation.xy >= output_size)) {
    return;
  }
  let source_size = parameters.extents.xy;
  let output_pixel = vec2<i32>(invocation.xy);
  let source_position = (vec2<f32>(invocation.xy) + vec2<f32>(0.5)) * source_size / parameters.extents.zw - vec2<f32>(0.5);
  let base_pixel = vec2<i32>(floor(source_position));
  let mode = u32(parameters.mode_strength_noise.x + 0.5);
  let maximum_radiance = parameters.safety.z;
  let base_radiance = select(
    load_radiance(vec2<i32>(round(source_position)), maximum_radiance),
    bilinear_radiance(source_position, maximum_radiance),
    mode == 2u
  );
  let loaded_alpha = textureLoad(estimator_noise, clamp_source_pixel(vec2<i32>(round(source_position))), 0).a;
  let base_alpha = select(0.0, clamp(loaded_alpha, 0.0, 1.0), loaded_alpha == loaded_alpha);
  let fallback = fallback_radiance(source_position, mode == 2u, maximum_radiance);
  if (mode == 0u || parameters.mode_strength_noise.y <= 0.0) {
    textureStore(output_hdr, output_pixel, vec4<f32>(fallback, base_alpha));
    return;
  }

  let fraction = fract(source_position);
  let phase = (invocation.x & 1u) + ((invocation.y & 1u) << 1u);
  let reference_pixel = select(
    vec2<i32>(round(source_position)),
    base_pixel + vec2<i32>(select(0, 1, fraction.x >= 0.5), select(0, 1, fraction.y >= 0.5)),
    mode == 2u
  );
  let estimator = textureLoad(estimator_noise, clamp_source_pixel(reference_pixel), 0);
  let raw_noise_estimate = estimator.a;
  let noise_estimate = raw_noise_estimate / sqrt(max(parameters.safety.w, 1.0));
  let sample_confidence = 1.0 - exp(-parameters.safety.w / 8.0);
  if (!guide_supported(reference_pixel)
      || !finite_scalar(raw_noise_estimate)
      || raw_noise_estimate < 0.0
      || raw_noise_estimate > 1.0) {
    textureStore(output_hdr, output_pixel, vec4<f32>(fallback, base_alpha));
    return;
  }
  if (noise_estimate <= parameters.mode_strength_noise.z) {
    textureStore(output_hdr, output_pixel, vec4<f32>(base_radiance, base_alpha));
    return;
  }
  let reference_surface = textureLoad(albedo_roughness, clamp_source_pixel(reference_pixel), 0);
  if (sample_confidence <= parameters.surface_gates.x || reference_surface.w <= parameters.surface_gates.z) {
    textureStore(output_hdr, output_pixel, vec4<f32>(fallback, base_alpha));
    return;
  }

  var filtered = vec3<f32>(0.0);
  var detail_values: array<vec3<f32>, 9>;
  var detail_supported = true;
  var weight_sum = 0.0;
  var support = 0.0;
  var compatible_taps = 0u;
  var local_minimum = base_radiance;
  var local_maximum = base_radiance;
  for (var tap = 0u; tap < 9u; tap += 1u) {
    let offset = vec2<i32>(i32(tap % 3u) - 1, i32(tap / 3u) - 1);
    let sample_pixel = select(reference_pixel + offset, base_pixel + offset, mode == 2u);
    let compatibility = guide_compatibility(reference_pixel, sample_pixel);
    if (compatibility < 0.95) {
      detail_supported = false;
    }
    if (compatibility <= 0.0) {
      continue;
    }
    let static_weight = kernel_weight(tap, mode, phase);
    let weight = static_weight * compatibility;
    let radiance = load_radiance(sample_pixel, maximum_radiance);
    let compressed = compress_hdr(radiance);
    detail_values[tap] = compressed;
    filtered += compressed * weight;
    weight_sum += weight;
    support += abs(static_weight) * compatibility;
    if (compatibility >= 0.12) {
      local_minimum = min(local_minimum, radiance);
      local_maximum = max(local_maximum, radiance);
    }
    if (compatibility >= 0.25) {
      compatible_taps += 1u;
    }
  }
  if (!detail_supported || abs(weight_sum) < 0.15 || support < parameters.safety.y || compatible_taps < 3u) {
    textureStore(output_hdr, output_pixel, vec4<f32>(fallback, base_alpha));
    return;
  }

  var candidate = expand_hdr(filtered / weight_sum, maximum_radiance);
  var structure = StructureSafety(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(1.0));
  if (detail_supported) { structure = structure_safety(detail_values, noise_estimate, mode == 2u, compress_hdr(base_radiance)); }
  if (detail_supported) {
    let head = select(0u, phase + 1u, mode == 2u);
    let reconstructed = expand_hdr(compress_hdr(base_radiance) + infer_residual(detail_values, noise_estimate, head), maximum_radiance);
    candidate = mix(reconstructed, candidate, structure.flat);
  }
  candidate = clamp(candidate, local_minimum, local_maximum);
  let residual_limit = max(vec3<f32>(0.002), abs(base_radiance) * max(vec3<f32>(parameters.safety.x), 0.96 * structure.outlier));
  candidate = clamp(candidate, base_radiance - residual_limit, base_radiance + residual_limit);
  let noise_gate = smooth_gate(parameters.mode_strength_noise.z, parameters.mode_strength_noise.w, noise_estimate);
  let confidence_gate = smooth_gate(parameters.surface_gates.x, parameters.surface_gates.y, sample_confidence);
  let roughness_gate = smooth_gate(parameters.surface_gates.z, parameters.surface_gates.w, reference_surface.w);
  let support_gate = smooth_gate(parameters.safety.y, parameters.safety.y * 1.8, support);
  let quality_gate = clamp(parameters.mode_strength_noise.y * noise_gate * confidence_gate * roughness_gate * support_gate, 0.0, 1.0);
  let supported = confidence_gate * roughness_gate * support_gate;
  let boosted = parameters.mode_strength_noise.y * (vec3<f32>(noise_gate) + (1.0 - noise_gate) * structure.flat * 0.8) * supported;
  var structure_gate = select(vec3<f32>(1.0), vec3<f32>(1.0) - 0.7 * structure.detail, mode == 1u && detail_supported);
  if (mode == 2u && detail_supported) { structure_gate = vec3<f32>(${LEARNED_RECONSTRUCTION_CALIBRATION.gain}); }
  let amount = clamp(max(min(boosted, structure_gate), structure.outlier * 0.96 * supported * noise_gate), vec3<f32>(0.0), vec3<f32>(1.0));
  let result = safe_radiance(mix(base_radiance, candidate, amount), maximum_radiance);
  textureStore(output_hdr, output_pixel, vec4<f32>(result, base_alpha));
}
`
  );

  // src/renderer/post/experimental-frame-interpolation.ts
  var EXPERIMENTAL_OPTICAL_FLOW_FALLBACK_SHADER = (
    /* wgsl */
    `
struct ExperimentalFrameParameters {
  dimensions_fraction_motion: vec4<f32>,
  rejection: vec4<f32>,
  search: vec4<f32>,
}

@group(0) @binding(0) var<uniform> parameters: ExperimentalFrameParameters;
@group(0) @binding(1) var previous_display_color: texture_2d<f32>;
@group(0) @binding(2) var current_display_color: texture_2d<f32>;
@group(0) @binding(3) var output_current_to_previous_flow: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var output_disocclusion_mask: texture_storage_2d<rgba8unorm, write>;

// The radius-two Lucas-Kanade window reads one extra texel for its gradients.
// Cache that 14x14 source tile once per 8x8 workgroup instead of issuing the
// same texture loads independently for every overlapping pixel window.
const FLOW_WORKGROUP_WIDTH: u32 = 8u;
const FLOW_TILE_HALO: i32 = 3;
const FLOW_TILE_WIDTH: u32 = 14u;
const FLOW_TILE_TEXELS: u32 = FLOW_TILE_WIDTH * FLOW_TILE_WIDTH;
var<workgroup> previous_luma_tile: array<f32, 196>;
var<workgroup> current_luma_tile: array<f32, 196>;

fn luminance(color: vec3<f32>) -> f32 {
  return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn clamped_pixel(pixel: vec2<i32>) -> vec2<i32> {
  let dimensions = vec2<i32>(textureDimensions(current_display_color));
  return clamp(pixel, vec2<i32>(0), dimensions - vec2<i32>(1));
}

fn previous_image_luma(pixel: vec2<i32>) -> f32 {
  return luminance(textureLoad(previous_display_color, clamped_pixel(pixel), 0).rgb);
}

fn current_image_luma(pixel: vec2<i32>) -> f32 {
  return luminance(textureLoad(current_display_color, clamped_pixel(pixel), 0).rgb);
}

fn flow_tile_index(pixel: vec2<i32>) -> u32 {
  return u32(pixel.y) * FLOW_TILE_WIDTH + u32(pixel.x);
}

fn previous_luma(pixel: vec2<i32>) -> f32 {
  return previous_luma_tile[flow_tile_index(pixel)];
}

fn current_luma(pixel: vec2<i32>) -> f32 {
  return current_luma_tile[flow_tile_index(pixel)];
}

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_index) local_index: u32,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let dimensions = textureDimensions(output_current_to_previous_flow);
  let workgroup_origin = vec2<i32>(workgroup_id.xy * vec2<u32>(FLOW_WORKGROUP_WIDTH));
  for (var tile_index = local_index; tile_index < FLOW_TILE_TEXELS; tile_index += 64u) {
    let tile_pixel = vec2<i32>(
      i32(tile_index % FLOW_TILE_WIDTH),
      i32(tile_index / FLOW_TILE_WIDTH)
    );
    let source_pixel = workgroup_origin + tile_pixel - vec2<i32>(FLOW_TILE_HALO);
    previous_luma_tile[tile_index] = previous_image_luma(source_pixel);
    current_luma_tile[tile_index] = current_image_luma(source_pixel);
  }
  workgroupBarrier();

  if (any(global_id.xy >= dimensions)) {
    return;
  }

  let center = vec2<i32>(global_id.xy);
  let tile_center = center - workgroup_origin + vec2<i32>(FLOW_TILE_HALO);
  let radius = clamp(i32(parameters.search.x), 1, 2);
  var xx = 0.0;
  var xy = 0.0;
  var yy = 0.0;
  var xt = 0.0;
  var yt = 0.0;

  // Local Lucas-Kanade solve. It is deliberately only a fallback approximation.
  for (var offset_y = -2; offset_y <= 2; offset_y += 1) {
    for (var offset_x = -2; offset_x <= 2; offset_x += 1) {
      if (abs(offset_x) > radius || abs(offset_y) > radius) {
        continue;
      }
      let pixel = tile_center + vec2<i32>(offset_x, offset_y);
      let gradient_x = 0.25 * (
        previous_luma(pixel + vec2<i32>(1, 0)) - previous_luma(pixel - vec2<i32>(1, 0))
        + current_luma(pixel + vec2<i32>(1, 0)) - current_luma(pixel - vec2<i32>(1, 0))
      );
      let gradient_y = 0.25 * (
        previous_luma(pixel + vec2<i32>(0, 1)) - previous_luma(pixel - vec2<i32>(0, 1))
        + current_luma(pixel + vec2<i32>(0, 1)) - current_luma(pixel - vec2<i32>(0, 1))
      );
      let temporal_difference = current_luma(pixel) - previous_luma(pixel);
      xx += gradient_x * gradient_x;
      xy += gradient_x * gradient_y;
      yy += gradient_y * gradient_y;
      xt += gradient_x * temporal_difference;
      yt += gradient_y * temporal_difference;
    }
  }

  let determinant = xx * yy - xy * xy;
  let epsilon = parameters.search.z;
  var flow_pixels = vec2<f32>(0.0);
  if (determinant > epsilon) {
    // Solves gradient dot flow = current - previous, yielding current-to-previous displacement.
    flow_pixels = vec2<f32>(yy * xt - xy * yt, xx * yt - xy * xt) / determinant;
  }
  let flow_length = length(flow_pixels);
  if (flow_length > parameters.dimensions_fraction_motion.w) {
    flow_pixels *= parameters.dimensions_fraction_motion.w / max(flow_length, 1e-6);
  }
  let trace = xx + yy;
  let confidence = clamp(
    determinant / max(trace * trace + epsilon, epsilon) * parameters.search.w,
    0.0,
    1.0
  );
  let flow_uv = flow_pixels / vec2<f32>(dimensions);
  let previous_pixel = center + vec2<i32>(round(flow_pixels));
  let previous_in_bounds = all(previous_pixel >= vec2<i32>(0))
    && all(previous_pixel < vec2<i32>(dimensions));
  let warped_previous_luma = previous_image_luma(previous_pixel);
  let center_current_luma = current_luma(tile_center);
  let photometric_residual = abs(center_current_luma - warped_previous_luma)
    / max(max(center_current_luma, warped_previous_luma), 0.05);
  let unreliable_change = photometric_residual * (1.0 - confidence * 0.5);
  let disocclusion = select(clamp(unreliable_change, 0.0, 1.0), 1.0, !previous_in_bounds);
  textureStore(
    output_current_to_previous_flow,
    center,
    vec4<f32>(flow_uv, confidence, 1.0)
  );
  textureStore(output_disocclusion_mask, center, vec4<f32>(disocclusion, 0.0, 0.0, 1.0));
}
`
  );
  function createExperimentalFrameInterpolationShader(format) {
    return (
      /* wgsl */
      `
struct ExperimentalFrameParameters {
  dimensions_fraction_motion: vec4<f32>,
  rejection: vec4<f32>,
  search: vec4<f32>,
}

@group(0) @binding(0) var<uniform> parameters: ExperimentalFrameParameters;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var previous_display_color: texture_2d<f32>;
@group(0) @binding(3) var current_display_color: texture_2d<f32>;
@group(0) @binding(4) var game_current_to_previous_motion: texture_2d<f32>;
@group(0) @binding(5) var fallback_current_to_previous_flow: texture_2d<f32>;
@group(0) @binding(6) var previous_depth: texture_2d<f32>;
@group(0) @binding(7) var current_depth: texture_2d<f32>;
@group(0) @binding(8) var disocclusion_mask: texture_2d<f32>;
@group(0) @binding(9) var output_interpolated_color: texture_storage_2d<${format}, write>;

fn in_bounds(uv: vec2<f32>) -> bool {
  return all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
}

fn depth_at(depth_texture: texture_2d<f32>, uv: vec2<f32>) -> f32 {
  let dimensions = vec2<i32>(textureDimensions(depth_texture));
  let pixel = clamp(vec2<i32>(uv * vec2<f32>(dimensions)), vec2<i32>(0), dimensions - vec2<i32>(1));
  return textureLoad(depth_texture, pixel, 0).w;
}

fn sample_at_uv(sampled_texture: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
  let dimensions = vec2<i32>(textureDimensions(sampled_texture));
  let pixel = clamp(vec2<i32>(uv * vec2<f32>(dimensions)), vec2<i32>(0), dimensions - vec2<i32>(1));
  return textureLoad(sampled_texture, pixel, 0);
}

fn clamp_motion_uv(motion: vec2<f32>, dimensions: vec2<u32>) -> vec2<f32> {
  var motion_pixels = motion * vec2<f32>(dimensions);
  let motion_length = length(motion_pixels);
  let maximum_motion = parameters.dimensions_fraction_motion.w;
  if (motion_length > maximum_motion) {
    motion_pixels *= maximum_motion / max(motion_length, 1e-6);
  }
  return motion_pixels / vec2<f32>(dimensions);
}

fn inpaint_hole(uv: vec2<f32>, factor: f32) -> vec3<f32> {
  let dimensions = vec2<i32>(textureDimensions(current_display_color));
  let center = vec2<i32>(uv * vec2<f32>(dimensions));
  let radius = clamp(i32(parameters.search.y), 1, 2);
  var accumulated = vec3<f32>(0.0);
  var accumulated_weight = 0.0;
  for (var offset_y = -2; offset_y <= 2; offset_y += 1) {
    for (var offset_x = -2; offset_x <= 2; offset_x += 1) {
      if (abs(offset_x) > radius || abs(offset_y) > radius) {
        continue;
      }
      let pixel = clamp(center + vec2<i32>(offset_x, offset_y), vec2<i32>(0), dimensions - vec2<i32>(1));
      let sample_uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(dimensions);
      let sample_mask = textureLoad(disocclusion_mask, pixel, 0).x;
      if (sample_mask < parameters.rejection.w) {
        let previous = textureSampleLevel(previous_display_color, linear_sampler, sample_uv, 0.0).rgb;
        let current = textureSampleLevel(current_display_color, linear_sampler, sample_uv, 0.0).rgb;
        let distance_weight = 1.0 / (1.0 + length(vec2<f32>(vec2<i32>(offset_x, offset_y))));
        accumulated += mix(previous, current, factor) * distance_weight;
        accumulated_weight += distance_weight;
      }
    }
  }
  if (accumulated_weight <= 1e-6) {
    return textureSampleLevel(current_display_color, linear_sampler, uv, 0.0).rgb;
  }
  return accumulated / accumulated_weight;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let dimensions = textureDimensions(output_interpolated_color);
  if (any(global_id.xy >= dimensions)) {
    return;
  }

  let pixel = vec2<i32>(global_id.xy);
  let target_uv = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) / vec2<f32>(dimensions);
  let factor = parameters.dimensions_fraction_motion.z;
  let game_motion_sample = sample_at_uv(game_current_to_previous_motion, target_uv);
  let fallback_flow_sample = sample_at_uv(fallback_current_to_previous_flow, target_uv);
  let game_motion_valid = game_motion_sample.w > parameters.rejection.y;
  var motion = vec2<f32>(0.0);
  if (game_motion_valid) {
    motion = clamp_motion_uv(game_motion_sample.xy, dimensions);
  } else if (fallback_flow_sample.z >= parameters.rejection.z) {
    motion = clamp_motion_uv(fallback_flow_sample.xy, dimensions);
  }

  // A game vector maps current UV to previous UV. Backward warping splits it around the target time.
  let previous_uv = target_uv + motion * factor;
  let current_uv = target_uv - motion * (1.0 - factor);
  // Zero-confidence fallback is treated as zero motion, which is correct for static/flat regions.
  var previous_valid = in_bounds(previous_uv);
  var current_valid = in_bounds(current_uv);
  let clamped_previous_uv = clamp(previous_uv, vec2<f32>(0.0), vec2<f32>(1.0));
  let clamped_current_uv = clamp(current_uv, vec2<f32>(0.0), vec2<f32>(1.0));
  let previous_depth_value = depth_at(previous_depth, clamped_previous_uv);
  let current_depth_value = depth_at(current_depth, clamped_current_uv);
  let current_depth_in_previous_view = select(
    current_depth_value,
    game_motion_sample.w,
    game_motion_valid
  );
  let relative_depth_difference = abs(previous_depth_value - current_depth_in_previous_view)
    / max(abs(current_depth_in_previous_view), 1e-5);
  let explicit_disocclusion = textureLoad(disocclusion_mask, pixel, 0).x >= parameters.rejection.w;
  let depth_disocclusion = relative_depth_difference > parameters.rejection.x;

  if (depth_disocclusion || explicit_disocclusion) {
    let depth_epsilon = parameters.rejection.x
      * max(max(abs(previous_depth_value), abs(current_depth_in_previous_view)), 1e-5);
    if (previous_depth_value + depth_epsilon < current_depth_in_previous_view) {
      current_valid = false;
    } else if (current_depth_in_previous_view + depth_epsilon < previous_depth_value) {
      previous_valid = false;
    } else {
      previous_valid = false;
      current_valid = false;
    }
  }

  var previous_weight = select(0.0, 1.0 - factor, previous_valid);
  var current_weight = select(0.0, factor, current_valid);
  // Preserve an endpoint even when the supplied motion has no confidence.
  if (factor <= 0.0 && in_bounds(target_uv)) {
    previous_weight = 1.0;
  }
  if (factor >= 1.0 && in_bounds(target_uv)) {
    current_weight = 1.0;
  }

  let weight_sum = previous_weight + current_weight;
  var interpolated = vec3<f32>(0.0);
  if (weight_sum > 1e-6) {
    let previous = textureSampleLevel(previous_display_color, linear_sampler, clamped_previous_uv, 0.0).rgb;
    let current = textureSampleLevel(current_display_color, linear_sampler, clamped_current_uv, 0.0).rgb;
    interpolated = (previous * previous_weight + current * current_weight) / weight_sum;
  } else {
    interpolated = inpaint_hole(target_uv, factor);
  }

  textureStore(output_interpolated_color, pixel, vec4<f32>(interpolated, 1.0));
}
`
    );
  }
  var EXPERIMENTAL_FRAME_INTERPOLATION_RGBA16FLOAT_SHADER = createExperimentalFrameInterpolationShader("rgba16float");
  var EXPERIMENTAL_FRAME_INTERPOLATION_RGBA8UNORM_SHADER = createExperimentalFrameInterpolationShader("rgba8unorm");

  // src/renderer/post/fsr1.ts
  function createFsr1EasuShader(format) {
    return (
      /* wgsl */
      `
// f32 WGSL translation of AMD FidelityFX Super Resolution 1 EASU.
// The complete upstream MIT notice is exported as AMD_FSR1_MIT_NOTICE and stored in NOTICE.fsr1.txt.
struct FsrEasuConstants {
  con0: vec4<f32>,
  con1: vec4<f32>,
  con2: vec4<f32>,
  con3: vec4<f32>,
}

struct EasuEdge {
  direction: vec2<f32>,
  length: f32,
}

struct EasuAccumulation {
  color: vec3<f32>,
  weight: f32,
}

@group(0) @binding(0) var<uniform> constants: FsrEasuConstants;
@group(0) @binding(1) var input_color: texture_2d<f32>;
@group(0) @binding(2) var output_color: texture_storage_2d<${format}, write>;

fn load_input(pixel: vec2<i32>) -> vec3<f32> {
  let dimensions = vec2<i32>(textureDimensions(input_color));
  return textureLoad(input_color, clamp(pixel, vec2<i32>(0), dimensions - vec2<i32>(1)), 0).rgb;
}

fn easu_luma(color: vec3<f32>) -> f32 {
  return color.b * 0.5 + (color.r * 0.5 + color.g);
}

fn easu_set(
  edge: EasuEdge,
  subpixel: vec2<f32>,
  corner: u32,
  luma_a: f32,
  luma_b: f32,
  luma_c: f32,
  luma_d: f32,
  luma_e: f32
) -> EasuEdge {
  var weight = 0.0;
  if (corner == 0u) {
    weight = (1.0 - subpixel.x) * (1.0 - subpixel.y);
  } else if (corner == 1u) {
    weight = subpixel.x * (1.0 - subpixel.y);
  } else if (corner == 2u) {
    weight = (1.0 - subpixel.x) * subpixel.y;
  } else {
    weight = subpixel.x * subpixel.y;
  }

  let dc = luma_d - luma_c;
  let cb = luma_c - luma_b;
  let length_x_denominator = max(abs(dc), abs(cb));
  let direction_x = luma_d - luma_b;
  var length_x = 0.0;
  if (length_x_denominator > 1e-8) {
    length_x = clamp(abs(direction_x) / length_x_denominator, 0.0, 1.0);
  }

  let ec = luma_e - luma_c;
  let ca = luma_c - luma_a;
  let length_y_denominator = max(abs(ec), abs(ca));
  let direction_y = luma_e - luma_a;
  var length_y = 0.0;
  if (length_y_denominator > 1e-8) {
    length_y = clamp(abs(direction_y) / length_y_denominator, 0.0, 1.0);
  }

  var result = edge;
  result.direction += vec2<f32>(direction_x, direction_y) * weight;
  result.length += (length_x * length_x + length_y * length_y) * weight;
  return result;
}

fn easu_tap(
  accumulation: EasuAccumulation,
  offset: vec2<f32>,
  direction: vec2<f32>,
  anisotropic_length: vec2<f32>,
  lobe: f32,
  clipping_point: f32,
  color: vec3<f32>
) -> EasuAccumulation {
  var rotated = vec2<f32>(
    offset.x * direction.x + offset.y * direction.y,
    offset.x * -direction.y + offset.y * direction.x
  );
  rotated *= anisotropic_length;
  let distance_squared = min(dot(rotated, rotated), clipping_point);

  var base = (2.0 / 5.0) * distance_squared - 1.0;
  var window = lobe * distance_squared - 1.0;
  base *= base;
  window *= window;
  base = (25.0 / 16.0) * base - (25.0 / 16.0 - 1.0);
  let weight = base * window;

  var result = accumulation;
  result.color += color * weight;
  result.weight += weight;
  return result;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let output_dimensions = textureDimensions(output_color);
  if (any(global_id.xy >= output_dimensions)) {
    return;
  }

  var subpixel = vec2<f32>(global_id.xy) * constants.con0.xy + constants.con0.zw;
  let base_pixel = vec2<i32>(floor(subpixel));
  subpixel -= floor(subpixel);

  // Same 12 taps and naming as FsrEasuF in ffx_fsr1.h.
  let b = load_input(base_pixel + vec2<i32>( 0, -1));
  let c = load_input(base_pixel + vec2<i32>( 1, -1));
  let e = load_input(base_pixel + vec2<i32>(-1,  0));
  let f = load_input(base_pixel + vec2<i32>( 0,  0));
  let g = load_input(base_pixel + vec2<i32>( 1,  0));
  let h = load_input(base_pixel + vec2<i32>( 2,  0));
  let i = load_input(base_pixel + vec2<i32>(-1,  1));
  let j = load_input(base_pixel + vec2<i32>( 0,  1));
  let k = load_input(base_pixel + vec2<i32>( 1,  1));
  let l = load_input(base_pixel + vec2<i32>( 2,  1));
  let n = load_input(base_pixel + vec2<i32>( 0,  2));
  let o = load_input(base_pixel + vec2<i32>( 1,  2));

  let b_luma = easu_luma(b);
  let c_luma = easu_luma(c);
  let e_luma = easu_luma(e);
  let f_luma = easu_luma(f);
  let g_luma = easu_luma(g);
  let h_luma = easu_luma(h);
  let i_luma = easu_luma(i);
  let j_luma = easu_luma(j);
  let k_luma = easu_luma(k);
  let l_luma = easu_luma(l);
  let n_luma = easu_luma(n);
  let o_luma = easu_luma(o);

  var edge = EasuEdge(vec2<f32>(0.0), 0.0);
  edge = easu_set(edge, subpixel, 0u, b_luma, e_luma, f_luma, g_luma, j_luma);
  edge = easu_set(edge, subpixel, 1u, c_luma, f_luma, g_luma, h_luma, k_luma);
  edge = easu_set(edge, subpixel, 2u, f_luma, i_luma, j_luma, k_luma, n_luma);
  edge = easu_set(edge, subpixel, 3u, g_luma, j_luma, k_luma, l_luma, o_luma);

  let direction_squared = dot(edge.direction, edge.direction);
  if (direction_squared < 1.0 / 32768.0) {
    edge.direction = vec2<f32>(1.0, 0.0);
  } else {
    edge.direction *= inverseSqrt(direction_squared);
  }
  edge.length = edge.length * 0.5;
  edge.length *= edge.length;
  let stretch = 1.0 / max(abs(edge.direction.x), abs(edge.direction.y));
  let anisotropic_length = vec2<f32>(
    1.0 + (stretch - 1.0) * edge.length,
    1.0 - 0.5 * edge.length
  );
  let lobe = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * edge.length;
  let clipping_point = 1.0 / lobe;

  let minimum_nearest = min(min(f, g), min(j, k));
  let maximum_nearest = max(max(f, g), max(j, k));
  var accumulation = EasuAccumulation(vec3<f32>(0.0), 0.0);
  accumulation = easu_tap(accumulation, vec2<f32>( 0.0, -1.0) - subpixel, edge.direction, anisotropic_length, lobe, clipping_point, b);
  accumulation = easu_tap(accumulation, vec2<f32>( 1.0, -1.0) - subpixel, edge.direction, anisotropic_length, lobe, clipping_point, c);
  accumulation = easu_tap(accumulation, vec2<f32>(-1.0,  1.0) - subpixel, edge.direction, anisotropic_length, lobe, clipping_point, i);
  accumulation = easu_tap(accumulation, vec2<f32>( 0.0,  1.0) - subpixel, edge.direction, anisotropic_length, lobe, clipping_point, j);
  accumulation = easu_tap(accumulation, vec2<f32>( 0.0,  0.0) - subpixel, edge.direction, anisotropic_length, lobe, clipping_point, f);
  accumulation = easu_tap(accumulation, vec2<f32>(-1.0,  0.0) - subpixel, edge.direction, anisotropic_length, lobe, clipping_point, e);
  accumulation = easu_tap(accumulation, vec2<f32>( 1.0,  1.0) - subpixel, edge.direction, anisotropic_length, lobe, clipping_point, k);
  accumulation = easu_tap(accumulation, vec2<f32>( 2.0,  1.0) - subpixel, edge.direction, anisotropic_length, lobe, clipping_point, l);
  accumulation = easu_tap(accumulation, vec2<f32>( 2.0,  0.0) - subpixel, edge.direction, anisotropic_length, lobe, clipping_point, h);
  accumulation = easu_tap(accumulation, vec2<f32>( 1.0,  0.0) - subpixel, edge.direction, anisotropic_length, lobe, clipping_point, g);
  accumulation = easu_tap(accumulation, vec2<f32>( 1.0,  2.0) - subpixel, edge.direction, anisotropic_length, lobe, clipping_point, o);
  accumulation = easu_tap(accumulation, vec2<f32>( 0.0,  2.0) - subpixel, edge.direction, anisotropic_length, lobe, clipping_point, n);

  let resolved = clamp(
    accumulation.color / max(accumulation.weight, 1e-8),
    minimum_nearest,
    maximum_nearest
  );
  textureStore(output_color, vec2<i32>(global_id.xy), vec4<f32>(resolved, 1.0));
}
`
    );
  }
  function createFsr1RcasShader(format) {
    return (
      /* wgsl */
      `
// f32 WGSL translation of AMD FidelityFX Super Resolution 1 RCAS.
// The complete upstream MIT notice is exported as AMD_FSR1_MIT_NOTICE and stored in NOTICE.fsr1.txt.
struct FsrRcasConstants {
  values: vec4<f32>,
}

@group(0) @binding(0) var<uniform> constants: FsrRcasConstants;
@group(0) @binding(1) var input_color: texture_2d<f32>;
@group(0) @binding(2) var output_color: texture_storage_2d<${format}, write>;

const RCAS_LIMIT: f32 = 0.25 - 1.0 / 16.0;

fn load_input(pixel: vec2<i32>) -> vec4<f32> {
  let dimensions = vec2<i32>(textureDimensions(input_color));
  return textureLoad(input_color, clamp(pixel, vec2<i32>(0), dimensions - vec2<i32>(1)), 0);
}

fn safe_scalar_ratio(numerator: f32, denominator: f32) -> f32 {
  if (abs(denominator) <= 1e-8) {
    return 0.0;
  }
  return numerator / denominator;
}

fn safe_ratio(numerator: vec3<f32>, denominator: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    safe_scalar_ratio(numerator.x, denominator.x),
    safe_scalar_ratio(numerator.y, denominator.y),
    safe_scalar_ratio(numerator.z, denominator.z)
  );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let output_dimensions = textureDimensions(output_color);
  if (any(global_id.xy >= output_dimensions)) {
    return;
  }

  let pixel = vec2<i32>(global_id.xy);
  let b = load_input(pixel + vec2<i32>( 0, -1)).rgb;
  let d = load_input(pixel + vec2<i32>(-1,  0)).rgb;
  let center = load_input(pixel);
  let e = center.rgb;
  let f = load_input(pixel + vec2<i32>( 1,  0)).rgb;
  let h = load_input(pixel + vec2<i32>( 0,  1)).rgb;

  let minimum_ring = min(min(b, d), min(f, h));
  let maximum_ring = max(max(b, d), max(f, h));
  let hit_minimum = safe_ratio(min(minimum_ring, e), 4.0 * maximum_ring);
  let hit_maximum = safe_ratio(vec3<f32>(1.0) - max(maximum_ring, e), 4.0 * minimum_ring - vec3<f32>(4.0));
  let lobe_channels = max(-hit_minimum, hit_maximum);
  let lobe = max(
    -RCAS_LIMIT,
    min(max(lobe_channels.r, max(lobe_channels.g, lobe_channels.b)), 0.0)
  ) * constants.values.x;
  let reciprocal_lobe = 1.0 / (4.0 * lobe + 1.0);
  let resolved = (lobe * (b + d + h + f) + e) * reciprocal_lobe;
  textureStore(output_color, pixel, vec4<f32>(resolved, center.a));
}
`
    );
  }
  var FSR1_EASU_RGBA16FLOAT_SHADER = createFsr1EasuShader("rgba16float");
  var FSR1_EASU_RGBA8UNORM_SHADER = createFsr1EasuShader("rgba8unorm");
  var FSR1_RCAS_RGBA16FLOAT_SHADER = createFsr1RcasShader("rgba16float");
  var FSR1_RCAS_RGBA8UNORM_SHADER = createFsr1RcasShader("rgba8unorm");

  // src/contracts.ts
  var CAMERA_F_STOP_RANGE = Object.freeze({ minimum: 1, maximum: 25 });
  var DEFAULT_OUTPUT_STYLE_SETTINGS = Object.freeze({
    mode: "photoreal",
    toonBands: 5,
    toonOutlineStrength: 0.7,
    pixelSize: 4,
    pixelPalette: "rgb-332",
    pixelDitherStrength: 0,
    monochromeTint: Object.freeze([1, 1, 1]),
    monochromeContrast: 1
  });
  var DEFAULT_RENDER_SETTINGS = {
    mode: "path-trace",
    outputStyle: {
      ...DEFAULT_OUTPUT_STYLE_SETTINGS,
      monochromeTint: [...DEFAULT_OUTPUT_STYLE_SETTINGS.monochromeTint]
    },
    antiAliasing: "temporal",
    toneMapping: "aces",
    renderScale: 1,
    automaticTextureLod: false,
    samplesPerFrame: 1,
    minBounces: 2,
    maxBounces: 4,
    lightSamples: 1,
    rayDiffuseWeight: 1,
    rayGlossyWeight: 1,
    rayTransparentWeight: 1,
    rayVolumeWeight: 1,
    noiseThreshold: 0.015,
    convergenceMinSamples: 8,
    exposure: 1,
    cameraFStop: 14,
    cameraWhiteBalance: [1, 1, 1],
    cameraShadowLiftStops: 0,
    cameraHighlightProtection: 0,
    fogDensity: 0,
    fogColor: [0.65, 0.75, 0.9],
    fogHeightFalloff: 0.08,
    fogAnisotropy: 0.2,
    volumetricSteps: 16,
    denoising: "fast",
    experimentalLearnedEnhancement: "off",
    experimentalLearnedEnhancementPreset: "photoreal-detail",
    upscaling: "native",
    frameGeneration: "off",
    bloomStrength: 0,
    bloomThreshold: 1,
    filmGrainStrength: 0,
    filmGrainScale: 1,
    chromaticAberrationPixels: 0,
    chromaticAberrationEdgeStart: 0.35,
    chromaticAberrationFalloff: 0.65,
    autoExposure: false,
    hdrOutput: "sdr",
    hdrEnvironmentSunMatching: false
  };

  // src/renderer/post/output-style.ts
  var OUTPUT_STYLE_LIMITS = Object.freeze({
    toonBands: Object.freeze({ minimum: 2, maximum: 16 }),
    toonOutlineStrength: Object.freeze({ minimum: 0, maximum: 1 }),
    pixelSize: Object.freeze({ minimum: 1, maximum: 64 }),
    pixelDitherStrength: Object.freeze({ minimum: 0, maximum: 1 }),
    monochromeTint: Object.freeze({ minimum: 0, maximum: 4 }),
    monochromeContrast: Object.freeze({ minimum: 0.25, maximum: 4 })
  });
  var STYLE_INDEX = Object.freeze({
    photoreal: 0,
    toon: 1,
    "pixel-art": 2,
    monochrome: 3
  });
  var PALETTE_INDEX = Object.freeze({
    "rgb-332": 0,
    "pico-8": 1,
    gameboy: 2,
    "grayscale-8": 3
  });
  var OUTPUT_STYLE_SHADER = (
    /* wgsl */
    `
struct OutputStyleParameters {
  sizes: vec4<f32>,
  controls: vec4<f32>,
  pixel_monochrome: vec4<f32>,
  monochrome_tint: vec4<f32>,
}

@group(0) @binding(0) var<uniform> parameters: OutputStyleParameters;
@group(0) @binding(1) var display_source: texture_2d<f32>;
@group(0) @binding(2) var normal_depth: texture_2d<f32>;
@group(0) @binding(3) var display_output: texture_storage_2d<rgba16float, write>;

const LUMINANCE_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

fn source_pixel_for_output(pixel: vec2<u32>, output_dimensions: vec2<u32>) -> vec2<i32> {
  let source_dimensions = textureDimensions(display_source);
  let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(output_dimensions);
  return vec2<i32>(min(
    vec2<u32>(uv * vec2<f32>(source_dimensions)),
    source_dimensions - vec2<u32>(1u)
  ));
}

fn guide_pixel_for_output(pixel: vec2<u32>, output_dimensions: vec2<u32>) -> vec2<i32> {
  let guide_dimensions = textureDimensions(normal_depth);
  let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) / vec2<f32>(output_dimensions);
  return vec2<i32>(min(
    vec2<u32>(uv * vec2<f32>(guide_dimensions)),
    guide_dimensions - vec2<u32>(1u)
  ));
}

fn clamped_guide_pixel(pixel: vec2<i32>) -> vec2<i32> {
  return clamp(pixel, vec2<i32>(0), vec2<i32>(textureDimensions(normal_depth)) - vec2<i32>(1));
}

fn guide_discontinuity(center: vec4<f32>, sample_value: vec4<f32>) -> f32 {
  let center_sky = center.w >= 65000.0;
  let sample_sky = sample_value.w >= 65000.0;
  if (center_sky != sample_sky) {
    return 1.0;
  }
  if (center_sky) {
    return 0.0;
  }
  let center_normal = normalize(select(vec3<f32>(0.0, 0.0, 1.0), center.xyz, dot(center.xyz, center.xyz) > 0.0001));
  let sample_normal = normalize(select(vec3<f32>(0.0, 0.0, 1.0), sample_value.xyz, dot(sample_value.xyz, sample_value.xyz) > 0.0001));
  let normal_edge = 1.0 - clamp(dot(center_normal, sample_normal), 0.0, 1.0);
  let depth_edge = abs(center.w - sample_value.w) / max(min(center.w, sample_value.w), 0.1);
  return max(smoothstep(0.035, 0.24, depth_edge), smoothstep(0.08, 0.42, normal_edge));
}

fn toon_outline(pixel: vec2<u32>, output_dimensions: vec2<u32>) -> f32 {
  let guide_pixel = guide_pixel_for_output(pixel, output_dimensions);
  let center = textureLoad(normal_depth, guide_pixel, 0);
  let left = textureLoad(normal_depth, clamped_guide_pixel(guide_pixel + vec2<i32>(-1, 0)), 0);
  let right = textureLoad(normal_depth, clamped_guide_pixel(guide_pixel + vec2<i32>(1, 0)), 0);
  let up = textureLoad(normal_depth, clamped_guide_pixel(guide_pixel + vec2<i32>(0, -1)), 0);
  let down = textureLoad(normal_depth, clamped_guide_pixel(guide_pixel + vec2<i32>(0, 1)), 0);
  return max(
    max(guide_discontinuity(center, left), guide_discontinuity(center, right)),
    max(guide_discontinuity(center, up), guide_discontinuity(center, down))
  );
}

fn toon_color(color: vec3<f32>, pixel: vec2<u32>, output_dimensions: vec2<u32>) -> vec3<f32> {
  let bounded = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
  let luminance = dot(bounded, LUMINANCE_WEIGHTS);
  let band_count = max(parameters.controls.y, 2.0);
  let banded_luminance = floor(luminance * (band_count - 1.0) + 0.5) / (band_count - 1.0);
  let chroma = bounded / max(luminance, 0.0001);
  let banded = clamp(chroma * banded_luminance, vec3<f32>(0.0), vec3<f32>(1.0));
  let outline = toon_outline(pixel, output_dimensions) * parameters.controls.z;
  return banded * mix(1.0, 0.08, clamp(outline, 0.0, 1.0));
}

fn bayer_4x4(pixel: vec2<u32>) -> f32 {
  let x = pixel.x & 3u;
  let y = pixel.y & 3u;
  var value = 0u;
  if (y == 0u) {
    value = array<u32, 4>(0u, 8u, 2u, 10u)[x];
  } else if (y == 1u) {
    value = array<u32, 4>(12u, 4u, 14u, 6u)[x];
  } else if (y == 2u) {
    value = array<u32, 4>(3u, 11u, 1u, 9u)[x];
  } else {
    value = array<u32, 4>(15u, 7u, 13u, 5u)[x];
  }
  return (f32(value) + 0.5) / 16.0 - 0.5;
}

fn closest_pico8(color: vec3<f32>) -> vec3<f32> {
  let palette = array<vec3<f32>, 16>(
    vec3<f32>(0.0000, 0.0000, 0.0000), vec3<f32>(0.1137, 0.1686, 0.3255),
    vec3<f32>(0.4941, 0.1451, 0.3255), vec3<f32>(0.0000, 0.5294, 0.3176),
    vec3<f32>(0.6706, 0.3216, 0.2118), vec3<f32>(0.3725, 0.3412, 0.3098),
    vec3<f32>(0.7608, 0.7647, 0.7804), vec3<f32>(1.0000, 0.9451, 0.9098),
    vec3<f32>(1.0000, 0.0000, 0.3020), vec3<f32>(1.0000, 0.6392, 0.0000),
    vec3<f32>(1.0000, 0.9255, 0.1529), vec3<f32>(0.0000, 0.8941, 0.2118),
    vec3<f32>(0.1608, 0.6784, 1.0000), vec3<f32>(0.5137, 0.4627, 0.6118),
    vec3<f32>(1.0000, 0.4667, 0.6588), vec3<f32>(1.0000, 0.8000, 0.6667)
  );
  var closest = palette[0];
  var closest_distance = dot(color - closest, color - closest);
  for (var index = 1u; index < 16u; index += 1u) {
    let candidate = palette[index];
    let candidate_distance = dot(color - candidate, color - candidate);
    if (candidate_distance < closest_distance) {
      closest = candidate;
      closest_distance = candidate_distance;
    }
  }
  return closest;
}

fn closest_gameboy(color: vec3<f32>) -> vec3<f32> {
  let palette = array<vec3<f32>, 4>(
    vec3<f32>(0.0588, 0.2196, 0.0588),
    vec3<f32>(0.1882, 0.3843, 0.1882),
    vec3<f32>(0.5451, 0.6745, 0.0588),
    vec3<f32>(0.6078, 0.7373, 0.0588)
  );
  var closest = palette[0];
  var closest_distance = dot(color - closest, color - closest);
  for (var index = 1u; index < 4u; index += 1u) {
    let candidate = palette[index];
    let candidate_distance = dot(color - candidate, color - candidate);
    if (candidate_distance < closest_distance) {
      closest = candidate;
      closest_distance = candidate_distance;
    }
  }
  return closest;
}

fn pixel_palette(color: vec3<f32>, logical_pixel: vec2<u32>) -> vec3<f32> {
  let palette_index = u32(parameters.pixel_monochrome.x);
  let dither = bayer_4x4(logical_pixel) * parameters.pixel_monochrome.y * 0.14;
  let adjusted = clamp(color + vec3<f32>(dither), vec3<f32>(0.0), vec3<f32>(1.0));
  if (palette_index == 1u) {
    return closest_pico8(adjusted);
  }
  if (palette_index == 2u) {
    return closest_gameboy(adjusted);
  }
  if (palette_index == 3u) {
    let luminance = clamp(dot(adjusted, LUMINANCE_WEIGHTS), 0.0, 1.0);
    let level = floor(luminance * 7.0 + 0.5) / 7.0;
    return vec3<f32>(level);
  }
  return floor(adjusted * vec3<f32>(7.0, 7.0, 3.0) + vec3<f32>(0.5))
    / vec3<f32>(7.0, 7.0, 3.0);
}

fn pixel_art_color(pixel: vec2<u32>, output_dimensions: vec2<u32>) -> vec3<f32> {
  let pixel_size = max(u32(parameters.controls.w), 1u);
  let logical_pixel = pixel / pixel_size;
  let cell_origin = logical_pixel * pixel_size;
  let sample_pixel = min(
    cell_origin + vec2<u32>(pixel_size / 2u),
    output_dimensions - vec2<u32>(1u)
  );
  let sampled = textureLoad(display_source, source_pixel_for_output(sample_pixel, output_dimensions), 0).rgb;
  return pixel_palette(clamp(sampled, vec3<f32>(0.0), vec3<f32>(1.0)), logical_pixel);
}

fn monochrome_color(color: vec3<f32>) -> vec3<f32> {
  let bounded = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
  let luminance = dot(bounded, LUMINANCE_WEIGHTS);
  let contrasted = clamp(
    (luminance - 0.5) * parameters.pixel_monochrome.z + 0.5,
    0.0,
    1.0
  );
  return clamp(contrasted * parameters.monochrome_tint.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let output_dimensions = textureDimensions(display_output);
  if (any(global_id.xy >= output_dimensions)) {
    return;
  }
  let source_pixel = source_pixel_for_output(global_id.xy, output_dimensions);
  let source_value = textureLoad(display_source, source_pixel, 0);
  let mode = u32(parameters.controls.x);
  if (mode == 0u) {
    textureStore(display_output, vec2<i32>(global_id.xy), source_value);
    return;
  }
  var styled = source_value.rgb;
  if (mode == 1u) {
    styled = toon_color(styled, global_id.xy, output_dimensions);
  } else if (mode == 2u) {
    styled = pixel_art_color(global_id.xy, output_dimensions);
  } else {
    styled = monochrome_color(styled);
  }
  textureStore(display_output, vec2<i32>(global_id.xy), vec4<f32>(styled, source_value.a));
}
`
  );

  // src/renderer/post/temporal-upscale.ts
  var TEMPORAL_REPROJECTION_UPSCALE_SHADER = (
    /* wgsl */
    `
struct TemporalParameters {
  sizes: vec4<f32>,
  history: vec4<f32>,
  response: vec4<f32>,
}

@group(0) @binding(0) var<uniform> parameters: TemporalParameters;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var current_color: texture_2d<f32>;
@group(0) @binding(3) var previous_color: texture_2d<f32>;
@group(0) @binding(4) var current_to_previous_motion: texture_2d<f32>;
@group(0) @binding(5) var current_surface: texture_2d<f32>;
@group(0) @binding(6) var previous_surface: texture_2d<f32>;
@group(0) @binding(7) var output_color: texture_storage_2d<rgba16float, write>;

const TEMPORAL_WORKGROUP_WIDTH: u32 = 8u;
const TEMPORAL_TILE_WIDTH: u32 = 10u;
const TEMPORAL_TILE_TEXEL_COUNT: u32 = TEMPORAL_TILE_WIDTH * TEMPORAL_TILE_WIDTH;
var<workgroup> current_color_tile: array<vec4<f32>, 100>;

fn rgb_to_ycocg(color: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    dot(color, vec3<f32>(0.25, 0.5, 0.25)),
    dot(color, vec3<f32>(0.5, 0.0, -0.5)),
    dot(color, vec3<f32>(-0.25, 0.5, -0.25))
  );
}

fn ycocg_to_rgb(color: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    color.x + color.y - color.z,
    color.x + color.z,
    color.x - color.y - color.z
  );
}

fn safe_normal(value: vec3<f32>) -> vec3<f32> {
  let length_squared = dot(value, value);
  if (length_squared <= 1e-8) {
    return vec3<f32>(0.0, 0.0, 1.0);
  }
  return value * inverseSqrt(length_squared);
}

fn source_pixel(uv: vec2<f32>) -> vec2<i32> {
  let dimensions = vec2<i32>(textureDimensions(current_color));
  return clamp(vec2<i32>(uv * vec2<f32>(dimensions)), vec2<i32>(0), dimensions - vec2<i32>(1));
}

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_index) local_index: u32,
  @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
  let output_dimensions = textureDimensions(output_color);
  let source_dimensions_u32 = textureDimensions(current_color);
  let source_dimensions = vec2<i32>(source_dimensions_u32);
  let use_shared_neighborhood = all(source_dimensions_u32 <= output_dimensions);
  let group_output_origin = workgroup_id.xy * vec2<u32>(TEMPORAL_WORKGROUP_WIDTH);
  let group_first_uv = (vec2<f32>(group_output_origin) + vec2<f32>(0.5))
    / vec2<f32>(output_dimensions);
  let tile_origin = source_pixel(group_first_uv) - vec2<i32>(1);

  if (use_shared_neighborhood) {
    for (var tile_index = local_index; tile_index < TEMPORAL_TILE_TEXEL_COUNT; tile_index += 64u) {
      let tile_offset = vec2<i32>(
        i32(tile_index % TEMPORAL_TILE_WIDTH),
        i32(tile_index / TEMPORAL_TILE_WIDTH)
      );
      let sample_pixel = clamp(
        tile_origin + tile_offset,
        vec2<i32>(0),
        source_dimensions - vec2<i32>(1)
      );
      current_color_tile[tile_index] = textureLoad(current_color, sample_pixel, 0);
    }
  }
  workgroupBarrier();

  if (any(global_id.xy >= output_dimensions)) {
    return;
  }

  let current_uv = (vec2<f32>(global_id.xy) + vec2<f32>(0.5)) / vec2<f32>(output_dimensions);
  let current_pixel = source_pixel(current_uv);
  let current = textureSampleLevel(current_color, linear_sampler, current_uv, 0.0).rgb;
  let motion_sample = textureLoad(current_to_previous_motion, current_pixel, 0);
  let motion = motion_sample.xy;

  // Motion is normalized UV displacement from the current frame to history.
  let previous_uv = current_uv + motion;
  let previous_in_bounds = all(previous_uv >= vec2<f32>(0.0)) && all(previous_uv <= vec2<f32>(1.0));
  let clamped_previous_uv = clamp(previous_uv, vec2<f32>(0.0), vec2<f32>(1.0));
  let current_surface_dimensions = vec2<i32>(textureDimensions(current_surface));
  let previous_surface_dimensions = vec2<i32>(textureDimensions(previous_surface));
  let current_surface_pixel = clamp(
    vec2<i32>(current_uv * vec2<f32>(current_surface_dimensions)),
    vec2<i32>(0),
    current_surface_dimensions - vec2<i32>(1)
  );
  let previous_surface_pixel = clamp(
    vec2<i32>(clamped_previous_uv * vec2<f32>(previous_surface_dimensions)),
    vec2<i32>(0),
    previous_surface_dimensions - vec2<i32>(1)
  );
  let current_surface_value = textureLoad(current_surface, current_surface_pixel, 0);
  let previous_surface_value = textureLoad(previous_surface, previous_surface_pixel, 0);
  let current_depth_value = current_surface_value.w;
  let previous_depth_value = previous_surface_value.w;
  let expected_previous_depth = motion_sample.w;
  let relative_depth_difference = abs(previous_depth_value - expected_previous_depth)
    / max(abs(expected_previous_depth), parameters.response.z);
  let current_normal_value = safe_normal(current_surface_value.xyz);
  let previous_normal_value = safe_normal(previous_surface_value.xyz);
  let surfaces_match = current_depth_value > 0.0
    && expected_previous_depth > 0.0
    && relative_depth_difference <= parameters.history.y
    && dot(current_normal_value, previous_normal_value) >= parameters.history.z;

  var neighborhood_min = vec3<f32>(1e30);
  var neighborhood_max = vec3<f32>(-1e30);
  for (var offset_y = -1; offset_y <= 1; offset_y += 1) {
    for (var offset_x = -1; offset_x <= 1; offset_x += 1) {
      let sample_pixel = clamp(current_pixel + vec2<i32>(offset_x, offset_y), vec2<i32>(0), source_dimensions - vec2<i32>(1));
      let tile_pixel = sample_pixel - tile_origin;
      let tile_contains_sample = all(tile_pixel >= vec2<i32>(0))
        && all(tile_pixel < vec2<i32>(i32(TEMPORAL_TILE_WIDTH)));
      var sample_color: vec3<f32>;
      if (use_shared_neighborhood && tile_contains_sample) {
        let tile_index = u32(tile_pixel.y) * TEMPORAL_TILE_WIDTH + u32(tile_pixel.x);
        sample_color = current_color_tile[tile_index].rgb;
      } else {
        sample_color = textureLoad(current_color, sample_pixel, 0).rgb;
      }
      let sample_value = rgb_to_ycocg(sample_color);
      neighborhood_min = min(neighborhood_min, sample_value);
      neighborhood_max = max(neighborhood_max, sample_value);
    }
  }

  let neighborhood_range = neighborhood_max - neighborhood_min;
  let clamp_margin = neighborhood_range * parameters.history.w + vec3<f32>(parameters.response.z);
  let history_color = textureSampleLevel(previous_color, linear_sampler, clamped_previous_uv, 0.0).rgb;
  let clamped_history = ycocg_to_rgb(clamp(
    rgb_to_ycocg(history_color),
    neighborhood_min - clamp_margin,
    neighborhood_max + clamp_margin
  ));

  let motion_in_pixels = length(motion * vec2<f32>(output_dimensions));
  var history_weight = parameters.history.x * exp(-motion_in_pixels * parameters.response.x);
  let color_reactivity = length(current - clamped_history) * parameters.response.y;
  let explicit_reactivity = motion_sample.z * parameters.response.y;
  let reactivity = clamp(max(color_reactivity, explicit_reactivity), 0.0, 1.0);
  history_weight *= 1.0 - reactivity;
  if (parameters.response.w < 0.5 || !previous_in_bounds || motion_sample.w <= 0.0 || !surfaces_match) {
    history_weight = 0.0;
  }

  let resolved = mix(current, clamped_history, history_weight);
  textureStore(output_color, vec2<i32>(global_id.xy), vec4<f32>(resolved, 1.0));
}
`
  );

  // src/renderer/post/volumetrics.ts
  var HALF_RES_VOLUMETRIC_SHADER = (
    /* wgsl */
    `
struct VolumetricParameters {
  inverse_view_projection: mat4x4<f32>,
  camera_position_max_distance: vec4<f32>,
  light_direction_anisotropy: vec4<f32>,
  light_radiance_intensity: vec4<f32>,
  fog_albedo_density: vec4<f32>,
  fog_parameters: vec4<f32>,
  resolutions: vec4<f32>,
}

@group(0) @binding(0) var<uniform> parameters: VolumetricParameters;
@group(0) @binding(1) var scene_depth: texture_2d<f32>;
@group(0) @binding(2) var light_visibility: texture_2d<f32>;
@group(0) @binding(3) var output_scattering_transmittance: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.141592653589793;
const MAX_VOLUMETRIC_STEPS: u32 = 64u;

fn henyey_greenstein(cos_theta: f32, anisotropy: f32) -> f32 {
  let g_squared = anisotropy * anisotropy;
  let denominator = max(1.0 + g_squared - 2.0 * anisotropy * cos_theta, 1e-6);
  return (1.0 - g_squared) / (4.0 * PI * pow(denominator, 1.5));
}

fn reconstruct_world(uv: vec2<f32>, device_depth: f32) -> vec3<f32> {
  let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, device_depth, 1.0);
  let world = parameters.inverse_view_projection * clip;
  return world.xyz / max(abs(world.w), 1e-6) * sign(world.w);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let output_dimensions = textureDimensions(output_scattering_transmittance);
  if (any(global_id.xy >= output_dimensions)) {
    return;
  }

  let full_dimensions = textureDimensions(scene_depth);
  let full_pixel = min(global_id.xy * vec2<u32>(2u) + vec2<u32>(1u), full_dimensions - vec2<u32>(1u));
  let uv = (vec2<f32>(full_pixel) + vec2<f32>(0.5)) / vec2<f32>(full_dimensions);
  let linear_depth = textureLoad(scene_depth, vec2<i32>(full_pixel), 0).w;
  let camera_position = parameters.camera_position_max_distance.xyz;
  let far_world = reconstruct_world(uv, 1.0);
  let ray_direction = normalize(far_world - camera_position);

  let ray_distance = min(
    parameters.camera_position_max_distance.w,
    max(linear_depth, 0.0)
  );

  let step_count = min(max(u32(parameters.fog_parameters.w), 1u), MAX_VOLUMETRIC_STEPS);
  let step_length = ray_distance / f32(step_count);
  let visibility = clamp(textureLoad(light_visibility, vec2<i32>(full_pixel), 0).x, 0.0, 1.0);
  let phase = henyey_greenstein(
    clamp(dot(ray_direction, parameters.light_direction_anisotropy.xyz), -1.0, 1.0),
    parameters.light_direction_anisotropy.w
  );
  let light_radiance = parameters.light_radiance_intensity.xyz
    * parameters.light_radiance_intensity.w * visibility;

  var scattering = vec3<f32>(0.0);
  var transmittance = 1.0;
  for (var step_index = 0u; step_index < MAX_VOLUMETRIC_STEPS; step_index += 1u) {
    if (step_index >= step_count || transmittance < 1e-4) {
      break;
    }

    let sample_distance = (f32(step_index) + 0.5) * step_length;
    let sample_position = camera_position + ray_direction * sample_distance;
    let height_density = exp(-parameters.fog_parameters.y
      * max(sample_position.y - parameters.fog_parameters.z, 0.0));
    let local_density = parameters.fog_albedo_density.w * height_density;
    let sigma_t = max(local_density * parameters.fog_parameters.x, 0.0);
    let sigma_s = local_density * parameters.fog_albedo_density.xyz;
    let segment_transmittance = exp(-sigma_t * step_length);
    var integrated_segment = step_length;
    if (sigma_t > 1e-6) {
      integrated_segment = (1.0 - segment_transmittance) / sigma_t;
    }
    scattering += transmittance * sigma_s * light_radiance * phase * integrated_segment;
    transmittance *= segment_transmittance;
  }

  textureStore(
    output_scattering_transmittance,
    vec2<i32>(global_id.xy),
    vec4<f32>(scattering, transmittance)
  );
}
`
  );

  // tests/browser/shader-catalog.ts
  var browserGlobal = globalThis;
  browserGlobal.PrioShaderCatalog = {
    advancedRayTracer: ADVANCED_RAY_TRACER_SHADER,
    advancedCinematicPathTracer: ADVANCED_CINEMATIC_PATH_TRACER_SHADER,
    animatedTriangleDeformation: ANIMATED_TRIANGLE_DEFORMATION_SHADER,
    atrousDenoise: ATROUS_DENOISE_SHADER,
    temporalUpscale: TEMPORAL_REPROJECTION_UPSCALE_SHADER,
    fsr1Easu: FSR1_EASU_RGBA16FLOAT_SHADER,
    fsr1Rcas: FSR1_RCAS_RGBA16FLOAT_SHADER,
    experimentalOpticalFlow: EXPERIMENTAL_OPTICAL_FLOW_FALLBACK_SHADER,
    experimentalFrameInterpolation: EXPERIMENTAL_FRAME_INTERPOLATION_RGBA16FLOAT_SHADER,
    halfResolutionVolumetrics: HALF_RES_VOLUMETRIC_SHADER,
    outputStyle: OUTPUT_STYLE_SHADER,
    autoExposure: AUTO_EXPOSURE_SHADER,
    displayPostProcess: DISPLAY_POST_PROCESS_SHADER,
    photographicEffects: PHOTOGRAPHIC_EFFECTS_SHADER
  };
})();
/*!
 * AMD FidelityFX Super Resolution 1
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
