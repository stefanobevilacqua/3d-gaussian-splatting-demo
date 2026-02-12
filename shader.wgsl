const quad_positions: array<vec3<f32>, 4> = array<vec3<f32>, 4>(
    vec3<f32>(-1.0, 1.0, 0.0),
    vec3<f32>(-1.0, -1.0, 0.0),
    vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(1.0, -1.0, 0.0),
);

struct GaussianSplat {
    position: vec3<f32>,
    rotation: mat3x3<f32>,
    scale: vec3<f32>,
    color: vec3<f32>,
    opacity: f32
};

struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) opacity: f32
}

@group(0) @binding(0) var<storage, read> splats: array<GaussianSplat>;
@group(0) @binding(1) var<uniform> mvp: mat4x4<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> VsOut {
    let splat = splats[instance_id];
    var out: VsOut;
    out.position = mvp * vec4<f32>(quad_positions[vertex_id] * splat.rotation * splat.scale + splat.position, 1.0);
    out.uv = (quad_positions[vertex_id].xy * 0.5) + vec2<f32>(0.5);
    out.color = splat.color;
    out.opacity = splat.opacity;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    let alpha = in.opacity * exp(-r2 * 4.0);
    return vec4<f32>(in.color, alpha);
    // return vec4<f32>(in.color, 1.0);
}
