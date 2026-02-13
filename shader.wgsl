const quad_positions: array<vec3<f32>, 4> = array<vec3<f32>, 4>(
    vec3<f32>(-1.0, 1.0, 0.0),
    vec3<f32>(-1.0, -1.0, 0.0),
    vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(1.0, -1.0, 0.0),
);

struct Gaussian {
    position: vec3<f32>,
    rotation: mat3x3<f32>,
    scale: vec3<f32>,
    color: vec3<f32>,
    opacity: f32
};

struct VsOut {
    @builtin(position) pixel_pos: vec4<f32>,
    @location(0) position: vec2<f32>,
    @location(1) @interpolate(flat) projected_p: vec2<f32>,
    @location(2) @interpolate(flat) color: vec3<f32>,
    @location(3) @interpolate(flat) opacity: f32,
    @location(4) @interpolate(flat) inv_cov_0: vec2<f32>,
    @location(5) @interpolate(flat) inv_cov_1: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(1) var<uniform> cameraView: mat4x4<f32>;
@group(0) @binding(2) var<uniform> aspectRatio: f32;

@vertex
fn vs_main(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> VsOut {
    let gaussian: Gaussian = gaussians[instance_id];

    // Split the view matrix into a 3x3 rotation matrix (W) and a translation vector (t)
    let W: mat3x3<f32> = mat3x3<f32>(cameraView[0].xyz, cameraView[1].xyz, cameraView[2].xyz);
    let t: vec3<f32> = cameraView[3].xyz;

    // Center of the gaussian in camera space
    let p: vec3<f32> = W * gaussian.position + t;

    // The 3x3 covariance Σ can be obtained with the rotation matrix and the scale of the gaussian
    // (formula (6) of the paper)
    let R = gaussian.rotation;
    let S = mat3x3<f32>(gaussian.scale[0], 0, 0,
			0, gaussian.scale[1], 0,
			0, 0, gaussian.scale[2]);
    let Sigma3x3 = R*S*transpose(S)*transpose(R);

    // Center of the gaussian in screen space (projected p) = [p.x * k/ar/p.z, p.y * k/p.z]
    let k: f32 = -1.0 / tan(radians(30.0));
    let pp = vec2<f32>(p.x * k/aspectRatio/p.z, p.y * k/p.z);

    // The 2x2 covariance Σ' can be obtained by projecting Σ to the screen using the jacobian J
    // of the perspective transform function (formula (5) of the paper)
    // The jacobian is the partial derivative of the projected p in dp.x, dp.y and dp.z
    // [dpp.x/dp.x, dpp.x/dp.y, dpp.x/dp.z
    //  dpp.y/dp.x, dpp.y/dp.y, dpp.y/dp.z]
    let J: mat3x2<f32> = mat3x2<f32>(
	vec2<f32>(k/p.z/aspectRatio, 0.0),
	vec2<f32>(0.0, k/p.z),
	vec2<f32>(p.x * -k/aspectRatio/p.z/p.z, p.y * -k/p.z/p.z));
    // let A: mat3x2<f32> = J * W * gaussian.rotation;
    // let s2 = gaussian.scale*gaussian.scale;
    // let cov: mat2x2<f32> = A * mat3x3<f32>(s2[0], 0, 0,
    // 					   0, s2[1], 0,
    // 					   0, 0, s2[2]) * transpose(A);
    let cov: mat2x2<f32> = J * W * Sigma3x3 * transpose(W) * transpose(J);

    // Extent of the 2D quad containing the ellipse
    let r = vec2<f32>(3.0*sqrt(cov[0][0]), 3.0*sqrt(cov[1][1]));

    var out: VsOut;
    out.position = (quad_positions[vertex_id].xy * r) + pp;
    out.pixel_pos = vec4<f32>(out.position, 0.0, 1.0);
    out.projected_p = pp;
    let d = max(0.0, determinant(cov));
    out.inv_cov_0 = 1/d * vec2<f32>(cov[1][1], -cov[1][0]);
    out.inv_cov_1 = 1/d * vec2<f32>(-cov[0][1], cov[0][0]);
    out.color = gaussian.color;
    out.opacity = gaussian.opacity;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // formula (4) of the paper
    // gauss(pos) = exp(-0.5 * transpose(pos-cen) * inv_cov * (pos-cen))
    let d = in.position - in.projected_p;
    let m = mat2x2<f32>(in.inv_cov_0, in.inv_cov_1);
    let gauss = exp(-0.5 * dot(d, m * d));
    let alpha = in.opacity * gauss;
    return vec4<f32>(in.color, alpha);
}
