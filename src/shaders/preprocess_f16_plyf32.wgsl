enable f16;

struct GeneralInfo{
  keys_size : u32, dispatch_x: u32, dispatch_y: u32, dispatch_z: u32,
  l0_x : u32, l0_y : u32, l0_z : u32, l0_t : u32,
  l1_x : u32, l1_y : u32, l1_z : u32, l1_t : u32,
};

const WG_SIZE = 256u;

const PI: f32 = acos(-1.0);

const SH_C0: f16 = f16(sqrt(1.0 / (4.0 * PI)));

const SH_C1: f16 = f16(sqrt(3.0 / (4.0 * PI)));

const SH_C2: array<f16, 5> = array<f16, 5>(
    f16(sqrt(15.0 / (4.0 * PI))),
    f16(-sqrt(15.0 / (4.0 * PI))),
    f16(sqrt(5.0 / (16.0 * PI))),
    f16(-sqrt(15.0 / (4.0 * PI))),
    f16(sqrt(15.0 / (16.0 * PI)))
);

const SH_C3: array<f16, 7> = array<f16, 7>(
    f16(-sqrt(35.0 / (32.0 * PI))),
    f16(sqrt(105.0 / (4.0 * PI))),
    f16(-sqrt(21.0 / (32.0 * PI))),
    f16(sqrt(7.0 / (16.0 * PI))),
    f16(-sqrt(21.0 / (32.0 * PI))),
    f16(sqrt(105.0 / (16.0 * PI))),
    f16(-sqrt(35.0 / (32.0 * PI)))
);

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct SHSolver {
    dir_xy: vec2<f16>,
    dir_z_opacity: vec2<f16>,
    idx: u32,
};

struct RenderSettings {
    canvas_size: vec2<u32>,
    max_sh_deg: u32,
    cur_sh_deg: u32,
    gaussian_scaling: f32,
    kernel_size: f32,
    mip_spatting: u32,
    walltime: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> render_settings: RenderSettings;

@group(1) @binding(0) var<storage, read> sort_infos: GeneralInfo;
@group(1) @binding(1) var<storage, read> sh_coefs : array<f32>;
@group(1) @binding(2) var<storage, read> sh_solvers : array<SHSolver>;
@group(1) @binding(3) var<storage, read_write> colors : array<vec4<f32>>;

fn sh_coef(idx: u32) -> vec3<f16> {
    return vec3<f16>(f16(sh_coefs[idx * 3u + 0u]), f16(sh_coefs[idx * 3u + 1u]), f16(sh_coefs[idx * 3u + 2u]));
}

fn evaluate_sh(dir: vec3<f16>, v_idx: u32, max_sh_deg: u32, sh_deg: u32) -> vec3<f16> {
    let sh_base = (max_sh_deg + 1u) * (max_sh_deg + 1u) * v_idx;

    var result = SH_C0 * sh_coef(sh_base + 0u);

    if sh_deg > 0u {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += SH_C1 * ( -y * sh_coef(sh_base + 1u)
                            +z * sh_coef(sh_base + 2u)
                            -x * sh_coef(sh_base + 3u) );

        if sh_deg > 1u {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let yz = y * z;
            let xz = x * z;

            result += SH_C2[0] * xy * sh_coef(sh_base + 4u) +
                      SH_C2[1] * yz * sh_coef(sh_base + 5u) +
                      SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(sh_base + 6u) +
                      SH_C2[3] * xz * sh_coef(sh_base + 7u) +
                      SH_C2[4] * (xx - yy) * sh_coef(sh_base + 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(sh_base + 9u) +
                          SH_C3[1] * xy * z * sh_coef(sh_base + 10u) +
                          SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(sh_base + 11u) +
                          SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(sh_base + 12u) +
                          SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(sh_base + 13u) +
                          SH_C3[5] * z * (xx - yy) * sh_coef(sh_base + 14u) +
                          SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(sh_base + 15u);
            }
        }
    }

    return max(result + 0.5, vec3(0.0));
}

@compute @workgroup_size(WG_SIZE)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= sort_infos.keys_size { return; }

    let solver = sh_solvers[gid.x];
    let dir = vec3<f16>(solver.dir_xy, solver.dir_z_opacity.x);
    let opacity_val = f32(solver.dir_z_opacity.y);

    let rgb = evaluate_sh(dir, solver.idx, render_settings.max_sh_deg, render_settings.cur_sh_deg);
    let rgb_f32 = vec3<f32>(rgb);
    colors[gid.x] = vec4<f32>(rgb_f32.r, rgb_f32.g, rgb_f32.b, opacity_val);
}
