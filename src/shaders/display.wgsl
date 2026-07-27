@group(0) @binding(0) var our_texture: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    // A single triangle to cover the screen
    const pos = array(
        vec2<f32>(-1.0, -1.0), 
        vec2<f32>(3.0, -1.0), 
        vec2<f32>(-1.0, 3.0)
    );
    return vec4<f32>(pos[in_vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = textureDimensions(our_texture);
    let tex_coords = vec2<u32>(floor(frag_coord.xy));
    // Clamp coordinates to be safe
    let clamped_coords = clamp(tex_coords, vec2<u32>(0,0), dims - vec2<u32>(1,1));
    return textureLoad(our_texture, vec2<u32>(clamped_coords.x, dims.y - clamped_coords.y), 0);
}