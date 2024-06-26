#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

// layout(set = 0, binding = 1) uniform Uniforms {
//     vec2 u_resolution;
// } uniforms;

float circle(vec2 uv, vec2 center, float d) {
    return length(uv - center) - d;
}

float cloud(vec2 uv, vec2 center, float d) {
    float step = 1.2;
    float c1 = circle(uv, vec2(center.x - d * 0.9 * 1.0 * step, center.y), d * 0.9);
    float c2 = circle(uv, vec2(center.x - d * 0.8 * 2.0 * step, center.y), d * 0.8);
    float c3 = circle(uv, vec2(center.x, center.y), d);
    float c4 = circle(uv, vec2(center.x + d * 0.9 * 1.0 * step, center.y), d * 0.9);
    float c5 = circle(uv, vec2(center.x + d * 0.8 * 2.0 * step, center.y), d * 0.8);
    return min(c5, min(c4, min(c3, min(c1, c2))));
}

void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);

    vec2 uv = vec2(pixel_coords) / vec2(1024, 1024);

    vec3 backCol = vec3(0.2078, 0.7765, 1.0);
    vec3 col = backCol;
    float d = cloud(uv, vec2(0.5, 0.5), 0.05);
    d = 1.0 - step(0.0, d);
    vec3 cloudCol = vec3(d);
    col = mix(col, cloudCol, d);
    vec4 color = vec4(col, 1.0);

    imageStore(img, pixel_coords, color);
}
