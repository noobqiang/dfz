#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
layout(location = 3) in vec2 uv;

layout(location = 0) out vec3 tex_coords;

layout(set = 0, binding = 0) uniform VP_Data {
    mat4 view;
    mat4 projection;
} vp_uniforms;

void main() {
    // gl_Position = vp_uniforms.projection * vp_uniforms.view * vec4(position, 1.0);
    // 让深度值永远为 1.0
    vec4 pos = vp_uniforms.projection * vp_uniforms.view * vec4(position, 1.0);
    gl_Position = pos.xyww;
    tex_coords = position;
}