#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

layout(location = 0) out vec3 out_color;

layout(set = 0, binding = 0) uniform MVP {
    mat4 model;
    mat4 view;
    mat4 projection;
} uniforms;

void main() {
    mat4 worldview = uniforms.view * uniforms.model;
    gl_Position = uniforms.projection * worldview * vec4(position, 1.0);
    out_color = color;
}