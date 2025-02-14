#version 460

layout(location = 0) in vec3 tex_coords;

layout(set = 1, binding = 0) uniform samplerCube skybox;

layout(location = 0) out vec4 f_color;

void main() {
    vec4 color = vec4(1.0, 1.0, 1.0, 1.0);
    f_color = texture(skybox, tex_coords) * color;
}
