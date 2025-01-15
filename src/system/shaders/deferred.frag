#version 460

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 tex_coords;
layout(location = 3) in vec4 in_frag_pos;

layout(set = 1, binding = 1) uniform sampler2D tex;
layout(location = 0) out vec4 f_color;
layout(location = 1) out vec3 f_normal;
layout(location = 2) out vec4 f_frag_pos;

void main() {
    f_color = vec4(in_color, 1.0);
    f_normal = in_normal;
    f_frag_pos = in_frag_pos;
    f_color = texture(tex, tex_coords) * f_color;
}