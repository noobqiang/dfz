#version 460

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 frag_pos;

layout(location = 0) out vec4 f_color;

// 环境光
layout(set = 0, binding = 1) uniform Ambient_Data {
    vec3 color;
    float intensity;
} ambient;

// 定向光
layout(set = 0, binding = 2) uniform Directional_Light_Data {
    vec3 position;
    vec3 color;
} directional;

void main() {
    // 叠加环境光
    vec3 ambient_color = ambient.intensity * ambient.color;

    // 定向光
    vec3 light_direction = normalize(directional.position - frag_pos);
    float directional_intensity = max(dot(in_normal, light_direction), 0.0);
    vec3 directional_color = directional.color * directional_intensity;

    vec3 combined_color = (ambient_color + directional_color) * in_color;
    f_color = vec4(combined_color, 1.0);
}