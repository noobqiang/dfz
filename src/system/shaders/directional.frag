#version 460

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;
layout(input_attachment_index = 2, set = 0, binding = 4) uniform subpassInput u_frag_pos;

layout(set = 0, binding = 2) uniform Directional_Light_Data {
    vec4 position;
    vec3 color;
} directional;
layout(set = 0, binding = 3) uniform Camera {
    vec3 position;
} camera;

layout(location = 0) out vec4 f_color;

void main() {
    // 漫反射
    // vec3 light_direction = normalize(directional.position.xyz - subpassLoad(u_normals).xyz);
    vec3 light_direction = normalize(directional.position.xyz - subpassLoad(u_frag_pos).xyz);
    float directional_intensity = max(dot(normalize(subpassLoad(u_normals).rgb), light_direction), 0.0);
    vec3 directional_color = directional_intensity * directional.color;

    // 镜面反射
    vec3 view_dir = normalize(camera.position - subpassLoad(u_normals).xyz);
    vec3 reflect_dir = reflect(-light_direction, subpassLoad(u_normals).xyz);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
    // 改变常数可以改变镜面反射强度
    vec3 specular = 0.5 * spec * directional_color;

    vec3 combined_color = (directional_color + specular) * subpassLoad(u_color).rgb;
    // vec3 combined_color = directional_color * subpassLoad(u_color).rgb;
    f_color = vec4(combined_color, 1.0);
    // f_color = vec4(subpassLoad(u_normals).rgb, 1.0);
}
