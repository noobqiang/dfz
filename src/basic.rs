use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(BufferContents, Clone)]
#[repr(C)]
/// 环境光
pub struct AmbientLight {
    /// 颜色
    pub color: [f32; 3],
    /// 亮度
    pub intensity: f32,
}

#[derive(BufferContents, Clone)]
#[repr(C)]
/// 定向光
pub struct DirectionalLight {
    /// 位置
    pub position: [f32; 4],
    /// 颜色
    pub color: [f32; 3],
}

#[derive(BufferContents, Vertex, Clone)]
#[repr(C)]
pub struct NormalVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
}
