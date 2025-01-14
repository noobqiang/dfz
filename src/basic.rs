#![allow(unused)]
use cgmath::{vec3, Matrix4, SquareMatrix, Vector3};
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(BufferContents, Clone)]
#[repr(C)]
/// 摄像机位置
pub struct CameraPosition {
    /// 位置
    pub position: [f32; 3],
}

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
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
}

#[derive(BufferContents, Vertex, Clone)]
#[repr(C)]
pub struct ColoredVertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub color: [f32; 3],
}

#[derive(BufferContents)]
#[repr(C)]
pub struct VP {
    pub view: [[f32; 4]; 4],
    pub projection: [[f32; 4]; 4],
}

impl VP {
    pub fn new() -> VP {
        VP {
            view: Matrix4::identity().into(),
            projection: Matrix4::identity().into(),
        }
    }
}

impl DirectionalLight {
    /// 获取方向光源位置
    pub fn get_position(&self) -> Vector3<f32> {
        vec3(self.position[0], self.position[1], self.position[2])
    }
}
