#![allow(unused)]

mod face;
mod loader;
mod vertex;
use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

pub use self::loader::Loader;

#[derive(BufferContents, Vertex, Default, Debug, Clone)]
#[repr(C)]
pub struct DummyVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
}

impl DummyVertex {
    pub fn list() -> [DummyVertex; 6] {
        [
            DummyVertex {
                position: [-1.0, -1.0],
            },
            DummyVertex {
                position: [-1.0, 1.0],
            },
            DummyVertex {
                position: [1.0, 1.0],
            },
            DummyVertex {
                position: [-1.0, -1.0],
            },
            DummyVertex {
                position: [1.0, 1.0],
            },
            DummyVertex {
                position: [1.0, -1.0],
            },
        ]
    }
}
