pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/GLSL/vertext.glsl"
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/GLSL/fragment.glsl"
    }
}
