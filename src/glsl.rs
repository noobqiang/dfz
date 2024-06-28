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

pub mod deferred_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/GLSL/deferred.vert",
    }
}

pub mod deferred_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/GLSL/deferred.frag"
    }
}

pub mod lighting_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/GLSL/lighting.vert",
    }
}

pub mod lighting_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/GLSL/lighting.frag",
    }
}
