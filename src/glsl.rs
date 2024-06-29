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

/// 环境光 -- 顶点着色器
pub mod ambient_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/GLSL/ambient.vert",
    }
}

/// 环境光 -- 片段着色器
pub mod ambient_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/GLSL/ambient.frag",
    }
}

/// 定向光 -- 顶点着色器
pub mod directional_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/GLSL/directional.vert",
    }
}

///定向光 -- 片段着色器
pub mod directional_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/GLSL/directional.frag",
    }
}
