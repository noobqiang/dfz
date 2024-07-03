//! This is the source code of the "Windowing" chapter at http://vulkano.rs.
//!
//! It is not commented, as the explanations can be found in the book itself.

mod basic;
mod glsl;
mod model;
mod model_loader;
mod system;
use basic::{AmbientLight, DirectionalLight, NormalVertex};
use cgmath::{InnerSpace, Matrix4, Point3, Vector3};
use model::ModelBuilder;
use system::System;
mod utils;
use std::time::Instant;
use utils::*;
use vulkano::sync::{self, GpuFuture};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    // let library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let event_loop = EventLoop::new();
    let mut system = System::new(&event_loop);

    let view = Matrix4::look_at_rh(
        Point3::new(0.0, 0.0, 0.1),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
    );
    system.set_view(&view);

    // 加载模型
    let mut model = ModelBuilder::new("resource/models/warcraft.obj").build();
    model.scale(0.5);
    model.translate(Vector3::new(0.0, 0.0, -10.0).normalize());

    let mut teapot_model = ModelBuilder::new("resource/models/teapot.obj").build();
    teapot_model.scale(0.5);
    // TODO: 平移不起作用
    teapot_model.translate(Vector3::new(1000.0, 100.0, -10.0).normalize());

    // 环境光
    let ambient_light = AmbientLight {
        color: [1.0; 3],
        intensity: 0.2,
    };

    // 定向光
    let directional_light = DirectionalLight {
        position: [-4.0, -4.0, 0.0, 1.0],
        color: [1.0, 1.0, 1.0],
    };
    let directional_light_r = DirectionalLight {
        position: [-4.0, 0.0, -4.0, 1.0],
        color: [1.0, 0.0, 0.0],
    };
    let directional_light_g = DirectionalLight {
        position: [0.0, -4.0, 1.0, 1.0],
        color: [0.0, 1.0, 0.0],
    };
    let directional_light_b = DirectionalLight {
        position: [4.0, -2.0, 1.0, 1.0],
        color: [0.0, 0.0, 1.0],
    };

    let mut previous_frame_end =
        Some(Box::new(sync::now(system.device.clone())) as Box<dyn GpuFuture>);

    let rotation_start = Instant::now();
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            system.recreate_swapchain();
        }
        // Event::MainEventsCleared => {
        Event::RedrawEventsCleared => {
            previous_frame_end
                .as_mut()
                .take()
                .unwrap()
                .cleanup_finished();

            let elapsed = rotation_start.elapsed();
            let rotation_rad =
                elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;

            model.rotate_zero();
            model.rotate(Vector3::new(1.0, 0.0, 0.0).normalize(), 1.57);
            model.rotate(Vector3::new(0.0, 1.0, 0.0).normalize(), rotation_rad as f32);

            system.start();
            system.geometry(&mut model);
            system.geometry(&mut teapot_model);
            system.ambient();
            system.directional(&directional_light);
            system.directional(&directional_light_r);
            system.directional(&directional_light_g);
            system.directional(&directional_light_b);
            system.finish(&mut previous_frame_end);
        }
        _ => (),
    });
}
