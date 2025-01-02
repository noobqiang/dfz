//! This is the source code of the "Windowing" chapter at http://vulkano.rs.
//!
//! It is not commented, as the explanations can be found in the book itself.

mod basic;
mod glsl;
mod model;
mod model_loader;
mod system;
use basic::{AmbientLight, ColoredVertex, DirectionalLight, NormalVertex};
use cgmath::{ElementWise, InnerSpace, Matrix4, Point3, Vector3};
use model::ModelBuilder;
use system::System;
use vulkano::buffer::sys;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use winit::dpi::Position;
mod utils;
use std::f32::consts::PI;
use std::io::Cursor;
use std::time::Instant;
use utils::*;
use vulkano::sync::{self, GpuFuture};
use winit::event::{Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

fn main() {
    let vertices = vec![
        NormalVertex {
            position: [-0.5, -0.5, 0.5],
            color: [1.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
        }, // top left corner
        NormalVertex {
            position: [-0.5, 0.5, 0.5],
            color: [0.0, 1.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 1.0],
        }, // bottom left corner
        NormalVertex {
            position: [0.5, -0.5, 0.5],
            color: [0.0, 0.0, 1.0],
            normal: [0.0, 1.0, 0.0],
            uv: [1.0, 0.0],
        }, // top right corner
        NormalVertex {
            position: [0.5, -0.5, 0.5],
            color: [0.0, 0.0, 1.0],
            normal: [0.0, 1.0, 0.0],
            uv: [1.0, 0.0],
        }, // top right corner
        NormalVertex {
            position: [-0.5, 0.5, 0.5],
            color: [0.0, 1.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 1.0],
        }, // bottom left corner
        NormalVertex {
            position: [0.5, 0.5, 0.5],
            color: [1.0, 0.0, 0.0],
            normal: [0.0, 0.0, -1.0],
            uv: [1.0, 1.0],
        }, // bottom right corner
    ];

    let event_loop = EventLoop::new();
    let mut system = System::new(&event_loop);

    let view = Matrix4::look_at_rh(
        Point3::new(0.0, 0.0, 10.0),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
    );
    system.set_view(&view);

    // 加载模型
    let mut craft_model = ModelBuilder::from_file("resource/models/warcraft.obj").build();
    craft_model.scale(0.5);
    craft_model.translate(Vector3::new(-2.0, 0.0, -10.0));

    let mut teapot_model = ModelBuilder::from_file("resource/models/teapot.obj").build();
    teapot_model.scale(0.2);
    teapot_model.translate(Vector3::new(0.0, 5.0, 0.0));

    let mut flat_rectangle_model = ModelBuilder::from_vertex(&vertices).build();
    // let mut flat_rectangle_model = ModelBuilder::from_file("resource/models/rectangle.obj").build();
    flat_rectangle_model.scale(4.0);
    flat_rectangle_model.translate(Vector3::new(0.0, 0.0, -10.0));

    // 环境光颜色
    let ambient_colors = [[1.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut ambient_color_index = 0;

    // 环境光
    let ambient_light = AmbientLight {
        color: ambient_colors[ambient_color_index].clone(),
        intensity: 0.2,
    };
    system.set_ambient(&ambient_light);

    // 定向光
    // let directional_light = DirectionalLight {
    //     position: [-4.0, -4.0, 0.0, 1.0],
    //     color: [1.0, 1.0, 1.0],
    // };
    // let directional_light_r = DirectionalLight {
    //     position: [-4.0, 0.0, -4.0, 1.0],
    //     color: [1.0, 0.0, 0.0],
    // };
    // let directional_light_g = DirectionalLight {
    //     position: [0.0, -4.0, 1.0, 1.0],
    //     color: [0.0, 1.0, 0.0],
    // };
    // let directional_light_b = DirectionalLight {
    //     position: [4.0, -2.0, 1.0, 1.0],
    //     color: [0.0, 0.0, 1.0],
    // };

    let mut light_obj_x = 0.0;
    let mut light_obj_y = 0.0;

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
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::KeyboardInput {
                input,
                is_synthetic: false,
                ..
            } => {
                let key = input.virtual_keycode.unwrap();
                match key {
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
                        light_obj_x += 0.1;
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left => {
                        light_obj_x -= 0.1;
                    }
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        light_obj_y -= 0.1;
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        light_obj_y += 0.1;
                    }
                    _ => (),
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left && state == winit::event::ElementState::Released {
                    if ambient_color_index == 0 {
                        ambient_color_index = 3;
                    } else {
                        ambient_color_index -= 1;
                    }
                }

                if button == MouseButton::Right && state == winit::event::ElementState::Released {
                    if ambient_color_index == 3 {
                        ambient_color_index = 0;
                    } else {
                        ambient_color_index += 1;
                    }
                }

                let ambient_light = AmbientLight {
                    color: ambient_colors[ambient_color_index].clone(),
                    intensity: 0.2,
                };
                system.set_ambient(&ambient_light);
            }
            _ => (),
        },
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

            craft_model.rotate_zero();
            craft_model.rotate(Vector3::new(1.0, 0.0, 0.0).normalize(), 5.41);
            craft_model.rotate(Vector3::new(0.0, 1.0, 0.0).normalize(), rotation_rad as f32);

            let elapsed = rotation_start.elapsed().as_secs() as f32
                + rotation_start.elapsed().subsec_nanos() as f32 / 1_000_000_000.0;
            let elapsed_as_radians = elapsed * 30.0 * (PI / 180.0);

            let x: f32 = 4.0 * elapsed_as_radians.cos();
            let z: f32 = -3.0 + (8.0 * elapsed_as_radians.sin());

            let directional_light_with_obj = DirectionalLight {
                // position: [x, -1.0, z, 1.0],
                position: [light_obj_x, light_obj_y, 0.0, 0.0],
                color: [1.0, 1.0, 1.0],
            };

            let mut light_obj_model = ModelBuilder::from_file("resource/models/sphere.obj")
                .color(directional_light_with_obj.color)
                .build();
            light_obj_model.scale(0.1);

            system.start();
            // system.geometry(&mut teapot_model);
            system.geometry(&mut flat_rectangle_model);
            system.geometry(&mut craft_model);
            system.ambient();
            // system.directional(&directional_light);
            // system.directional(&directional_light_r);
            // system.directional(&directional_light_g);
            // system.directional(&directional_light_b);
            system.directional(&directional_light_with_obj);
            system.light_object(&directional_light_with_obj, &mut light_obj_model);
            system.finish(&mut previous_frame_end);
        }
        _ => (),
    });
}
