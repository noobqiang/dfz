//! This is the source code of the "Windowing" chapter at http://vulkano.rs.
//!
//! It is not commented, as the explanations can be found in the book itself.

mod basic;
mod glsl;
use basic::{AmbientLight, MyVertex};
use glsl::{fs, vs};
mod utils;
use std::sync::Arc;
use std::time::Instant;
use utils::*;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo};
use vulkano::image::ImageUsage;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::swapchain::{self, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

fn main() {
    let library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let event_loop = EventLoop::new();

    let required_extensions = Surface::required_extensions(&event_loop);
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .expect("failed to create instance");

    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());

    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) =
        select_physical_device(&instance, &surface, &device_extensions);

    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions, // new
            ..Default::default()
        },
    )
    .expect("failed to create device");

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .expect("failed to get surface capabilities");

        let dimensions = window.inner_size();
        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let image_format = physical_device
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;

        Swapchain::new(
            device.clone(),
            surface,
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count,
                image_format,
                image_extent: dimensions.into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha,
                ..Default::default()
            },
        )
        .unwrap()
    };

    let render_pass = get_render_pass(device.clone(), swapchain.clone());

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let mut framebuffers = get_framebuffers(&images, render_pass.clone(), memory_allocator.clone());

    // let vertexts: Vec<MyVertex> = vec![
    //     MyVertex {
    //         position: [-0.5, -0.5],
    //         color: [1.0, 0.0, 0.0],
    //     },
    //     MyVertex {
    //         position: [0.0, 0.5],
    //         color: [0.0, 1.0, 0.0],
    //     },
    //     MyVertex {
    //         position: [0.5, -0.25],
    //         color: [0.0, 0.0, 1.0],
    //     },
    // ];
    // let vertices = [
    //     MyVertex {
    //         position: [-0.5, 0.5, -0.5],
    //         color: [0.0, 0.0, 0.0],
    //     },
    //     MyVertex {
    //         position: [0.5, 0.5, -0.5],
    //         color: [0.0, 0.0, 0.0],
    //     },
    //     MyVertex {
    //         position: [0.0, -0.5, -0.5],
    //         color: [0.0, 0.0, 0.0],
    //     },
    //     MyVertex {
    //         position: [-0.5, -0.5, -0.6],
    //         color: [1.0, 1.0, 1.0],
    //     },
    //     MyVertex {
    //         position: [0.5, -0.5, -0.6],
    //         color: [1.0, 1.0, 1.0],
    //     },
    //     MyVertex {
    //         position: [0.0, 0.5, -0.6],
    //         color: [1.0, 1.0, 1.0],
    //     },
    // ];
    let vertices = [
        // front face
        MyVertex {
            position: [-1.000000, -1.000000, 1.000000],
            normal: [0.0000, 0.0000, 1.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [-1.000000, 1.000000, 1.000000],
            normal: [0.0000, 0.0000, 1.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [1.000000, 1.000000, 1.000000],
            normal: [0.0000, 0.0000, 1.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [-1.000000, -1.000000, 1.000000],
            normal: [0.0000, 0.0000, 1.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [1.000000, 1.000000, 1.000000],
            normal: [0.0000, 0.0000, 1.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [1.000000, -1.000000, 1.000000],
            normal: [0.0000, 0.0000, 1.0000],
            color: [1.0, 0.35, 0.137],
        },
        // back face
        MyVertex {
            position: [1.000000, -1.000000, -1.000000],
            normal: [0.0000, 0.0000, -1.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [1.000000, 1.000000, -1.000000],
            normal: [0.0000, 0.0000, -1.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [-1.000000, 1.000000, -1.000000],
            normal: [0.0000, 0.0000, -1.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [1.000000, -1.000000, -1.000000],
            normal: [0.0000, 0.0000, -1.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [-1.000000, 1.000000, -1.000000],
            normal: [0.0000, 0.0000, -1.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [-1.000000, -1.000000, -1.000000],
            normal: [0.0000, 0.0000, -1.0000],
            color: [1.0, 0.35, 0.137],
        },
        // top face
        MyVertex {
            position: [-1.000000, -1.000000, 1.000000],
            normal: [0.0000, -1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [1.000000, -1.000000, 1.000000],
            normal: [0.0000, -1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [1.000000, -1.000000, -1.000000],
            normal: [0.0000, -1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [-1.000000, -1.000000, 1.000000],
            normal: [0.0000, -1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [1.000000, -1.000000, -1.000000],
            normal: [0.0000, -1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [-1.000000, -1.000000, -1.000000],
            normal: [0.0000, -1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        // bottom face
        MyVertex {
            position: [1.000000, 1.000000, 1.000000],
            normal: [0.0000, 1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [-1.000000, 1.000000, 1.000000],
            normal: [0.0000, 1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [-1.000000, 1.000000, -1.000000],
            normal: [0.0000, 1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [1.000000, 1.000000, 1.000000],
            normal: [0.0000, 1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [-1.000000, 1.000000, -1.000000],
            normal: [0.0000, 1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [1.000000, 1.000000, -1.000000],
            normal: [0.0000, 1.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        // left face
        MyVertex {
            position: [-1.000000, -1.000000, -1.000000],
            normal: [-1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [-1.000000, 1.000000, -1.000000],
            normal: [-1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [-1.000000, 1.000000, 1.000000],
            normal: [-1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [-1.000000, -1.000000, -1.000000],
            normal: [-1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [-1.000000, 1.000000, 1.000000],
            normal: [-1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [-1.000000, -1.000000, 1.000000],
            normal: [-1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        // right face
        MyVertex {
            position: [1.000000, -1.000000, 1.000000],
            normal: [1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [1.000000, 1.000000, 1.000000],
            normal: [1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [1.000000, 1.000000, -1.000000],
            normal: [1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [1.000000, -1.000000, 1.000000],
            normal: [1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [1.000000, 1.000000, -1.000000],
            normal: [1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
        MyVertex {
            position: [1.000000, -1.000000, -1.000000],
            normal: [1.0000, 0.0000, 0.0000],
            color: [1.0, 0.35, 0.137],
        },
    ];
    let vertex_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vertices.clone(),
    )
    .unwrap();

    let rotation_start = Instant::now();
    let vs = vs::load(device.clone()).expect("failed to create shader module");
    let fs = fs::load(device.clone()).expect("failed to create shader module");

    let mut viewport = Viewport {
        offset: [0.0, 0.0],
        extent: window.inner_size().into(),
        depth_range: 0.0..=1.0,
    };

    let mut pipeline = get_pipeline(
        device.clone(),
        vs.clone(),
        fs.clone(),
        render_pass.clone(),
        viewport.clone(),
    );

    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    let mut window_resized = false;
    let mut recreate_swapchain = false;

    let frames_in_flight = images.len();
    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
    let mut previous_fence_i = 0;

    let set = get_descriptor_set(
        &rotation_start,
        memory_allocator.clone(),
        &swapchain,
        &device,
        &pipeline,
    );

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
            window_resized = true;
        }
        // Event::MainEventsCleared => {
        Event::RedrawEventsCleared => {
            let image_extent: [u32; 2] = window.inner_size().into();

            if image_extent.contains(&0) {
                return;
            }
            if window_resized || recreate_swapchain {
                recreate_swapchain = false;

                let new_dimensions = window.inner_size();

                let (new_swapchain, new_images) = swapchain
                    .recreate(SwapchainCreateInfo {
                        image_extent: new_dimensions.into(),
                        ..swapchain.create_info()
                    })
                    .expect("failed to recreate swapchain");

                if window_resized {
                    window_resized = false;

                    viewport.extent = new_dimensions.into();
                    pipeline = get_pipeline(
                        device.clone(),
                        vs.clone(),
                        fs.clone(),
                        render_pass.clone(),
                        viewport.clone(),
                    );
                }
                swapchain = new_swapchain;
                framebuffers =
                    get_framebuffers(&new_images, render_pass.clone(), memory_allocator.clone());
            }

            let set = get_descriptor_set(
                &rotation_start,
                memory_allocator.clone(),
                &swapchain,
                &device,
                &pipeline,
            );
            let command_buffers = get_command_buffers(
                &command_buffer_allocator,
                &queue,
                &pipeline,
                &framebuffers,
                &vertex_buffer,
                &set,
            );

            let (image_i, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None)
                    .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            // wait for the fence related to this image to finish (normally this would be the oldest fence)
            if let Some(image_fence) = &fences[image_i as usize] {
                image_fence.wait(None).unwrap();
            }

            let previous_future = match fences[previous_fence_i as usize].clone() {
                // Create a NowFuture
                None => {
                    let mut now = sync::now(device.clone());
                    now.cleanup_finished();

                    now.boxed()
                }
                // Use the existing FenceSignalFuture
                Some(fence) => fence.boxed(),
            };

            let future = previous_future
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffers[image_i as usize].clone())
                .unwrap()
                .then_swapchain_present(
                    queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
                )
                .then_signal_fence_and_flush();

            fences[image_i as usize] = match future.map_err(Validated::unwrap) {
                Ok(value) => Some(Arc::new(value)),
                Err(VulkanError::OutOfDate) => {
                    recreate_swapchain = true;
                    None
                }
                Err(e) => {
                    println!("failed to flush future: {e}");
                    None
                }
            };

            previous_fence_i = image_i;
        }
        _ => (),
    });
}
