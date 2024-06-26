//! This is the source code of the "Windowing" chapter at http://vulkano.rs.
//!
//! It is not commented, as the explanations can be found in the book itself.

use std::sync::Arc;
use std::time::Instant;

use cgmath::{vec3, Matrix3, Matrix4, Point3, Rad, Vector3};
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    self, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::ShaderModule;
use vulkano::swapchain::{self, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

#[derive(BufferContents, Vertex, Clone)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    color: [f32; 3],
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/GLSL/vertext.glsl"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/GLSL/fragment.glsl"
    }
}
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

pub fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    device_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("failed to enumerate physical devices")
        .filter(|p| p.supported_extensions().contains(device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .expect("no device available")
}

fn get_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain>) -> Arc<RenderPass> {
    vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                format: swapchain.image_format(), // set the format the same as the swapchain
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
            depth: {
                format: Format::D16_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: DontCare,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {depth},
        },
    )
    .unwrap()
}

fn get_framebuffers(
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
    memory_allocator: Arc<StandardMemoryAllocator>,
) -> Vec<Arc<Framebuffer>> {
    let depth_buffer = ImageView::new_default(
        Image::new(
            memory_allocator,
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::D16_UNORM,
                extent: images[0].extent(),
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
    )
    .unwrap();

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn get_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

    let vertex_input_state = MyVertex::per_vertex()
        .definition(&vs.info().input_interface)
        .unwrap();

    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
    // let mut depth_stencil_state = DepthStencilState::default();
    // depth_stencil_state.depth = Some(DepthState::simple());

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState {
                cull_mode: CullMode::Back,
                front_face: FrontFace::Clockwise,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            depth_stencil_state: Some(DepthStencilState {
                depth: Some(DepthState::simple()),
                ..Default::default()
            }),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

fn get_command_buffers(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: &Arc<Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    vertex_buffer: &Subbuffer<[MyVertex]>,
    set: &Arc<PersistentDescriptorSet>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.0, 0.68, 1.0, 1.0].into()), Some(1.0.into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    pipeline::PipelineBindPoint::Graphics,
                    pipeline.layout().clone(),
                    0,
                    set.clone(),
                )
                .unwrap()
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .unwrap()
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .end_render_pass(Default::default())
                .unwrap();

            builder.build().unwrap()
        })
        .collect()
}

/// get uniform buffer descriptor set
fn get_descriptor_set(
    rotation_start: &Instant,
    memory_allocator: Arc<dyn MemoryAllocator>,
    swapchain: &Swapchain,
    device: &Arc<Device>,
    pipeline: &Arc<GraphicsPipeline>,
) -> Arc<PersistentDescriptorSet> {
    let elapsed = rotation_start.elapsed();
    let rotation = elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
    let rotation =
        Matrix3::from_angle_y(Rad(rotation as f32)) * Matrix3::from_angle_z(Rad(rotation as f32));
    // let rotation = Matrix4::from_axis_angle(vec3(0.0, 0.0, 1.0), Rad(50 as f32))
    //     * Matrix4::from_axis_angle(vec3(0.0, 1.0, 0.0), Rad(30 as f32))
    //     * Matrix4::from_axis_angle(vec3(1.0, 0.0, 0.0), Rad(20 as f32));

    // note: this teapot was meant for OpenGL where the origin is at the lower left
    //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis
    let aspect_ratio = swapchain.image_extent()[0] as f32 / swapchain.image_extent()[1] as f32;
    let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100.0);
    let view = Matrix4::look_at_rh(
        Point3::new(0.0, 0.0, 0.1),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
    );
    let scale = Matrix4::from_scale(0.03);

    let uniform_data = vs::MVP {
        model: Matrix4::from(rotation).into(),
        view: (view * scale).into(),
        projection: proj.into(),
    };
    // let uniform_data = vs::MVP {
    //     model: Matrix4::from(rotation).into(),
    //     view: view.into(),
    //     projection: proj.into(),
    // };
    let uniform_buffer = Buffer::from_data(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::UNIFORM_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        uniform_data,
    )
    .expect("failed to create uniform_buffer");

    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());
    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [WriteDescriptorSet::buffer(0, uniform_buffer.clone())],
        [],
    )
    .unwrap()
}
