use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3};
use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceExtensions, Queue, QueueFlags};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::Instance;
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
use vulkano::swapchain::{Surface, Swapchain};

use crate::basic::{AmbientLight, DirectionalLight, MyVertex};
use crate::glsl::vs;

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

pub fn get_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain>) -> Arc<RenderPass> {
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

pub fn get_framebuffers(
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

pub fn get_pipeline(
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

pub fn get_command_buffers(
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
pub fn get_descriptor_set(
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

    // 环境光
    let ambient_light = AmbientLight {
        color: [1.0; 3],
        intensity: 0.2,
    };
    let ambient_buffer = Buffer::from_data(
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
        ambient_light.clone(),
    )
    .expect("failed to create ambient buffer");

    // 定向光
    let directional_light = DirectionalLight {
        position: [-4.0, -4.0, 0.0],
        color: [1.0, 1.0, 1.0],
    };
    let directional_buffer = Buffer::from_data(
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
        directional_light.clone(),
    )
    .expect("failed to create directional light buffer");

    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());
    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [
            WriteDescriptorSet::buffer(0, uniform_buffer.clone()),
            WriteDescriptorSet::buffer(1, ambient_buffer.clone()),
            WriteDescriptorSet::buffer(2, directional_buffer.clone()),
        ],
        [],
    )
    .unwrap()
}