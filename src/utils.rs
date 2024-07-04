#![allow(unused)]

use cgmath::{InnerSpace, Matrix4, Point3, Rad, Vector3};
use std::sync::Arc;
use std::time::Instant;
use vulkano::buffer::{
    subbuffer, Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo,
    SubpassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{
    PhysicalDevice, PhysicalDeviceType, RayTracingInvocationReorderMode,
};
use vulkano::device::{Device, DeviceExtensions, Queue, QueueFlags};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::Instance;
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::depth_stencil::{DepthState, DepthStencilState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::{CullMode, FrontFace, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::{self, PipelineDescriptorSetLayoutCreateInfo};
use vulkano::pipeline::{
    self, GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::EntryPoint;
use vulkano::swapchain::{Surface, Swapchain};

use crate::basic::{NormalVertex, VP};
use crate::glsl::{deferred_vert, vs};
use crate::model::Model;
use crate::model_loader::DummyVertex;

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
    vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
            final_color: {
                format: swapchain.image_format(), // set the format the same as the swapchain
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
            color: {
                format: Format::A2B10G10R10_UNORM_PACK32,
                samples: 1,
                load_op: Clear,
                store_op: DontCare,
            },
            normals: {
                format: Format::R16G16B16A16_SFLOAT,
                samples: 1,
                load_op: Clear,
                store_op: DontCare,
            },
            depth: {
                format: Format::D16_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: DontCare,
            },
        },
        passes: [
            {
                color: [color, normals],
                depth_stencil: {depth},
                input: []
            },
            {
                color: [final_color],
                depth_stencil: {},
                input: [color, normals]
            }
        ]
    )
    .unwrap()
}

pub fn get_framebuffers(
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
    memory_allocator: Arc<dyn MemoryAllocator>,
) -> (Vec<Arc<Framebuffer>>, Arc<ImageView>, Arc<ImageView>) {
    let depth_buffer = ImageView::new_default(
        Image::new(
            memory_allocator.clone(),
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

    let color_buffer = ImageView::new_default(
        Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::A2B10G10R10_UNORM_PACK32,
                extent: images[0].extent(),
                usage: ImageUsage::INPUT_ATTACHMENT | ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
    )
    .unwrap();
    let normal_buffer = ImageView::new_default(
        Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: Format::R16G16B16A16_SFLOAT,
                extent: images[0].extent(),
                usage: ImageUsage::INPUT_ATTACHMENT | ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
    )
    .unwrap();

    let framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![
                        view,
                        color_buffer.clone(),
                        normal_buffer.clone(),
                        depth_buffer.clone(),
                    ],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    (framebuffers, color_buffer.clone(), normal_buffer.clone())
}

pub fn get_deferred_pipeline(
    device: Arc<Device>,
    subpass: Subpass,
    vs: EntryPoint,
    fs: EntryPoint,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    let vertex_input_state = NormalVertex::per_vertex()
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
                front_face: FrontFace::CounterClockwise,
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

/// 获取一个 lighting GraphicsPipeline
pub fn get_lighting_pipeline(
    device: Arc<Device>,
    subpass: Subpass,
    vs: EntryPoint,
    fs: EntryPoint,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    let vertex_input_state = NormalVertex::per_vertex()
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
                front_face: FrontFace::CounterClockwise,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend {
                        color_blend_op: BlendOp::Add,
                        src_color_blend_factor: BlendFactor::One,
                        dst_color_blend_factor: BlendFactor::One,
                        alpha_blend_op: BlendOp::Max,
                        src_alpha_blend_factor: BlendFactor::One,
                        dst_alpha_blend_factor: BlendFactor::One,
                    }),
                    ..Default::default()
                },
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

pub fn get_dummy_pipeline(
    device: Arc<Device>,
    subpass: Subpass,
    vs: EntryPoint,
    fs: EntryPoint,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    let vertex_input_state = DummyVertex::per_vertex()
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
                front_face: FrontFace::CounterClockwise,
                ..Default::default()
            }),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState {
                    blend: Some(AttachmentBlend {
                        color_blend_op: BlendOp::Add,
                        src_color_blend_factor: BlendFactor::One,
                        dst_color_blend_factor: BlendFactor::One,
                        alpha_blend_op: BlendOp::Max,
                        src_alpha_blend_factor: BlendFactor::One,
                        dst_alpha_blend_factor: BlendFactor::One,
                    }),
                    ..Default::default()
                },
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

/// 构建命令缓冲区
pub fn get_basic_command_buffers(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: &Arc<Queue>,
    deferred_pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    vertex_buffer: &Subbuffer<[NormalVertex]>,
    vp_set: &Arc<PersistentDescriptorSet>,
    model_set: &Arc<PersistentDescriptorSet>,
    viewport: Viewport,
) -> Vec<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>> {
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
                        clear_values: vec![
                            Some([0.0, 0.0, 0.0, 1.0].into()),
                            Some([0.0, 0.0, 0.0, 1.0].into()),
                            Some([0.0, 0.0, 0.0, 1.0].into()),
                            Some(1.0.into()),
                        ],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap()
                // TODO: viewport ?
                .set_viewport(0, [viewport.clone()].into_iter().collect())
                .unwrap()
                // deferred pipeline - subpass 1
                .bind_pipeline_graphics(deferred_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    pipeline::PipelineBindPoint::Graphics,
                    deferred_pipeline.layout().clone(),
                    0,
                    (vp_set.clone(), model_set.clone()),
                )
                .unwrap()
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .unwrap()
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .next_subpass(
                    Default::default(),
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap();
            builder
        })
        .collect()
}
pub fn append_light_command(
    builders: Vec<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    vertex_buffer: &Subbuffer<[NormalVertex]>,
    light_pipeline: &Arc<GraphicsPipeline>,
    light_set: &Arc<PersistentDescriptorSet>,
) -> Vec<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>> {
    builders
        .into_iter()
        .map(|mut builder| {
            builder
                .bind_pipeline_graphics(light_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    pipeline::PipelineBindPoint::Graphics,
                    light_pipeline.layout().clone(),
                    0,
                    light_set.clone(),
                )
                .unwrap()
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap();
            builder
        })
        .collect()
}

pub fn append_dummy_command(
    builders: Vec<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    vertex_buffer: &Subbuffer<[DummyVertex]>,
    light_pipeline: &Arc<GraphicsPipeline>,
    light_set: &Arc<PersistentDescriptorSet>,
) -> Vec<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>> {
    builders
        .into_iter()
        .map(|mut builder| {
            builder
                .bind_pipeline_graphics(light_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    pipeline::PipelineBindPoint::Graphics,
                    light_pipeline.layout().clone(),
                    0,
                    light_set.clone(),
                )
                .unwrap()
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .unwrap()
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap();
            builder
        })
        .collect()
}

pub fn end_render_pass(
    builders: Vec<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    builders
        .into_iter()
        .map(|mut builder| {
            builder.end_render_pass(Default::default()).unwrap();
            builder.build().unwrap()
        })
        .collect()
}

/// 构建命令缓冲区
#[allow(unused)]
pub fn get_command_buffers(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: &Arc<Queue>,
    deferred_pipeline: &Arc<GraphicsPipeline>,
    first_light_pipeline: &Arc<GraphicsPipeline>,
    second_light_pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    vertex_buffer: &Subbuffer<[NormalVertex]>,
    deferred_set: &Arc<PersistentDescriptorSet>,
    first_light_set: &Arc<PersistentDescriptorSet>,
    second_light_set: &Arc<PersistentDescriptorSet>,
    viewport: Viewport,
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
                        clear_values: vec![
                            Some([0.0, 0.0, 0.0, 1.0].into()),
                            Some([0.0, 0.0, 0.0, 1.0].into()),
                            Some([0.0, 0.0, 0.0, 1.0].into()),
                            Some(1.0.into()),
                        ],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap()
                // TODO: viewport ?
                .set_viewport(0, [viewport.clone()].into_iter().collect())
                .unwrap()
                // deferred pipeline - subpass 1
                .bind_pipeline_graphics(deferred_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    pipeline::PipelineBindPoint::Graphics,
                    deferred_pipeline.layout().clone(),
                    0,
                    deferred_set.clone(),
                )
                .unwrap()
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .unwrap()
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .next_subpass(
                    Default::default(),
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap()
                // directional light pipeline - subpass 2
                .bind_pipeline_graphics(first_light_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    pipeline::PipelineBindPoint::Graphics,
                    first_light_pipeline.layout().clone(),
                    0,
                    first_light_set.clone(),
                )
                .unwrap()
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                // ambient light pipeline -  subpass 2
                .bind_pipeline_graphics(second_light_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    pipeline::PipelineBindPoint::Graphics,
                    second_light_pipeline.layout().clone(),
                    0,
                    second_light_set.clone(),
                )
                .unwrap()
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                // end
                .end_render_pass(Default::default())
                .unwrap();

            builder.build().unwrap()
        })
        .collect()
}

/// get uniform buffer descriptor set
pub fn get_vp_descriptor_set(
    memory_allocator: Arc<dyn MemoryAllocator>,
    swapchain: &Swapchain,
    pipeline: &Arc<GraphicsPipeline>,
    descriptor_set_allocator: &StandardDescriptorSetAllocator,
) -> Arc<PersistentDescriptorSet> {
    let vp_buffer = get_vp_buffer(swapchain, memory_allocator.clone());
    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    PersistentDescriptorSet::new(
        descriptor_set_allocator,
        layout.clone(),
        [WriteDescriptorSet::buffer(0, vp_buffer)],
        [],
    )
    .unwrap()
}

pub fn get_model_descriptor_set(
    model: &mut Model,
    memory_allocator: Arc<dyn MemoryAllocator>,
    swapchain: &Swapchain,
    pipeline: &Arc<GraphicsPipeline>,
    descriptor_set_allocator: &StandardDescriptorSetAllocator,
) -> Arc<PersistentDescriptorSet> {
    let model_buffer = get_model_buffer(model, memory_allocator.clone());
    let layout = pipeline.layout().set_layouts().get(1).unwrap();
    PersistentDescriptorSet::new(
        descriptor_set_allocator,
        layout.clone(),
        [WriteDescriptorSet::buffer(0, model_buffer)],
        [],
    )
    .unwrap()
}

/// get uniform buffer descriptor set
pub fn get_lighting_descriptor_set<T: BufferContents + Clone>(
    light: &T,
    memory_allocator: Arc<dyn MemoryAllocator>,
    color_buffer: Arc<ImageView>,
    normal_buffer: Arc<ImageView>,
    swapchain: &Swapchain,
    pipeline: &Arc<GraphicsPipeline>,
    descriptor_set_allocator: &StandardDescriptorSetAllocator,
) -> Arc<PersistentDescriptorSet> {
    let uniform_buffer = get_vp_buffer(swapchain, memory_allocator.clone());

    let light_buffer = get_light_buffer(light, memory_allocator.clone());

    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    PersistentDescriptorSet::new(
        descriptor_set_allocator,
        layout.clone(),
        [
            WriteDescriptorSet::image_view(0, color_buffer),
            WriteDescriptorSet::image_view(1, normal_buffer),
            WriteDescriptorSet::buffer(2, uniform_buffer),
            WriteDescriptorSet::buffer(3, light_buffer),
        ],
        [],
    )
    .unwrap()
}

pub fn get_dummy_descriptor_set<T: BufferContents + Clone>(
    light: &T,
    memory_allocator: Arc<dyn MemoryAllocator>,
    color_buffer: Arc<ImageView>,
    normal_buffer: Arc<ImageView>,
    pipeline: &Arc<GraphicsPipeline>,
    descriptor_set_allocator: &StandardDescriptorSetAllocator,
) -> Arc<PersistentDescriptorSet> {
    let light_buffer = get_light_buffer(light, memory_allocator.clone());

    let layout = pipeline.layout().set_layouts().get(0).unwrap();
    PersistentDescriptorSet::new(
        descriptor_set_allocator,
        layout.clone(),
        [
            WriteDescriptorSet::image_view(0, color_buffer),
            WriteDescriptorSet::image_view(1, normal_buffer),
            WriteDescriptorSet::buffer(2, light_buffer),
        ],
        [],
    )
    .unwrap()
}

pub fn get_vp_buffer(
    swapchain: &Swapchain,
    // TOOD: 这里如果使用 &Arc<dyn MemoryAllocator> 在 system.rs 那边会报错
    memory_allocator: Arc<dyn MemoryAllocator>,
) -> Subbuffer<VP> {
    let aspect_ratio = swapchain.image_extent()[0] as f32 / swapchain.image_extent()[1] as f32;
    let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100.0);
    let view = Matrix4::look_at_rh(
        Point3::new(0.0, 0.0, 0.1),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
    );
    let scale = Matrix4::from_scale(0.03);
    let vp_data = VP {
        view: (view * scale).into(),
        projection: proj.into(),
    };
    let vp_buffer = Buffer::from_data(
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
        vp_data,
    )
    .expect("failed to create uniform_buffer");
    vp_buffer
}

pub fn get_model_buffer(
    model: &mut Model,
    memory_allocator: Arc<dyn MemoryAllocator>,
) -> Subbuffer<deferred_vert::Model_Data> {
    let (model_mat, normal_mat) = model.model_matrices();
    let model_buffer = Buffer::from_data(
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
        deferred_vert::Model_Data {
            model: model_mat.into(),
            normals: normal_mat.into(),
        },
    )
    .expect("failed to create uniform_buffer");
    model_buffer
}

/// 获取一个 light buffer
pub fn get_light_buffer<T: BufferContents + Clone>(
    light: &T,
    memory_allocator: Arc<dyn MemoryAllocator>,
) -> Subbuffer<T> {
    Buffer::from_data(
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
        (*light).clone(),
    )
    .expect("failed to create a light buffer")
}
