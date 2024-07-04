use std::{mem, sync::Arc};

use cgmath::{Matrix4, Point3, Rad, Vector3};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo},
    image::{view::ImageView, Image},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{
        AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryAllocator,
        MemoryTypeFilter, StandardMemoryAllocator,
    },
    pipeline::{self, graphics::viewport::Viewport, GraphicsPipeline, Pipeline},
    render_pass::{Framebuffer, RenderPass, Subpass},
    swapchain::{
        self, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated, VulkanError,
};
use winit::{
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

use crate::{
    basic::{AmbientLight, DirectionalLight, VP},
    get_deferred_pipeline, get_dummy_pipeline, get_framebuffers, get_light_buffer,
    get_model_descriptor_set, get_render_pass, get_vp_buffer, get_vp_descriptor_set,
    model::Model,
    model_loader::DummyVertex,
    select_physical_device,
};

pub struct System {
    instance: Arc<Instance>,
    pub device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,

    window: Arc<Window>,
    surface: Arc<Surface>,

    memory_allocator: Arc<dyn MemoryAllocator>,
    command_buffer_allocator: StandardCommandBufferAllocator,
    descriptor_set_allocator: StandardDescriptorSetAllocator,

    vp_buffer: Subbuffer<VP>,
    // model_buffer: Subbuffer<deferred_vert::Model_Data>,
    ambient_buffer: Subbuffer<AmbientLight>,
    directional_buffer: Subbuffer<DirectionalLight>,
    render_pass: Arc<RenderPass>,
    deferred_pipeline: Arc<GraphicsPipeline>,
    ambient_pipeline: Arc<GraphicsPipeline>,
    directional_pipeline: Arc<GraphicsPipeline>,

    framebuffers: Vec<Arc<Framebuffer>>,
    color_buffer: Arc<ImageView>,
    normal_buffer: Arc<ImageView>,
    dummy_buffer: Subbuffer<[DummyVertex]>,
    vp_set: Arc<PersistentDescriptorSet>,
    vp: VP,
    viewport: Viewport,

    render_stage: RenderStage,

    commands: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
    image_index: u32,
    acquire_future: Option<SwapchainAcquireFuture>,
}

impl System {
    pub fn new(event_loop: &EventLoop<()>) -> System {
        let library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");

        // vulkano instance
        let required_extensions = Surface::required_extensions(event_loop);
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        // window、surface
        let window = Arc::new(WindowBuilder::new().build(event_loop).unwrap());
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        // device、queue
        let (physical_device, queue_family_index) =
            select_physical_device(&instance, &surface, &device_extensions);
        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create device");
        let queue = queues.next().unwrap();

        // swapchain、image
        let (swapchain, images) = {
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
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: caps.min_image_count,
                    image_format,
                    image_extent: dimensions.into(),
                    image_usage: caps.supported_usage_flags,
                    composite_alpha,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        // allocators
        let memory_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>> =
            Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let descriptor_set_allocator =
            StandardDescriptorSetAllocator::new(device.clone(), Default::default());

        // shader's entrypoints
        let deferred_vert = deferred_vert::load(device.clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();
        let deferred_frag = deferred_frag::load(device.clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();
        let directional_vert = directional_vert::load(device.clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();
        let directional_frag = directional_frag::load(device.clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();
        let ambient_vert = ambient_vert::load(device.clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();
        let ambient_frag = ambient_frag::load(device.clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();

        // buffers
        let vp_buffer = get_vp_buffer(&swapchain, memory_allocator.clone());
        let ambient_buffer = get_light_buffer(
            &AmbientLight {
                color: [1.0; 3],
                intensity: 0.1,
            },
            memory_allocator.clone(),
        );
        let directional_buffer = get_light_buffer(
            &DirectionalLight {
                position: [0.0; 4],
                color: [0.0; 3],
            },
            memory_allocator.clone(),
        );

        // render pass
        let render_pass = get_render_pass(device.clone(), swapchain.clone());
        // subpass
        let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
        let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

        // viewport
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window.inner_size().into(),
            depth_range: 0.0..=1.0,
        };

        // pipelines
        let deferred_pipeline = get_deferred_pipeline(
            device.clone(),
            deferred_pass.clone(),
            deferred_vert.clone(),
            deferred_frag.clone(),
            viewport.clone(),
        );
        let directional_pipeline = get_dummy_pipeline(
            device.clone(),
            lighting_pass.clone(),
            directional_vert.clone(),
            directional_frag.clone(),
            viewport.clone(),
        );
        let ambient_pipeline = get_dummy_pipeline(
            device.clone(),
            lighting_pass.clone(),
            ambient_vert.clone(),
            ambient_frag.clone(),
            viewport.clone(),
        );

        let (framebuffers, color_buffer, normal_buffer) =
            get_framebuffers(&images, render_pass.clone(), memory_allocator.clone());

        let dummy_buffer = Buffer::from_iter(
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
            // vertices.clone(),
            DummyVertex::list().iter().cloned(),
        )
        .unwrap();

        let vp_set = get_vp_descriptor_set(
            memory_allocator.clone(),
            &swapchain,
            &deferred_pipeline,
            &descriptor_set_allocator,
        );

        let aspect_ratio = swapchain.image_extent()[0] as f32 / swapchain.image_extent()[1] as f32;
        let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100.0);
        let view = Matrix4::look_at_rh(
            Point3::new(0.0, 0.0, 0.1),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        let scale = Matrix4::from_scale(0.03);
        let vp = VP {
            view: (view * scale).into(),
            projection: proj.into(),
        };

        let render_stage = RenderStage::Stopped;

        let commands = None;
        let image_index = 0;
        let acquire_future = None;

        System {
            instance,
            device,
            queue,
            swapchain,
            images,
            surface,
            window,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
            vp_buffer,
            ambient_buffer,
            directional_buffer,
            render_pass,
            deferred_pipeline,
            ambient_pipeline,
            directional_pipeline,
            framebuffers,
            color_buffer,
            normal_buffer,
            dummy_buffer,
            vp_set,
            vp,
            viewport,
            render_stage,
            commands,
            image_index,
            acquire_future,
        }
    }
    pub fn start(&mut self) {
        match self.render_stage {
            RenderStage::Stopped => {
                self.render_stage = RenderStage::Deferred;
            }
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
        }
        let (image_i, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None)
                .map_err(Validated::unwrap)
            {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.recreate_swapchain();
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            self.recreate_swapchain();
            return;
        }
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let framebuffer = self.framebuffers[image_i as usize].clone();
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
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
            .unwrap();
        self.commands = Some(builder);
        self.image_index = image_i;
        self.acquire_future = Some(acquire_future);
    }
    pub fn finish(&mut self, previous_frame_end: &mut Option<Box<dyn GpuFuture>>) {
        match self.render_stage {
            RenderStage::Directional => {}
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
        }

        let mut commands = self.commands.take().unwrap();
        commands.end_render_pass(Default::default()).unwrap();
        let command_buffer = commands.build().unwrap();

        let af = self.acquire_future.take().unwrap();

        let mut local_future: Option<Box<dyn GpuFuture>> =
            Some(Box::new(sync::now(self.device.clone())) as Box<dyn GpuFuture>);

        mem::swap(&mut local_future, previous_frame_end);

        let future = local_future
            .take()
            .unwrap()
            .join(af)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.swapchain.clone(),
                    self.image_index,
                ),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                *previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(VulkanError::OutOfDate) => {
                self.recreate_swapchain();
                *previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                *previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
            }
        }

        self.commands = None;
        self.render_stage = RenderStage::Stopped;
    }
    pub fn geometry(&mut self, model: &mut Model) {
        match self.render_stage {
            RenderStage::Deferred => {}
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
            _ => {
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
        }
        let model_set = get_model_descriptor_set(
            model,
            self.memory_allocator.clone(),
            &self.swapchain,
            &self.deferred_pipeline,
            &self.descriptor_set_allocator,
        );
        let vertex_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            model.data().iter().cloned(),
        )
        .unwrap();

        self.commands
            .as_mut()
            .unwrap()
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(self.deferred_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                pipeline::PipelineBindPoint::Graphics,
                self.deferred_pipeline.layout().clone(),
                0,
                (self.vp_set.clone(), model_set.clone()),
            )
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .unwrap()
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();
    }
    pub fn set_ambient(&mut self, light: &AmbientLight) {
        self.ambient_buffer = get_light_buffer(light, self.memory_allocator.clone());
    }
    pub fn ambient(&mut self) {
        match self.render_stage {
            RenderStage::Deferred => {
                self.render_stage = RenderStage::Ambient;
            }
            RenderStage::Ambient => {
                return;
            }
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
        }
        let layout = self.ambient_pipeline.layout().set_layouts().get(0).unwrap();
        let ambient_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.color_buffer.clone()),
                WriteDescriptorSet::image_view(1, self.normal_buffer.clone()),
                WriteDescriptorSet::buffer(2, self.ambient_buffer.clone()),
            ],
            [],
        )
        .unwrap();

        self.commands
            .as_mut()
            .unwrap()
            .next_subpass(
                Default::default(),
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .bind_pipeline_graphics(self.ambient_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                pipeline::PipelineBindPoint::Graphics,
                self.ambient_pipeline.layout().clone(),
                0,
                ambient_set.clone(),
            )
            .unwrap()
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_vertex_buffers(0, self.dummy_buffer.clone())
            .unwrap()
            .draw(self.dummy_buffer.len() as u32, 1, 0, 0)
            .unwrap();
    }
    pub fn directional(&mut self, light: &DirectionalLight) {
        match self.render_stage {
            RenderStage::Ambient => {
                self.render_stage = RenderStage::Directional;
            }
            RenderStage::Directional => {}
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
        }
        self.directional_buffer = get_light_buffer(light, self.memory_allocator.clone());
        let layout = self
            .directional_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();
        let directional_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::image_view(0, self.color_buffer.clone()),
                WriteDescriptorSet::image_view(1, self.normal_buffer.clone()),
                WriteDescriptorSet::buffer(2, self.directional_buffer.clone()),
            ],
            [],
        )
        .unwrap();

        self.commands
            .as_mut()
            .unwrap()
            .bind_pipeline_graphics(self.directional_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                pipeline::PipelineBindPoint::Graphics,
                self.directional_pipeline.layout().clone(),
                0,
                directional_set.clone(),
            )
            .unwrap()
            .set_viewport(0, [self.viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_vertex_buffers(0, self.dummy_buffer.clone())
            .unwrap()
            .draw(self.dummy_buffer.len() as u32, 1, 0, 0)
            .unwrap();
    }

    pub fn recreate_swapchain(&mut self) {
        self.render_stage = RenderStage::NeedsRedraw;
        self.commands = None;

        let new_dimensions = self.window.inner_size();

        let (new_swapchain, new_images) = match self.swapchain.recreate(SwapchainCreateInfo {
            image_extent: new_dimensions.into(),
            ..self.swapchain.create_info()
        }) {
            Ok(r) => r,
            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
        };

        let (new_framebuffers, new_color_buffer, new_normal_buffer) =
            System::window_size_dependent_setup(
                &new_images,
                self.render_pass.clone(),
                self.memory_allocator.clone(),
                &mut self.viewport,
            );

        self.swapchain = new_swapchain;
        self.framebuffers = new_framebuffers;
        self.color_buffer = new_color_buffer;
        self.normal_buffer = new_normal_buffer;

        // 更新投影矩阵
        let aspect_ratio =
            self.swapchain.image_extent()[0] as f32 / self.swapchain.image_extent()[1] as f32;
        self.vp.projection =
            cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100.0).into();

        self.vp_buffer = get_vp_buffer(&self.swapchain, self.memory_allocator.clone());

        let vp_layout = self
            .deferred_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();
        self.vp_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            vp_layout.clone(),
            [WriteDescriptorSet::buffer(0, self.vp_buffer.clone())],
            [],
        )
        .unwrap();

        self.render_stage = RenderStage::Stopped;
    }

    /// 更新 size 变化相关值
    pub fn window_size_dependent_setup(
        images: &Vec<Arc<Image>>,
        render_pass: Arc<RenderPass>,
        memory_allocator: Arc<dyn MemoryAllocator>,
        viewport: &mut Viewport,
    ) -> (Vec<Arc<Framebuffer>>, Arc<ImageView>, Arc<ImageView>) {
        let extent = images[0].extent();
        viewport.extent = [extent[0] as f32, extent[1] as f32];
        let (framebuffers, color_buffer, normal_buffer) =
            get_framebuffers(images, render_pass.clone(), memory_allocator.clone());
        (framebuffers, color_buffer, normal_buffer)
    }

    /// 更新 vp 相关值
    pub fn set_view(&mut self, view: &Matrix4<f32>) {
        self.vp.view = view.clone().into();
        self.vp_buffer = Buffer::from_data(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            VP {
                view: view.clone().into(),
                projection: self.vp.projection.clone(),
            },
        )
        .expect("failed to create uniform_buffer");
        let layout = self
            .deferred_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();
        self.vp_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [WriteDescriptorSet::buffer(0, self.vp_buffer.clone())],
            [],
        )
        .unwrap()
    }
}

#[derive(Debug, Clone)]
enum RenderStage {
    Stopped,
    Deferred,
    Ambient,
    Directional,
    NeedsRedraw,
}

mod deferred_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/system/shaders/deferred.vert",
    }
}

mod deferred_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/system/shaders/deferred.frag"
    }
}

/// 环境光 -- 顶点着色器
mod ambient_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/system/shaders/ambient.vert",
    }
}

/// 环境光 -- 片段着色器
mod ambient_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/system/shaders/ambient.frag",
    }
}

/// 定向光 -- 顶点着色器
mod directional_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/system/shaders/directional.vert",
    }
}

///定向光 -- 片段着色器
mod directional_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/system/shaders/directional.frag",
    }
}
