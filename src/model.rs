#![allow(unused)]
use cgmath::{BaseFloat, Matrix4, One, Rad, SquareMatrix, Transform, Vector3};

use super::model_loader::Loader;
use crate::basic::{ColoredVertex, NormalVertex};

pub struct Model {
    data: Vec<NormalVertex>,
    translation: Matrix4<f32>,
    rotation: Matrix4<f32>,
    model: Matrix4<f32>,
    normals: Matrix4<f32>,
    require_update: bool,
    scale: Matrix4<f32>,
}

pub struct ModelBuilder {
    file_name: String,
    custom_color: [f32; 3],
    invert: bool,
}

impl ModelBuilder {
    pub fn new(file: &str) -> ModelBuilder {
        ModelBuilder {
            file_name: file.to_string(),
            custom_color: [1.0, 1.0, 1.0],
            invert: true,
        }
    }

    pub fn color(mut self, color: [f32; 3]) -> ModelBuilder {
        self.custom_color = color;
        self
    }

    pub fn invert(mut self, invert: bool) -> ModelBuilder {
        self.invert = invert;
        self
    }

    pub fn build(self) -> Model {
        let loader = Loader::new(self.file_name.as_str(), self.custom_color, self.invert);
        Model {
            data: loader.as_normal_vertices(),
            translation: Matrix4::identity(),
            rotation: Matrix4::identity(),
            model: Matrix4::identity(),
            normals: Matrix4::identity(),
            require_update: true,
            scale: Matrix4::identity(),
        }
    }
}

impl Model {
    pub fn data(&self) -> Vec<NormalVertex> {
        self.data.clone()
    }

    pub fn translate(&mut self, v: Vector3<f32>) {
        self.translation += Matrix4::from_translation(v);
        self.require_update = true;
    }

    pub fn rotate(&mut self, axis: Vector3<f32>, radians: f32) {
        self.rotation = Matrix4::from_axis_angle(axis, Rad(radians)) * self.rotation;
        self.require_update = true;
    }

    pub fn scale(&mut self, scale: f32) {
        self.scale = Matrix4::from_scale(scale) * self.scale;
        self.require_update = true;
    }

    pub fn rotate_zero(&mut self) {
        self.rotation = Matrix4::identity();
        self.require_update = true;
    }

    pub fn model_matrices(&mut self) -> (Matrix4<f32>, Matrix4<f32>) {
        if self.require_update {
            self.model = self.translation * self.rotation * self.scale;
            self.normals = Matrix4::inverse_transform(&self.model).unwrap();
            self.normals.transpose_self();
            self.require_update = false;
        }
        (self.model, self.normals)
    }

    pub fn color_data(&self) -> Vec<ColoredVertex> {
        let mut ret: Vec<ColoredVertex> = Vec::new();
        for v in &self.data {
            ret.push(ColoredVertex {
                position: v.position,
                color: v.color,
            });
        }
        ret
    }
}
