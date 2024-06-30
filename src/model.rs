#![allow(unused)]
use cgmath::{BaseFloat, Matrix4, One, Rad, SquareMatrix, Vector3};

use super::model_loader::Loader;
use crate::basic::NormalVertex;

pub struct Model {
    data: Vec<NormalVertex>,
    translation: Matrix4<f32>,
    rotation: Matrix4<f32>,
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
            custom_color: [1.0, 0.35, 0.137],
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
        }
    }
}

impl Model {
    pub fn data(&self) -> Vec<NormalVertex> {
        self.data.clone()
    }

    pub fn translate(&mut self, v: Vector3<f32>) {
        self.translation += Matrix4::from_translation(v);
    }

    pub fn rotate(&mut self, axis: Vector3<f32>, radians: f32) {
        self.rotation = self.rotation * Matrix4::from_axis_angle(axis, Rad(radians));
    }

    pub fn rotate_zero(&mut self) {
        self.rotation = Matrix4::identity();
    }
}
