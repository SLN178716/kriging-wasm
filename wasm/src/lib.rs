use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use tsify_next::Tsify;

// 输入数据结构
#[derive(Serialize, Deserialize, Debug, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct KrigingInput {
    pub points: Vec<Point>,
    pub values: Vec<f64>,
    pub target_points: Vec<Point>,
    pub model_type: String, // "spherical", "exponential", "gaussian"
    pub nugget: Option<f64>,
    pub sill: Option<f64>,
    pub range: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

// 输出数据结构
#[derive(Serialize, Deserialize, Debug, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct KrigingOutput {
    pub predictions: Vec<Prediction>,
    pub variogram_model: VariogramModelInfo,
    pub mse: Option<f64>, // 均方误差
}

#[derive(Serialize, Deserialize, Debug, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct Prediction {
    pub x: f64,
    pub y: f64,
    pub value: f64,
    pub variance: f64, // 克里金方差
}

#[derive(Serialize, Deserialize, Debug, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct VariogramModelInfo {
    pub model_type: String,
    pub nugget: f64,
    pub sill: f64,
    pub range: f64,
}

// 将 Rust 结构转换为 JavaScript 对象
#[wasm_bindgen]
pub fn kriging_interpolate(input: &KrigingInput) -> String {
    println!("input: {:?}", input);
    return "{}".to_string()
}

// 版本信息
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}