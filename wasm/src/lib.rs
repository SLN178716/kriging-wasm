use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use tsify_next::Tsify;

mod kriging;
pub use kriging::{train, predict, grid, VariogramModel, GridResult};

#[derive(Serialize, Deserialize, Debug, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[derive(Clone)]
pub struct KrigingOption {
    pub model_type: VariogramModel,
    pub sigma2: f64,
    pub alpha: f64,
}

#[derive(Serialize, Deserialize, Debug, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[derive(Clone)]
pub struct KnownPoint {
    pub point: (f64, f64),
    pub value: f64,
}

#[derive(Serialize, Deserialize, Debug, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ResultPoint {
    pub point: (f64, f64),
    pub value: f64,
}

#[derive(Serialize, Deserialize, Debug, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct InterpolateGridResult {
    pub points: Vec<ResultPoint>,
    pub xlim: (f64, f64),
    pub ylim: (f64, f64),
}

#[derive(Serialize, Deserialize, Debug, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct InterpolatePointsOption {
    pub base: KrigingOption,
    pub known_points: Vec<KnownPoint>,
    pub target_points: Vec<(f64, f64)>,
}

#[derive(Serialize, Deserialize, Debug, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct InterpolateGridOption {
    pub base: KrigingOption,
    pub known_points: Vec<KnownPoint>,
    pub polygons: Vec<Vec<(f64, f64)>>,
    pub interval: f64,
}

#[wasm_bindgen]
pub fn interpolate_points(
    option: &InterpolatePointsOption,
) -> Result<Vec<ResultPoint>, JsError> {
    let n = option.known_points.len();
    let mut t = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    
    for p in &option.known_points {
        t.push(p.value);
        x.push(p.point.0);
        y.push(p.point.1);
    }
    
    let variogram = train(&t, &x, &y, option.base.model_type.clone(), option.base.sigma2, option.base.alpha);
    
    let mut result_points = Vec::with_capacity(option.target_points.len());
    for point in &option.target_points {
        let value = predict(point.0, point.1, &variogram);
        result_points.push(ResultPoint {
            point: *point,
            value,
        });
    }
    
    Ok(result_points)
}

#[wasm_bindgen]
pub fn interpolate_grid(
    option: &InterpolateGridOption,
) -> Result<Option<GridResult>, JsError> {
    let n = option.known_points.len();
    let mut t = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    
    for p in &option.known_points {
        t.push(p.value);
        x.push(p.point.0);
        y.push(p.point.1);
    }
    
    let variogram = train(&t, &x, &y, option.base.model_type.clone(), option.base.sigma2, option.base.alpha);

    let result = grid(&option.polygons, &variogram, option.interval);
    Ok(result)
}

#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
