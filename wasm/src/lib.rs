use kriging::{OrdinaryKriging, VariogramModel};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// 输入数据结构
#[derive(Serialize, Deserialize, Debug)]
pub struct KrigingInput {
    pub points: Vec<Point>,
    pub values: Vec<f64>,
    pub target_points: Vec<Point>,
    pub model_type: String, // "spherical", "exponential", "gaussian"
    pub nugget: Option<f64>,
    pub sill: Option<f64>,
    pub range: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

// 输出数据结构
#[derive(Serialize, Deserialize, Debug)]
pub struct KrigingOutput {
    pub predictions: Vec<Prediction>,
    pub variogram_model: VariogramModelInfo,
    pub mse: Option<f64>, // 均方误差
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Prediction {
    pub x: f64,
    pub y: f64,
    pub value: f64,
    pub variance: f64, // 克里金方差
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VariogramModelInfo {
    pub model_type: String,
    pub nugget: f64,
    pub sill: f64,
    pub range: f64,
}

// 将 Rust 结构转换为 JavaScript 对象
#[wasm_bindgen]
pub fn kriging_interpolate(input: JsValue) -> Result<JsValue, JsError> {
    // 解析输入数据
    let input: KrigingInput = serde_wasm_bindgen::from_value(input)?;

    // 准备数据
    let coords: Vec<(f64, f64)> = input.points.iter().map(|p| (p.x, p.y)).collect();

    // 创建克里金模型
    let model_type = match input.model_type.as_str() {
        "spherical" => VariogramModel::Spherical,
        "exponential" => VariogramModel::Exponential,
        "gaussian" => VariogramModel::Gaussian,
        _ => return Err(JsError::new("Invalid variogram model type")),
    };

    // 创建普通克里金实例
    let mut kriging = OrdinaryKriging::new(&coords, &input.values)
        .map_err(|e| JsError::new(&format!("Failed to create kriging model: {}", e)))?;

    // 设置变差函数参数
    if let (Some(nugget), Some(sill), Some(range)) = (input.nugget, input.sill, input.range) {
        kriging
            .set_variogram(model_type, nugget, sill, range)
            .map_err(|e| JsError::new(&format!("Failed to set variogram: {}", e)))?;
    } else {
        // 自动拟合变差函数
        kriging
            .fit_variogram(model_type)
            .map_err(|e| JsError::new(&format!("Failed to fit variogram: {}", e)))?;
    }

    // 获取变差函数参数
    let (nugget, sill, range) = kriging.get_variogram_params();

    // 对目标点进行插值
    let mut predictions = Vec::new();
    for target in &input.target_points {
        match kriging.predict((target.x, target.y)) {
            Ok((value, variance)) => {
                predictions.push(Prediction {
                    x: target.x,
                    y: target.y,
                    value,
                    variance,
                });
            }
            Err(e) => {
                predictions.push(Prediction {
                    x: target.x,
                    y: target.y,
                    value: f64::NAN,
                    variance: f64::NAN,
                });
            }
        }
    }

    // 计算均方误差（如果有验证数据）
    let mse = if !input.values.is_empty() && predictions.len() == input.values.len() {
        let mut sum_sq = 0.0;
        for i in 0..predictions.len().min(input.values.len()) {
            let diff = predictions[i].value - input.values[i];
            sum_sq += diff * diff;
        }
        Some(sum_sq / predictions.len() as f64)
    } else {
        None
    };

    // 创建输出
    let output = KrigingOutput {
        predictions,
        variogram_model: VariogramModelInfo {
            model_type: input.model_type,
            nugget,
            sill,
            range,
        },
        mse,
    };

    // 转换为 JavaScript 对象
    Ok(serde_wasm_bindgen::to_value(&output)?)
}

// 版本信息
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// 批量插值函数（用于网格数据）
#[wasm_bindgen]
pub fn kriging_grid(
    points: JsValue,
    values: JsValue,
    grid_params: JsValue,
    model_type: String,
) -> Result<JsValue, JsError> {
    #[derive(Deserialize)]
    struct GridParams {
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        x_resolution: usize,
        y_resolution: usize,
    }

    // 解析输入
    let points: Vec<Point> = serde_wasm_bindgen::from_value(points)?;
    let values: Vec<f64> = serde_wasm_bindgen::from_value(values)?;
    let params: GridParams = serde_wasm_bindgen::from_value(grid_params)?;

    // 准备坐标
    let coords: Vec<(f64, f64)> = points.iter().map(|p| (p.x, p.y)).collect();

    // 创建克里金模型
    let mut kriging = OrdinaryKriging::new(&coords, &values)
        .map_err(|e| JsError::new(&format!("Failed to create kriging model: {}", e)))?;

    // 设置变差函数模型
    let model_type = match model_type.as_str() {
        "spherical" => VariogramModel::Spherical,
        "exponential" => VariogramModel::Exponential,
        "gaussian" => VariogramModel::Gaussian,
        _ => return Err(JsError::new("Invalid variogram model type")),
    };

    kriging
        .fit_variogram(model_type)
        .map_err(|e| JsError::new(&format!("Failed to fit variogram: {}", e)))?;

    // 生成网格
    let x_step = (params.x_max - params.x_min) / (params.x_resolution - 1) as f64;
    let y_step = (params.y_max - params.y_min) / (params.y_resolution - 1) as f64;

    let mut grid_values = Vec::new();
    let mut grid_vars = Vec::new();

    for i in 0..params.y_resolution {
        let y = params.y_min + i as f64 * y_step;
        for j in 0..params.x_resolution {
            let x = params.x_min + j as f64 * x_step;

            match kriging.predict((x, y)) {
                Ok((value, variance)) => {
                    grid_values.push(value);
                    grid_vars.push(variance);
                }
                Err(_) => {
                    grid_values.push(f64::NAN);
                    grid_vars.push(f64::NAN);
                }
            }
        }
    }

    // 返回网格数据
    let result = js_sys::Object::new();
    js_sys::Reflect::set(
        &result,
        &"values".into(),
        &serde_wasm_bindgen::to_value(&grid_values)?,
    )?;
    js_sys::Reflect::set(
        &result,
        &"variances".into(),
        &serde_wasm_bindgen::to_value(&grid_vars)?,
    )?;
    js_sys::Reflect::set(&result, &"x_resolution".into(), &params.x_resolution.into())?;
    js_sys::Reflect::set(&result, &"y_resolution".into(), &params.y_resolution.into())?;

    Ok(result.into())
}
