use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use tsify_next::Tsify;
use oxigdal_analytics::interpolation::kriging;
use ndarray::{Array2, ArrayView, ArrayView1};
use web_sys::console;

#[derive(Serialize, Deserialize, Debug, Clone, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub enum KrigingType {
    Ordinary,
    Universal,
}

#[derive(Serialize, Deserialize, Debug, Clone, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub enum VariogramModel {
    Spherical,
    Exponential,
    Gaussian,
    Linear,
}

// 输入数据结构
#[derive(Serialize, Deserialize, Debug, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[derive(Clone)]
pub struct KrigingOption {
    pub kriging_type: KrigingType,
    pub model_type: VariogramModel,
    pub nugget: f64,
    pub range: f64,
    pub sill: Option<f64>,
}

// 输入数据结构
#[derive(Serialize, Deserialize, Debug, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
#[derive(Clone)]
pub struct KnownPoint {
    pub point: (f64, f64),
    pub value: f64,
}

// 输出数据点结构
#[derive(Serialize, Deserialize, Debug, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct ResultPoint {
    pub point: (f64, f64),
    pub value: f64,
    pub variance: f64,
}

// 输出网格数据结构
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

// 将 Rust 结构转换为 JavaScript 对象
#[wasm_bindgen]
pub fn interpolate_points(input: &InterpolatePointsOption) -> Result<Vec<ResultPoint>, JsError> {
    // 准备数据
    let points: Vec<f64> = input.known_points.iter().flat_map(|p| vec![p.point.0, p.point.1]).collect();
    let points: Array2<f64> = Array2::from_shape_vec((input.known_points.len(), 2), points).unwrap();
    let values: Vec<f64> = input.known_points.iter().map(|p| p.value).collect();
    let values: ArrayView1<f64> = ArrayView::from(values.as_slice());
    let target_points: Vec<f64> = input.target_points.iter().flat_map(|p| vec![p.0, p.1]).collect();
    let target_points: Array2<f64> = Array2::from_shape_vec((input.target_points.len(), 2), target_points).unwrap();
    let base_option = &input.base;
    // 创建克里金插值器
    let kriging_type = match base_option.kriging_type {
        KrigingType::Ordinary => kriging::KrigingType::Ordinary,
        KrigingType::Universal => kriging::KrigingType::Universal,
    };
    
    // 设置变异函数模型
    let model = match base_option.model_type {
        VariogramModel::Spherical => kriging::VariogramModel::Spherical,
        VariogramModel::Exponential => kriging::VariogramModel::Exponential,
        VariogramModel::Gaussian => kriging::VariogramModel::Gaussian,
        VariogramModel::Linear => kriging::VariogramModel::Linear,
    };

    let var = values.var(0.0);

    let variogram = kriging::Variogram::new(model, base_option.nugget, base_option.sill.unwrap_or(var), base_option.range);

    let interpolator = kriging::KrigingInterpolator::new(kriging_type, variogram);
    
    // 执行插值
    match interpolator.interpolate(&points, &values, &target_points) {
        Ok(result) => {
            let result_points: Vec<ResultPoint> = result.values.into_iter().enumerate().map(|(i, v)| {
                ResultPoint {
                    point: (result.coordinates[[i, 0]], result.coordinates[[i, 1]]),
                    value: v,
                    variance: result.variances[i],
                }
            }).collect();
            
            Ok(result_points)
        },
        Err(e) => Err(JsError::new(&e.to_string())),
    }
}

#[wasm_bindgen]
pub fn interpolate_grid(input: &InterpolateGridOption) -> Result<InterpolateGridResult, JsError> {
    let (target_points, xlim, ylim) = grid2points(&input.polygons, input.interval);
    
    let result_points = interpolate_points(&InterpolatePointsOption {
        base: input.base.clone(),
        known_points: input.known_points.clone(),
        target_points,
    })?;
    
    Ok(InterpolateGridResult {
        points: result_points,
        xlim: (xlim.0, xlim.1),
        ylim: (ylim.0, ylim.1),
    })
}

fn grid2points(polygons: &Vec<Vec<(f64, f64)>>, interval: f64) -> (Vec<(f64, f64)>, (f64, f64), (f64, f64)) {
    let n = polygons.len();
    // 至少需要 1 个多边形来定义一个区域
    if n < 1 {
        return (Vec::new(), (0.0, 0.0), (0.0, 0.0));
    }

    // 计算所有多边形的边界范围
    let mut xlim = (polygons[0][0].0, polygons[0][0].0);
    let mut ylim = (polygons[0][0].1, polygons[0][0].1);

    let mut lims: Vec<((f64, f64), (f64, f64))> = Vec::new();
    for polygon in polygons {
        let mut xy = ((polygon[0].0, polygon[0].0), (polygon[0].1, polygon[0].1));
        for point in polygon {
            xlim.0 = xlim.0.min(point.0);
            xlim.1 = xlim.1.max(point.0);
            ylim.0 = ylim.0.min(point.1);
            ylim.1 = ylim.1.max(point.1);
            xy.0.0 = xy.0.0.min(point.0);
            xy.0.1 = xy.0.1.max(point.0);
            xy.1.0 = xy.1.0.min(point.1);
            xy.1.1 = xy.1.1.max(point.1);
        }
        lims.push(xy);
    }

    let mut points = Vec::new();
    for (idx, polygon) in polygons.iter().enumerate() {
        let (lxlim, lylim) = lims[idx];
        let mut i = xlim.0;
        while i < xlim.1 {
            if i < lxlim.0 {
                i += interval;
                continue;
            }
            if i > lxlim.1 {
                break;
            }
            let mut j = ylim.0;
            while j < ylim.1 {
                if j < lylim.0 {
                    j += interval;
                    continue;
                }
                if j > lylim.1 {
                    break;
                }
                let p = (i, j);
                if point_in_polygon(&p, &polygon) {
                    points.push(p);
                }
                j += interval;
            }
            i += interval;
        }
    }
    (points, xlim, ylim)
}

fn point_in_polygon(point: &(f64, f64), polygon: &Vec<(f64, f64)>) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    let (x, y) = *point;
    let mut inside = false;
    let n = polygon.len();

    for i in 0..n {
        let j = (i + 1) % n;
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];

        // 检查点是否在顶点上
        if (x == xi && y == yi) || (x == xj && y == yj) {
            return true;
        }

        // 检查点的 y 坐标是否在边的 y 范围内
        if (yi > y) != (yj > y) {
            // 计算交点的 x 坐标
            let x_intersect = ((y - yi) * (xj - xi)) / (yj - yi) + xi;
            // 如果交点在点的右侧，则切换 inside 状态
            if x < x_intersect {
                inside = !inside;
            }
        }
    }

    inside
}

// 版本信息
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}