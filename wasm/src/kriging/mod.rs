pub mod utils;

use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use tsify_next::Tsify;
use utils::*;
use web_sys::console;

// 变异函数模型
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub enum VariogramModel {
    Gaussian,
    Exponential,
    Spherical,
}

// 克里金参数结构体
#[derive(Debug)]
pub struct Variogram {
    pub t: Vec<f64>,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub nugget: f64,
    pub range: f64,
    pub sill: f64,
    pub a: f64,
    pub n: usize,
    pub model: VariogramModel,
    pub model_func: fn(f64, f64, f64, f64, f64) -> f64,
    pub k: Array2<f64>,
    pub m: Array1<f64>,
}

// 网格结果结构体
#[derive(Debug, Clone, Serialize, Deserialize, Tsify)]
#[tsify(into_wasm_abi, from_wasm_abi)]
pub struct GridResult {
    pub xlim: [f64; 2],
    pub ylim: [f64; 2],
    pub width: f64,
    pub data: Vec<Vec<Option<f64>>>,
    pub zlim: [f64; 2],
}

// 训练函数
pub fn train(
    t: &[f64],
    x: &[f64],
    y: &[f64],
    model: VariogramModel,
    sigma2: f64,
    alpha: f64,
) -> Variogram {
    let mut variogram = Variogram {
        t: t.to_vec(),
        x: x.to_vec(),
        y: y.to_vec(),
        nugget: 0.0,
        range: 0.0,
        sill: 0.0,
        a: 1.0 / 3.0,
        n: 0,
        model: model.clone(),
        model_func: match model {
            VariogramModel::Gaussian => variogram_gaussian,
            VariogramModel::Exponential => variogram_exponential,
            VariogramModel::Spherical => variogram_spherical,
        },
        k: Array2::zeros((0, 0)),
        m: Array1::zeros(0),
    };

    // 计算距离和半方差
    let n = t.len();
    let distance_count = (n * n - n) / 2;
    let mut distance = Vec::with_capacity(distance_count);

    for i in 0..n {
        for j in 0..i {
            let dist = ((x[i] - x[j]).powi(2) + (y[i] - y[j]).powi(2)).sqrt();
            let semi = (t[i] - t[j]).abs();
            distance.push((dist, semi));
        }
    }

    // 按距离排序
    distance.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    if !distance.is_empty() {
        variogram.range = distance.last().unwrap().0;
    }

    // 分箱
    let lags = if distance_count > 30 {
        30
    } else {
        distance_count
    };
    let tolerance = if variogram.range > 0.0 {
        variogram.range / lags as f64
    } else {
        1.0
    };
    let mut lag = vec![0.0; lags];
    let mut semi = vec![0.0; lags];

    if lags < 30 {
        for l in 0..lags {
            lag[l] = distance[l].0;
            semi[l] = distance[l].1;
        }
    } else {
        let mut i = 0;
        let mut j = 0;
        let mut l = 0;

        while i < lags && j < distance_count {
            let mut k = 0;
            let mut sum_lag = 0.0;
            let mut sum_semi = 0.0;

            while j < distance_count && distance[j].0 <= ((i + 1) as f64 * tolerance) {
                sum_lag += distance[j].0;
                sum_semi += distance[j].1;
                j += 1;
                k += 1;
            }

            if k > 0 {
                lag[l] = sum_lag / k as f64;
                semi[l] = sum_semi / k as f64;
                l += 1;
            }
            i += 1;
        }

        if l < 2 {
            return variogram;
        }
    }

    // 特征变换
    let n_lag = lag.len();
    variogram.range = if n_lag > 1 {
        lag[n_lag - 1] - lag[0]
    } else {
        1.0
    };

    let mut x_mat = Array2::ones((n_lag, 2));
    let mut y_vec = Array1::zeros(n_lag);

    for i in 0..n_lag {
        match variogram.model {
            VariogramModel::Gaussian => {
                x_mat[[i, 1]] =
                    1.0 - (-(1.0 / variogram.a) * (lag[i] / variogram.range).powi(2)).exp();
            }
            VariogramModel::Exponential => {
                x_mat[[i, 1]] = 1.0 - (-(1.0 / variogram.a) * (lag[i] / variogram.range)).exp();
            }
            VariogramModel::Spherical => {
                x_mat[[i, 1]] =
                    1.5 * (lag[i] / variogram.range) - 0.5 * (lag[i] / variogram.range).powi(3);
            }
        }
        y_vec[i] = semi[i];
    }

    // 最小二乘
    let xt = matrix_transpose(&x_mat);
    let mut z = matrix_multiply(&xt, &x_mat);
    let diag = matrix_diag(1.0 / alpha, 2);
    z = matrix_add(&z, &diag);

    let mut clone_z = z.clone();
    let success = matrix_chol(&mut z);
    if success {
        matrix_chol2inv(&mut z);
    } else {
        matrix_solve(&mut clone_z);
        z = clone_z;
    }

    let w = matrix_multiply(&matrix_multiply(&z, &xt), &y_vec.insert_axis(Axis(1)));

    // 变异函数参数
    variogram.nugget = w[[0, 0]];
    variogram.sill = w[[1, 0]] * variogram.range + variogram.nugget;
    variogram.n = x.len();

    //  Gram 矩阵
    let n_gram = x.len();
    let mut k = Array2::zeros((n_gram, n_gram));

    for i in 0..n_gram {
        for j in 0..i {
            let dist = ((x[i] - x[j]).powi(2) + (y[i] - y[j]).powi(2)).sqrt();
            let value = match variogram.model {
                VariogramModel::Gaussian => variogram_gaussian(
                    dist,
                    variogram.nugget,
                    variogram.range,
                    variogram.sill,
                    variogram.a,
                ),
                VariogramModel::Exponential => variogram_exponential(
                    dist,
                    variogram.nugget,
                    variogram.range,
                    variogram.sill,
                    variogram.a,
                ),
                VariogramModel::Spherical => variogram_spherical(
                    dist,
                    variogram.nugget,
                    variogram.range,
                    variogram.sill,
                    variogram.a,
                ),
            };
            k[[i, j]] = value;
            k[[j, i]] = value;
        }
        k[[i, i]] = match variogram.model {
            VariogramModel::Gaussian => variogram_gaussian(
                0.0,
                variogram.nugget,
                variogram.range,
                variogram.sill,
                variogram.a,
            ),
            VariogramModel::Exponential => variogram_exponential(
                0.0,
                variogram.nugget,
                variogram.range,
                variogram.sill,
                variogram.a,
            ),
            VariogramModel::Spherical => variogram_spherical(
                0.0,
                variogram.nugget,
                variogram.range,
                variogram.sill,
                variogram.a,
            ),
        };
    }

    // 带惩罚的 Gram 矩阵求逆
    let mut c = matrix_add(&k, &matrix_diag(sigma2, n_gram));
    let mut clone_c = c.clone();
    let success = matrix_chol(&mut c);
    if success {
        matrix_chol2inv(&mut c);
    } else {
        matrix_solve(&mut clone_c);
        c = clone_c;
    }

    // 复制未投影的逆矩阵作为 K
    let k1 = c.clone();
    let m = matrix_multiply(&c, &Array1::from_vec(t.to_vec()).insert_axis(Axis(1)));

    variogram.k = k1;
    variogram.m = m.column(0).to_owned();

    variogram
}
// 预测函数
pub fn predict(x: f64, y: f64, variogram: &Variogram) -> f64 {
    let mut k = Array1::zeros(variogram.n);

    for i in 0..variogram.n {
        let dx = x - variogram.x[i];
        let dy = y - variogram.y[i];
        let func = variogram.model_func;
        k[i] = func(
            (dx * dx + dy * dy).sqrt(),
            variogram.nugget,
            variogram.range,
            variogram.sill,
            variogram.a,
        )
    }

    let result = matrix_multiply(
        &k.insert_axis(Axis(0)),
        &variogram.m.clone().insert_axis(Axis(1)),
    );
    result[[0, 0]]
}

// 方差函数
pub fn variance(x: f64, y: f64, variogram: &Variogram) -> f64 {
    let mut k = Array1::zeros(variogram.n);

    for i in 0..variogram.n {
        let dist = ((x - variogram.x[i]).powi(2) + (y - variogram.y[i]).powi(2)).sqrt();
        k[i] = match variogram.model {
            VariogramModel::Gaussian => variogram_gaussian(
                dist,
                variogram.nugget,
                variogram.range,
                variogram.sill,
                variogram.a,
            ),
            VariogramModel::Exponential => variogram_exponential(
                dist,
                variogram.nugget,
                variogram.range,
                variogram.sill,
                variogram.a,
            ),
            VariogramModel::Spherical => variogram_spherical(
                dist,
                variogram.nugget,
                variogram.range,
                variogram.sill,
                variogram.a,
            ),
        };
    }

    let val = matrix_multiply(
        &matrix_multiply(&k.clone().insert_axis(Axis(0)), &variogram.k),
        &k.clone().insert_axis(Axis(1)),
    );

    let model_val = match variogram.model {
        VariogramModel::Gaussian => variogram_gaussian(
            0.0,
            variogram.nugget,
            variogram.range,
            variogram.sill,
            variogram.a,
        ),
        VariogramModel::Exponential => variogram_exponential(
            0.0,
            variogram.nugget,
            variogram.range,
            variogram.sill,
            variogram.a,
        ),
        VariogramModel::Spherical => variogram_spherical(
            0.0,
            variogram.nugget,
            variogram.range,
            variogram.sill,
            variogram.a,
        ),
    };

    model_val + val[[0, 0]]
}

// 网格函数
pub fn grid(
    polygons: &Vec<Vec<(f64, f64)>>,
    variogram: &Variogram,
    width: f64,
) -> Option<GridResult> {
    let n = polygons.len();
    if n == 0 {
        return None;
    }

    // 计算多边形边界
    let mut xlim = (polygons[0][0].0, polygons[0][0].0);
    let mut ylim = (polygons[0][0].1, polygons[0][0].1);

    for polygon in polygons {
        for point in polygon {
            xlim.0 = xlim.0.min(point.0);
            xlim.1 = xlim.1.max(point.0);
            ylim.0 = ylim.0.min(point.1);
            ylim.1 = ylim.1.max(point.1);
        }
    }

    // 计算网格大小
    let x = ((xlim.1 - xlim.0) / width).ceil() as usize;
    let y = ((ylim.1 - ylim.0) / width).ceil() as usize;

    // 初始化网格数据
    let mut data = vec![vec![None; y + 1]; x + 1];

    // 遍历每个多边形
    for polygon in polygons {
        // 计算当前多边形的边界
        let mut lxlim = (polygon[0].0, polygon[0].0);
        let mut lylim = (polygon[0].1, polygon[0].1);

        for point in polygon {
            lxlim.0 = lxlim.0.min(point.0);
            lxlim.1 = lxlim.1.max(point.0);
            lylim.0 = lylim.0.min(point.1);
            lylim.1 = lylim.1.max(point.1);
        }

        // 计算网格范围
        let a0 = ((lxlim.0 - ((lxlim.0 - xlim.0) % width)) - xlim.0) / width;
        let a1 = ((lxlim.1 - ((lxlim.1 - xlim.1) % width)) - xlim.0) / width;
        let b0 = ((lylim.0 - ((lylim.0 - ylim.0) % width)) - ylim.0) / width;
        let b1 = ((lylim.1 - ((lylim.1 - ylim.1) % width)) - ylim.0) / width;

        let a0 = a0.floor() as usize;
        let a1 = a1.ceil() as usize;
        let b0 = b0.floor() as usize;
        let b1 = b1.ceil() as usize;

        // 遍历网格点
        for j in a0..=a1 {
            for k in b0..=b1 {
                let xtarget = xlim.0 + j as f64 * width;
                let ytarget = ylim.0 + k as f64 * width;

                if pip(polygon, xtarget, ytarget) {
                    let value = predict(xtarget, ytarget, variogram);
                    data[j][k] = Some(value);
                }
            }
        }
    }

    // 计算z轴范围
    let zlim = (min(&variogram.t), max(&variogram.t));

    Some(GridResult {
        xlim: [xlim.0, xlim.1],
        ylim: [ylim.0, ylim.1],
        width,
        data,
        zlim: [zlim.0, zlim.1],
    })
}
