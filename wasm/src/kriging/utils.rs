use ndarray::{Array1, Array2};

// 变异函数计算
pub fn variogram_gaussian(h: f64, nugget: f64, range: f64, sill: f64, a: f64) -> f64 {
    let r = h / range;
    nugget + ((sill - nugget) / range) * (1.0 - ((- (1.0 / a) * r * r) as f32).exp() as f64)
}

// 变异函数计算
pub fn variogram_exponential(h: f64, nugget: f64, range: f64, sill: f64, a: f64) -> f64 {
    nugget + ((sill - nugget) / range) * (1.0 - ((- (1.0 / a) * (h / range)) as f32).exp() as f64)
}

// 变异函数计算
pub fn variogram_spherical(h: f64, nugget: f64, range: f64, sill: f64, _a: f64) -> f64 {
    if h > range {
        return nugget + (sill - nugget) / range;
    }
    let r = h / range;
    nugget + ((sill - nugget) / range) * (1.5 * r - 0.5 * r * r * r)
}

// 辅助函数
pub fn max(arr: &[f64]) -> f64 {
    arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

pub fn min(arr: &[f64]) -> f64 {
    arr.iter().cloned().fold(f64::INFINITY, f64::min)
}

pub fn mean(arr: &[f64]) -> f64 {
    let sum: f64 = arr.iter().sum();
    sum / arr.len() as f64
}

pub fn rep(value: f64, n: usize) -> Vec<f64> {
    vec![value; n]
}

// 点在多边形内检测
pub fn pip(polygon: &[(f64, f64)], x: f64, y: f64) -> bool {
    let mut c = false;
    let n = polygon.len();
    let mut j = n - 1;
    
    for i in 0..n {
        if ((polygon[i].1 > y) != (polygon[j].1 > y)) && 
           (x < (polygon[j].0 - polygon[i].0) * (y - polygon[i].1) / 
            (polygon[j].1 - polygon[i].1) + polygon[i].0) {
            c = !c;
        }
        j = i;
    }
    c
}

// 矩阵运算
pub fn matrix_diag(c: f64, n: usize) -> Array2<f64> {
    let mut z = Array2::zeros((n, n));
    for i in 0..n {
        z[[i, i]] = c;
    }
    z
}

pub fn matrix_transpose(x: &Array2<f64>) -> Array2<f64> {
    x.t().to_owned()
}

pub fn matrix_add(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    x + y
}

pub fn matrix_multiply(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    x.dot(y)
}

// Cholesky 分解
pub fn matrix_chol(x: &mut Array2<f64>) -> bool {
    let n = x.nrows();
    let mut p = Array1::zeros(n);
    
    for i in 0..n {
        p[i] = x[[i, i]];
    }
    
    for i in 0..n {
        for j in 0..i {
            p[i] -= x[[i, j]] * x[[i, j]];
        }
        if p[i] <= 0.0 {
            return false;
        }
        p[i] = p[i].sqrt();
        for j in i+1..n {
            for k in 0..i {
                x[[j, i]] -= x[[j, k]] * x[[i, k]];
            }
            x[[j, i]] /= p[i];
        }
    }
    
    for i in 0..n {
        x[[i, i]] = p[i];
    }
    true
}

// Cholesky 分解求逆
pub fn matrix_chol2inv(x: &mut Array2<f64>) {
    let n = x.nrows();
    
    for i in 0..n {
        x[[i, i]] = 1.0 / x[[i, i]];
        for j in i+1..n {
            let mut sum = 0.0;
            for k in i..j {
                sum -= x[[j, k]] * x[[k, i]];
            }
            x[[j, i]] = sum / x[[j, j]];
        }
    }
    
    for i in 0..n {
        for j in i+1..n {
            x[[i, j]] = 0.0;
        }
    }
    
    for i in 0..n {
        x[[i, i]] *= x[[i, i]];
        for k in i+1..n {
            x[[i, i]] += x[[k, i]] * x[[k, i]];
        }
        for j in i+1..n {
            for k in j..n {
                x[[i, j]] += x[[k, i]] * x[[k, j]];
            }
        }
    }
    
    for i in 0..n {
        for j in 0..i {
            x[[i, j]] = x[[j, i]];
        }
    }
}

// 高斯-约旦消元法求逆
pub fn matrix_solve(x: &mut Array2<f64>) -> bool {
    let n = x.nrows();
    let m = n;
    let mut b = Array2::eye(n);
    let mut indxc = vec![0; n];
    let mut indxr = vec![0; n];
    let mut ipiv = vec![0; n];
    
    for j in 0..n {
        ipiv[j] = 0;
    }
    
    for i in 0..n {
        let mut big = 0.0;
        let mut irow = 0;
        let mut icol = 0;
        
        for j in 0..n {
            if ipiv[j] != 1 {
                for k in 0..n {
                    if ipiv[k] == 0 {
                        if x[[j, k]].abs() >= big {
                            big = x[[j, k]].abs();
                            irow = j;
                            icol = k;
                        }
                    }
                }
            }
        }
        
        ipiv[icol] += 1;
        
        if irow != icol {
            for l in 0..n {
                let temp = x[[irow, l]];
                x[[irow, l]] = x[[icol, l]];
                x[[icol, l]] = temp;
            }
            for l in 0..m {
                let temp = b[[irow, l]];
                b[[irow, l]] = b[[icol, l]];
                b[[icol, l]] = temp;
            }
        }
        
        indxr[i] = irow;
        indxc[i] = icol;
        
        if x[[icol, icol]] == 0.0 {
            return false;
        }
        
        let pivinv = 1.0 / x[[icol, icol]];
        x[[icol, icol]] = 1.0;
        for l in 0..n {
            x[[icol, l]] *= pivinv;
        }
        for l in 0..m {
            b[[icol, l]] *= pivinv;
        }
        
        for ll in 0..n {
            if ll != icol {
                let dum = x[[ll, icol]];
                x[[ll, icol]] = 0.0;
                for l in 0..n {
                    x[[ll, l]] -= x[[icol, l]] * dum;
                }
                for l in 0..m {
                    b[[ll, l]] -= b[[icol, l]] * dum;
                }
            }
        }
    }
    
    for l in (0..n).rev() {
        if indxr[l] != indxc[l] {
            for k in 0..n {
                let temp = x[[k, indxr[l]]];
                x[[k, indxr[l]]] = x[[k, indxc[l]]];
                x[[k, indxc[l]]] = temp;
            }
        }
    }
    
    // 将结果复制回 x
    for i in 0..n {
        for j in 0..n {
            x[[i, j]] = b[[i, j]];
        }
    }
    
    true
}
