use kriging_wasm::{kriging_interpolate, version};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_version() {
    assert!(!version().is_empty());
}

#[wasm_bindgen_test]
fn test_kriging() {
    use serde_json::json;

    // 准备测试数据
    let input = json!({
        "points": [
            {"x": 0.0, "y": 0.0},
            {"x": 1.0, "y": 0.0},
            {"x": 0.0, "y": 1.0},
            {"x": 1.0, "y": 1.0}
        ],
        "values": [10.0, 20.0, 30.0, 40.0],
        "target_points": [
            {"x": 0.5, "y": 0.5}
        ],
        "model_type": "spherical"
    });

    let result = kriging_interpolate(JsValue::from_serde(&input).unwrap());
    assert!(result.is_ok());
}
