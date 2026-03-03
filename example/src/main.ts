import * as L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import kriging from '@sakitam-gis/kriging';
import { colors, type releaseType } from './constant';
import * as turf from '@turf/turf';
import './main.css';
import initKrigingWasm, { interpolate_grid, type VariogramModel } from './kriging-wasm/kriging_wasm'


const tdt_key = '36294b070b7f84a954582df42f1b6cb1'
let map: L.Map | null = null;
let imageLayerGroup: L.FeatureGroup | null = null;
const initMap = () => {
  // 矢量地图
  const tiandituMap = new L.TileLayer(`http://t0.tianditu.gov.cn/cva_c/wmts?layer=cva&style=default&tilematrixset=c&Service=WMTS&Request=GetTile&Version=1.0.0&Format=tiles&TileMatrix={z}&TileCol={x}&TileRow={y}&tk=${tdt_key}`,
    {
      tileSize: 512,
      noWrap: true,
      bounds: [[-90, -180], [90, 180]]
    })
  // 文字注记
  const tiandituText = new L.TileLayer(`http://t0.tianditu.com/vec_c/wmts?layer=vec&style=default&tilematrixset=c&Service=WMTS&Request=GetTile&Version=1.0.0&Format=tiles&TileMatrix={z}&TileCol={x}&TileRow={y}&tk=${tdt_key}`,
    {
      tileSize: 512,
      noWrap: true,
      bounds: [[-90, -180], [90, 180]]
    })
  const layers = L.layerGroup([tiandituText, tiandituMap])
  map = L.map('myMap', {  //需绑定地图容器div的id
    center: [35.8617, 104.1954], //初始地图中心
    crs: L.CRS.EPSG4326,
    zoom: 5, //初始缩放等级
    maxZoom: 18, //最大缩放等级
    minZoom: 0, //最小缩放等级
    zoomControl: true, //缩放组件
    attributionControl: false, //去掉右下角logol
    scrollWheelZoom: true, //默认开启鼠标滚轮缩放
    // 限制显示地理范围
    maxBounds: L.latLngBounds(L.latLng(-90, -180), L.latLng(90, 180)),
    layers: [layers] // 图层
  })
  // 图像图层组
  imageLayerGroup = new L.FeatureGroup().addTo(map).bringToFront()
}
// 清空图层
const clearKriging = () => {
  imageLayerGroup?.clearLayers();
}

const generateData = () => {
  // 随机点的边界(折线的最大包围盒坐标)
  const boundaries = turf.lineString([[110, 32], [118, 40], [120, 35]]);
  // 随机50个点状要素数据
  const positionData = turf.randomPoint(50, { bbox: turf.bbox(boundaries) });
  // 再生成些随机数做属性
  turf.featureEach(positionData, function (currentFeature) {
    currentFeature.properties = { value: Number((Math.random() * 100).toFixed(2)) };
  });
  return { boundaries, positionData }
}

const showKriging = (type: releaseType) => {
  // 清空图层
  clearKriging();

  // 完全透明
  const scope = L.geoJSON(boundaries, {
    style: function () {
      return {
        fillColor: '6666ff',
        color: 'red',
        weight: 2,
        opacity: 0,
        fillOpacity: 0,
      };
    }
  }).addTo(imageLayerGroup!);

  map!.fitBounds(scope.getBounds());

  const canvas = document.createElement('canvas');
  canvas.width = 2000;
  canvas.height = 1000;

  // 将插值范围封装成特定格式
  const bbox = turf.bbox(boundaries); // 外包矩形范围
  // 根据外包矩形范围生成外包矩形面Polygon
  const bboxPolygon = turf.bboxPolygon(bbox);
  const range: [number, number][][] = bboxPolygon.geometry.coordinates.map(coordinate => {
    return coordinate.map(v => [v[0], v[1]])
  })
  
  // 克里金插值参数
  const params = {
    krigingModel: 'Exponential',//model还可选'gaussian','spherical'
    krigingSigma2: 0,
    krigingAlpha: 100,
    interval: 0.05, // 插值格点精度大小
    canvasAlpha: 0.8,//canvas图层透明度-0.75
    colors
  }

  function jsKriging() {
    performance.mark('js start');
    const points = positionData
    // 数量
    const pointLength = points.features.length;
    const t = [];// 数值
    const x = [];// 经度
    const y = [];// 纬度
    // 加载点数过多的话，会出现卡顿
    for (let i = 0; i < pointLength; i++) {
      x.push(points.features[i].geometry.coordinates[0]);
      y.push(points.features[i].geometry.coordinates[1]);
      t.push(points.features[i].properties.value);
    }

    // 对数据集进行训练
    const variogram = kriging.train(t, x, y, params.krigingModel.toLowerCase(), params.krigingSigma2, params.krigingAlpha);
    
    // 使用variogram对象使polygons描述的地理位置内的格网元素具备不一样的预测值,最后一个参数，是插值格点精度大小
    const grid = kriging.grid(range, variogram, params.interval);
    console.log(grid!.data.reduce((acc, cur) => {
      const val = cur.filter((v: unknown) => v !== null && v !== undefined).length
      return acc + val
    }, 0))
    // 将得到的格网grid渲染至canvas上
    kriging.plot(canvas, grid!, grid!.xlim, grid!.ylim, params.colors);
    
    const imageBounds: L.LatLngBoundsExpression = range[0].map(v => [v[1], v[0]]);
    L.imageOverlay(canvas.toDataURL("image/png"), imageBounds, { opacity: 0.8 }).addTo(imageLayerGroup!);
    performance.mark('js end');
    console.log(performance.measure('jsKriging', 'js start', 'js end'));
  }

  function wasmKriging() {
    performance.mark('wasm start');
    const zlim = [positionData.features[0].properties.value, positionData.features[0].properties.value];
    const { points, xlim, ylim } = interpolate_grid({ 
      base: {
        kriging_type: 'Ordinary',
        model_type: params.krigingModel as VariogramModel,
        nugget: params.krigingSigma2,
        range: params.krigingAlpha,
        sill: null
      },
      known_points: positionData.features.map(feature => {
        zlim[0] = Math.min(zlim[0], feature.properties.value);
        zlim[1] = Math.max(zlim[1], feature.properties.value);
        return {
          point: feature.geometry.coordinates as [number, number],
          value: feature.properties.value,
        }
      }),
      polygons: range,
      interval: params.interval,
    })
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const range = [xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]];
      const wx = Math.ceil(params.interval * canvas.width / (xlim[1] - xlim[0]));
      const wy = Math.ceil(params.interval * canvas.height / (ylim[1] - ylim[0]));
      for (const point of points) {
        const x = canvas.width * (point.point[0] - xlim[0]) / range[0];
        const y = canvas.height * (1 - (point.point[1] - ylim[0]) / range[1]);
        let z = (point.value - zlim[0]) / range[2];
        if (z < 0.0) z = 0.0;
        if (z > 1.0) z = 1.0;
        ctx.fillStyle = colors[Math.floor((colors.length - 1) * z)];
        ctx.fillRect(Math.round(x - wx / 2), Math.round(y - wy / 2), wx, wy);
      }
    }
    const imageBounds: L.LatLngBoundsExpression = range[0].map(v => [v[1], v[0]]);;
    L.imageOverlay(canvas.toDataURL("image/png"), imageBounds, { opacity: 0.8 }).addTo(imageLayerGroup!);
    
    performance.mark('wasm end');
    console.log(performance.measure('wasmKriging', 'wasm start', 'wasm end'));
  }

  // 执行克里金插值函数
  if (type === 'wasm') {
    wasmKriging()
  } else {
    jsKriging();
  }
}

declare global {
  interface Window {
    showKriging: (type: releaseType) => void;
  }
}

const { boundaries, positionData } = generateData()

initKrigingWasm().then(() => {
  initMap()
  window.showKriging = showKriging
})
