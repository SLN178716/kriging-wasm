import * as L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import kriging from '@sakitam-gis/kriging';
import { colors, type releaseType } from './constant';
import * as turf from '@turf/turf';
import './main.css';


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
    currentFeature.properties = { value: (Math.random() * 100).toFixed(2) };
  });
  return { boundaries, positionData }
}

const showKriging = (type: releaseType) => {
  if (type !== 'js') {
    return
  }
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
  //根据scope边界线，生成范围信息
  const xlim = [scope.getBounds().getSouthWest().lng, scope.getBounds().getNorthEast().lng];
  const ylim = [scope.getBounds().getSouthWest().lat, scope.getBounds().getNorthEast().lat];

  const canvas = document.createElement('canvas');
  canvas.width = 2000;
  canvas.height = 1000;
  function loadkriging() {
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

    // 克里金插值参数
    const params = {
      krigingModel: 'exponential',//model还可选'gaussian','spherical'
      krigingSigma2: 0,
      krigingAlpha: 100,
      canvasAlpha: 0.8,//canvas图层透明度-0.75
      colors
    }
    // 对数据集进行训练
    const variogram = kriging.train(t, x, y, params.krigingModel, params.krigingSigma2, params.krigingAlpha);
    // 将插值范围封装成特定格式
    const bbox = turf.bbox(boundaries); // 外包矩形范围
    // 根据外包矩形范围生成外包矩形面Polygon
    const bboxPolygon = turf.bboxPolygon(bbox);
    const positions: number[][] = [];
    bboxPolygon.geometry.coordinates[0].forEach((v) => {
      positions.push([v[0], v[1]])
    })
    // 将边界封装成特定的格式
    const range = [positions]
    // 使用variogram对象使polygons描述的地理位置内的格网元素具备不一样的预测值,最后一个参数，是插值格点精度大小
    const grid = kriging.grid(range, variogram, 0.05);
    // 将得到的格网grid渲染至canvas上
    kriging.plot(canvas, grid!, xlim, ylim, params.colors);
  }

  // 执行克里金插值函数
  loadkriging();

  const imageBounds: L.LatLngBoundsExpression = [[ylim[0], xlim[0]], [ylim[1], xlim[1]]];
  L.imageOverlay(canvas.toDataURL("image/png"), imageBounds, { opacity: 0.8 }).addTo(imageLayerGroup!);
}

initMap()
const { boundaries, positionData } = generateData()
showKriging('js')
