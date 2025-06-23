<template>
  <ion-page class="page-transparent">
    <ion-content fullscreen class="content-transparent">
      <div id="cameraPreview" class="camera-preview"></div>
      <canvas ref="canvas" class="bounding-box-canvas"></canvas>

      <!-- Stack ảnh đối tượng đã cắt -->
      <div class="detected-stack" v-if="snapshots.length">
        <div v-for="(snap, idx) in snapshots" :key="idx" class="detected-item">
          <img :src="snap.src" class="snap-img" />
          <p class="snap-label">{{ snap.label }}</p>
        </div>
      </div>
    </ion-content>
  </ion-page>
</template>

<script setup lang="ts">
console.log('[VD] VideoDetector.vue setup chạy');
import { ref, onMounted, onBeforeUnmount, nextTick } from 'vue';
import { CameraPreview, CameraPreviewOptions } from '@capacitor-community/camera-preview';
import * as tf from '@tensorflow/tfjs';
import { loadTFLiteModel, TFLiteModel } from '@tensorflow/tfjs-tflite';
import { TextToSpeech } from '@capacitor-community/text-to-speech';

console.log('🚀 VideoDetector module loaded');

import rawMapping from '@/assets/label_mapping.json';
import vnMappingJson from '@/assets/label_vn_mapping.json';
const vnMapping = vnMappingJson as Record<string, string>;

const canvas = ref<HTMLCanvasElement | null>(null);
const displayLabel = ref<string | null>(null);

// Danh sách ảnh cắt đối tượng
const snapshots = ref<{ src: string; label: string }[]>([]);

let intervalId: number;

const labelMapping = rawMapping as Record<string, string>;
let tfliteModel: TFLiteModel | null = null;
const MODEL_INPUT_WIDTH = 640;
const MODEL_INPUT_HEIGHT = 640;

let lastSpokenSeq = '';

// Ngưỡng confidence (từ 0 đến 1)
const confThreshold = 0.5;

// Cấu hình smoothing: số frame để gom vote nhãn
const historySize = 5;
const predictionHistory: number[] = [];

// Hàm tính mode – trả về giá trị xuất hiện nhiều nhất trong mảng
function mode(arr: number[]): number {
  const freq = arr.reduce((acc, v) => {
    acc[v] = (acc[v] || 0) + 1;
    return acc;
  }, {} as Record<number, number>);
  return Number(
    Object.entries(freq)
      .sort(([, a], [, b]) => (b as number) - (a as number))[0][0]
  );
}

// Đọc nhiều nhãn theo thứ tự confidence (cao→thấp)
async function speakLabels(labels: string[]) {
  const seq = labels.join('|');
  if (!labels.length || seq === lastSpokenSeq) return;
  lastSpokenSeq = seq;

  for (const lb of labels) {
    try {
      await TextToSpeech.speak({ text: lb, lang: 'vi-VN', rate: 1.0 });
    } catch (e) {
      console.warn('TTS error', e);
    }
  }
}

// speakLabel (đơn) vẫn giữ để dùng lại nếu cần
async function speakLabel(label: string) {
  await speakLabels([label]);
}

async function initTFLiteModel() {
  console.log('⚙️ initTFLiteModel() start');
  const url = import.meta.env.BASE_URL + 'assets/models/yolov8n_final_float32_nms.tflite';
  tfliteModel = await loadTFLiteModel(url);
  console.log('✅ TFLite model ready');
  
  // Test model với tensor zeros
  const testInput = tf.zeros([1, 640, 640, 3]);
  const testOutput = tfliteModel.predict(testInput);
  
  // Kiểm tra type của testOutput
  if (testOutput instanceof tf.Tensor) {
    const tensor = testOutput as tf.Tensor;
    console.log('Test output shape:', tensor.shape);
    console.log('Test output sample:', await tensor.array());
  } else if (Array.isArray(testOutput)) {
    const firstTensor = testOutput[0] as tf.Tensor;
    console.log('Test output is array of tensors');
    console.log('First tensor shape:', firstTensor.shape);
    console.log('First tensor sample:', await firstTensor.array());
  } else {
    const outMap = testOutput as Record<string, tf.Tensor>;
    const key = outMap['Identity'] ? 'Identity' : Object.keys(outMap)[0];
    const tensor = outMap[key] as tf.Tensor;
    console.log('Test output is named tensor map');
    console.log('Output tensor shape:', tensor.shape);
    console.log('Output tensor sample:', await tensor.array());
  }
}

// ---
// Tiền xử lý theo phong cách "letterbox" – giữ nguyên tỉ lệ, thêm padding để
// vừa khít khung 640×640 giống YOLO. Hàm trả lại tensor đã sẵn sàng cho model
// kèm theo các tham số để map ngược bbox về ảnh gốc.
interface PreprocessResult {
  tensor: tf.Tensor4D;   // [1, 640, 640, 3]
  r: number;             // scale ratio (0‒1]
  padX: number;          // padding ở chiều X (pixel, trên input 640)
  padY: number;          // padding ở chiều Y (pixel)
  srcW: number;          // kích thước ảnh gốc
  srcH: number;
}

function preprocessFrame(bitmap: ImageBitmap): PreprocessResult {
  // Chuyển ảnh thành tensor [H, W, 3]
  const img = tf.browser.fromPixels(bitmap);
  const [srcH, srcW] = img.shape as number[];

  // Tỉ lệ co
  const r = Math.min(MODEL_INPUT_WIDTH / srcW, MODEL_INPUT_HEIGHT / srcH);
  const newW = Math.round(srcW * r);
  const newH = Math.round(srcH * r);

  // Padding để vừa 640×640
  const padX = Math.floor((MODEL_INPUT_WIDTH - newW) / 2);
  const padY = Math.floor((MODEL_INPUT_HEIGHT - newH) / 2);

  // Resize + pad
  const processed = tf.tidy(() => {
    const resized = tf.image.resizeBilinear(img, [newH, newW]);
    const padded  = tf.pad(resized, [[padY, MODEL_INPUT_HEIGHT - newH - padY],
                                     [padX, MODEL_INPUT_WIDTH  - newW - padX],
                                     [0, 0]]);
    const normalized = padded.div(255);
    return normalized.expandDims(0) as tf.Tensor4D; // [1,640,640,3]
  });

  // img tensor đã dùng xong, giải phóng
  tf.dispose(img);

  return { tensor: processed, r, padX, padY, srcW, srcH };
}

// 1) Định nghĩa interface cho box
interface Box {
  x1: number;  y1: number;
  x2: number;  y2: number;
  score: number;
  classId: number;
}

async function runInference(bitmap: ImageBitmap): Promise<Box[]> {
  if (!tfliteModel) {
    console.error('Model not loaded');
    return [];
  }

  // 1) Tiền xử lý ảnh đầu vào
  const { tensor, r, padX, padY, srcW, srcH } = preprocessFrame(bitmap);

  // 2) Thực hiện suy luận
  const rawOutput = tfliteModel.predict(tensor);

  // Các biến sẽ chứa tensor đầu ra tuỳ thuộc vào cấu trúc model
  let boxesTensor: tf.Tensor | undefined;
  let classesTensor: tf.Tensor | undefined;
  let scoresTensor: tf.Tensor | undefined;
  let countTensor: tf.Tensor | undefined;

  // a) Trường hợp phổ biến: model trả về mảng 4 tensor
  if (Array.isArray(rawOutput) && rawOutput.length >= 4) {
    [boxesTensor, classesTensor, scoresTensor, countTensor] = rawOutput as tf.Tensor[];
  }
  // b) Trường hợp trả về map tên-tensor
  else if (typeof rawOutput === 'object' && !(rawOutput instanceof tf.Tensor)) {
    const outMap = rawOutput as Record<string, tf.Tensor>;
    // Sắp xếp key để lấy theo thứ tự ổn định (boxes, classes, scores, count)
    const keys = Object.keys(outMap).sort();
    boxesTensor   = outMap[keys[0]];
    classesTensor = outMap[keys[1]];
    scoresTensor  = outMap[keys[2]];
    countTensor   = outMap[keys[3]];
  }
  // c) Trường hợp cuối: 1 tensor, shape [1, N, 6] (x1, y1, x2, y2, score, class)
  else if (rawOutput instanceof tf.Tensor) {
    boxesTensor = rawOutput;
  }

  const boxes: Box[] = [];

  // ---- Giải mã output khi đã có các tensor riêng biệt ----
  if (boxesTensor && classesTensor && scoresTensor) {
    const [boxesArr, classesArr, scoresArr, countArr] = await Promise.all([
      boxesTensor.array() as Promise<number[][][]>,
      classesTensor.array() as Promise<number[][]>,
      scoresTensor.array() as Promise<number[][]>,
      countTensor ? countTensor.array() as Promise<number[]> : Promise.resolve([ (boxesTensor.shape[1] || 0) ])
    ]);

    const numDetections = Math.min(countArr[0] ?? boxesArr[0].length, boxesArr[0].length);

    for (let i = 0; i < numDetections; i++) {
      const score = scoresArr[0][i];
      if (score < confThreshold) continue;

      // TFLite Detection PostProcess xuất [ymin, xmin, ymax, xmax]
      const [ymin, xmin, ymax, xmax] = boxesArr[0][i];
      const cls = Math.round(classesArr[0][i]);

      const x1Pix = ((xmin * MODEL_INPUT_WIDTH)  - padX) / r;
      const y1Pix = ((ymin * MODEL_INPUT_HEIGHT) - padY) / r;
      const x2Pix = ((xmax * MODEL_INPUT_WIDTH)  - padX) / r;
      const y2Pix = ((ymax * MODEL_INPUT_HEIGHT) - padY) / r;

      // Chuẩn hoá về 0‒1 theo kích thước ảnh gốc
      const x1 = Math.max(0, Math.min(1, x1Pix / srcW));
      const y1 = Math.max(0, Math.min(1, y1Pix / srcH));
      const x2 = Math.max(0, Math.min(1, x2Pix / srcW));
      const y2 = Math.max(0, Math.min(1, y2Pix / srcH));

      boxes.push({
        x1,
        y1,
        x2,
        y2,
        score,
        classId: cls,
      });
    }

    tf.dispose([tensor, boxesTensor, classesTensor, scoresTensor]);
    if (countTensor) tf.dispose(countTensor);
  }
  // ---- Giải mã output khi model cho 1 tensor [1, N, 6] ----
  else if (boxesTensor) {
    const arr = await boxesTensor.array() as number[][][]; // [1, N, 6]
    const data = arr[0];

    for (let i = 0; i < data.length; i++) {
      const [rawX1, rawY1, rawX2, rawY2, score, clsFloat] = data[i] as unknown as number[];
      if (score < confThreshold) continue;
      const cls = Math.round(clsFloat);

      const x1Pix = ((rawX1 * MODEL_INPUT_WIDTH)  - padX) / r;
      const y1Pix = ((rawY1 * MODEL_INPUT_HEIGHT) - padY) / r;
      const x2Pix = ((rawX2 * MODEL_INPUT_WIDTH)  - padX) / r;
      const y2Pix = ((rawY2 * MODEL_INPUT_HEIGHT) - padY) / r;

      // Chuẩn hoá về 0‒1 theo kích thước ảnh gốc
      const x1 = Math.max(0, Math.min(1, x1Pix / srcW));
      const y1 = Math.max(0, Math.min(1, y1Pix / srcH));
      const x2 = Math.max(0, Math.min(1, x2Pix / srcW));
      const y2 = Math.max(0, Math.min(1, y2Pix / srcH));

      boxes.push({ x1, y1, x2, y2, score, classId: cls });
    }

    tf.dispose([tensor, boxesTensor]);
  }

  // Sắp xếp để bounding-box có score cao nhất đứng đầu
  boxes.sort((a, b) => b.score - a.score);

  return boxes;
}

function drawAndSpeak(boxes: Box[], bitmap: ImageBitmap, videoW: number, videoH: number) {
  const cv = canvas.value!;
  const ctx = cv.getContext('2d')!;
  ctx.clearRect(0, 0, cv.width, cv.height);

  if (!boxes.length) {
    displayLabel.value = null;
    snapshots.value = [];
    return;
  }

  // Reset snapshots mỗi khung hình
  snapshots.value = [];

  // Vẽ tất cả các bounding-box
  boxes.forEach(b => {
    const x = b.x1 * videoW;
    const y = b.y1 * videoH;
    const w = (b.x2 - b.x1) * videoW;
    const h = (b.y2 - b.y1) * videoH;
    
    ctx.strokeStyle = 'red';
    ctx.lineWidth   = 2;
    ctx.strokeRect(x, y, w, h);

    const className = labelMapping[b.classId.toString()] || b.classId.toString();
    // Hiển thị mã nhãn (className) trên bounding box
    ctx.font      = '16px sans-serif';
    ctx.fillStyle = 'red';
    ctx.fillText(
      className + ' (' + Math.round(b.score * 100) + '%)', 
      x,
      y > 20 ? y - 5 : y + 20
    );

    // --- Cắt và lưu ảnh đối tượng ---
    try {
      // Toạ độ theo bitmap gốc
      const sx = Math.max(0, Math.round(b.x1 * bitmap.width));
      const sy = Math.max(0, Math.round(b.y1 * bitmap.height));
      const sw = Math.max(1, Math.round((b.x2 - b.x1) * bitmap.width));
      const sh = Math.max(1, Math.round((b.y2 - b.y1) * bitmap.height));

      // Canvas tạm để crop & downscale (128px max)
      const maxDim = 128;
      const scale   = Math.min(1, maxDim / Math.max(sw, sh));
      const dw = Math.round(sw * scale);
      const dh = Math.round(sh * scale);

      const tmpCanvas = document.createElement('canvas');
      tmpCanvas.width  = dw;
      tmpCanvas.height = dh;

      const tmpCtx = tmpCanvas.getContext('2d')!;
      tmpCtx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, dw, dh);
      const dataUrl = tmpCanvas.toDataURL('image/jpeg', 0.8);

      snapshots.value.push({ src: dataUrl, label: vnMapping[className] || className });
    } catch (e) {
      console.warn('Crop error', e);
    }
  });

  // --- Chuẩn bị danh sách nhãn theo thứ tự confidence cao → thấp ---
  const spokenLabels: string[] = [];
  boxes.forEach(b => {
    const nameEn = labelMapping[b.classId.toString()] || b.classId.toString();
    const vnLabel = vnMapping[nameEn] || nameEn;
    if (!spokenLabels.includes(vnLabel)) spokenLabels.push(vnLabel);
  });

  if (spokenLabels.length) {
    // Nếu ≥2 đối tượng đọc lần lượt, ngược lại đọc 1 đối tượng
    speakLabels(spokenLabels);
  }
}

async function captureFrame(): Promise<ImageBitmap | null> {
  try {
    // Lấy frame đã down-scale, mặc định ~previewSize
    const { value: b64 } = await CameraPreview.captureSample({ quality: 60 });

    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    return await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }));
  } catch (e) {
    console.error('captureFrame error', e);
    return null;
  }
}

let isProcessing = false;

async function processLoop() {
  if (isProcessing || !tfliteModel) return;
  isProcessing = true;

  const bitmap = await captureFrame();
  if (!bitmap) {
    isProcessing = false;
    return;
  }

  const cv = canvas.value;
  if (!cv) {
    bitmap.close();
    isProcessing = false;
    return;
  }

  try {
    const dets = await runInference(bitmap);
    if (dets.length) {
      drawAndSpeak(dets, bitmap, cv.width, cv.height);
    } else {
      // Không có detections: xóa canvas và nhãn
      const ctx = cv.getContext('2d')!;
      ctx.clearRect(0, 0, cv.width, cv.height);
      displayLabel.value = null;
    }
  } catch (e) {
    console.error('Inference error', e);
  } finally {
    bitmap.close();
    isProcessing = false;
  }
}

async function startCamera() {
  // 1) Đảm bảo camera cũ dừng
  try { await CameraPreview.stop() } catch {}

  // 2) Đợi DOM render xong
  await nextTick()

  // 3) Lấy kích thước thực của div#cameraPreview
  const previewEl = document.getElementById('cameraPreview')
  const vw = previewEl?.clientWidth  || window.innerWidth
  const vh = previewEl?.clientHeight || window.innerHeight
  console.log('ℹ️ cameraPreview size:', vw, 'x', vh)

  // 4) Cấu hình và start
  const opts: CameraPreviewOptions = {
    parent: 'cameraPreview',
    width:  vw,
    height: vh,
    position: 'rear',
    toBack: true,
  }

  try {
    await CameraPreview.start(opts)
    console.log('✅ CameraPreview.start() ok')
  } catch (e: any) {
    console.error('❌ Không thể start camera:', e)
    return
  }

  // 5) Thiết lập canvas cũng bằng kích thước này
  if (canvas.value) {
    canvas.value.width  = vw
    canvas.value.height = vh
  }

  // 6) Bắt đầu vòng lặp inference
  if (intervalId) clearInterval(intervalId)
  intervalId = window.setInterval(processLoop, 250)
}


async function stopCamera() {
  clearInterval(intervalId);
  try {
    await CameraPreview.stop();
  } catch (e) {
    console.warn('Lỗi khi stop camera:', e);
  }
}

onMounted(async () => {
  console.log('🔔 onMounted() fired')

  // 1) Đợi TF.js khởi xong
  console.log('🧠 waiting tf.ready()...')
  await tf.ready()
  console.log('🧠 tf.ready() done')

  try {
    // 2) Khởi tạo TFLite và camera
    await initTFLiteModel()
    await startCamera()
  } catch (e) {
    console.error('Init error', e)
  }
})

onBeforeUnmount(async () => {
  clearInterval(intervalId)
  try {
    await CameraPreview.stop()
  } catch {}
})
</script>

<style scoped>
.page-transparent {
  --background: transparent !important;
}

.content-transparent {
  --background: transparent !important;
}

.camera-preview {
  position: absolute;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  z-index: 0;
}

.bounding-box-canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  pointer-events: none;
  z-index: 10;
}

.detected-stack {
  position: absolute;
  left: 10px;
  bottom: 10px;
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 8px;
  z-index: 30;
}

.detected-item {
  background: rgba(0,0,0,0.6);
  padding: 4px;
  border-radius: 4px;
  text-align: center;
}

.snap-img {
  max-width: 128px;
  max-height: 128px;
  display: block;
}

.snap-label {
  color: white;
  font-size: 12px;
  margin: 2px 0 0 0;
}
</style>