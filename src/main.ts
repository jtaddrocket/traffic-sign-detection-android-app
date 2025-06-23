// src/main.ts
import { createApp } from 'vue'
import App from './App.vue'
import { IonicVue } from '@ionic/vue'
import { defineCustomElements } from '@ionic/pwa-elements/loader';
defineCustomElements(window);
import { setWasmPath } from '@tensorflow/tfjs-tflite'

console.log('[BOOT] main.ts bắt đầu');

async function bootstrap() {
  console.log('[BOOT] vào bootstrap()');
  try {
    // thiết lập đúng đường dẫn tới tf-wasm
    setWasmPath(import.meta.env.BASE_URL + 'tf-wasm/')

    // load TF.js và TFLite
    await import('@tensorflow/tfjs')
    await import('@tensorflow/tfjs-tflite')
    console.log('[BOOT] TFJS & TFLite đã load');

    const app = createApp(App)
      .use(IonicVue)       // chỉ cần IonicVue
    console.log('[BOOT] Đã tạo App, chuẩn bị mount');

    // mount ngay, không chờ router
    app.mount('#app')
    console.log('[BOOT] Đã mount App lên #app');
    defineCustomElements(window)
  } catch (err) {
    console.error('[BOOT] Lỗi trong bootstrap:', err);
  }
}

bootstrap()
