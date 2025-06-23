import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import path from 'path';
import copy from 'rollup-plugin-copy';

export default defineConfig({
  base: "./",
  plugins: [
    vue(),
    copy({
      targets: [
        { src: 'node_modules/@tensorflow/tfjs-tflite/dist/tflite_web_api_cc*.js', dest: 'public/tf-wasm' },
        { src: 'node_modules/@tensorflow/tfjs-tflite/dist/tflite_web_api_cc*.wasm', dest: 'public/tf-wasm' },
      ]
    })
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  }
});
