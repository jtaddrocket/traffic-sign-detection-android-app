// src/router/index.ts
import { createRouter, createWebHistory } from '@ionic/vue-router';
import { RouteRecordRaw } from 'vue-router';
import VideoDetector from '@/composables/VideoDetector.vue';

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    component: VideoDetector
  }
];

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
});

export default router;
