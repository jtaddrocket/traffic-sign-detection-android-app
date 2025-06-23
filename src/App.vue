<!-- src/App.vue -->
<template>
  <ion-app>
    <VideoDetector />
  </ion-app>
</template>

<script setup lang="ts">
console.log('[APP] App.vue script setup chạy');
import VideoDetector from '@/composables/VideoDetector.vue'
import { defineCustomElements } from '@ionic/pwa-elements/loader'

defineCustomElements(window)
</script>
