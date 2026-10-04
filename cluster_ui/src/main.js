import { createApp } from 'vue'
import App from './App.vue'
import './style.css'

// Optional forced theme for producing print figures: ?theme=light | ?theme=dark
// (no visible UI; without the param the tool follows the OS colour scheme).
const forced = new URLSearchParams(location.search).get('theme')
if (forced === 'light' || forced === 'dark') {
  document.documentElement.dataset.theme = forced
}

createApp(App).mount('#app')
