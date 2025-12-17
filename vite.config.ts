import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: './', 
  define: {
    'process.env.API_KEY': JSON.stringify("AIzaSyCZSqy5CN2DSbm0BWii3k7z4qtrhitzB8Q")
  }
})