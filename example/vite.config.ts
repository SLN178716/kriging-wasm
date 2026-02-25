import { defineConfig } from 'vite'
// @ts-expect-error: 暂无解决
import eslintPlugin from 'vite-plugin-eslint';

export default defineConfig(() => {
  return {
    base: '/',
    plugins: [eslintPlugin()],
    resolve: {
      alias: {
        '@': './src',
      },
    },
    server: {
      port: 21101,
      host: '0.0.0.0',
      cors: true,
      strictPort: true,
      open: true,
    }
  }
})