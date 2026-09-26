// Minimal static file server for the boxing demo. Run: node server.mjs [port]
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const port = Number(process.argv[2] || 8080);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.wasm': 'application/wasm',
  '.stl': 'model/stl',
  '.xml': 'text/xml; charset=utf-8',
  '.png': 'image/png',
  '.d.ts': 'text/plain; charset=utf-8',
};

http.createServer((req, res) => {
  // 策略/场景更新频繁：一律禁缓存，避免浏览器拿到旧策略看不到新动作
  res.setHeader('Cache-Control', 'no-cache');
  const urlPath = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  // 老页面的 index.html 可能被浏览器长期缓存（历史响应没带禁缓存头）——
  // 把 "/" 302 到带版本号的地址，强制一次缓存未命中，此后 no-cache 接管
  if (urlPath === '/') {
    const search = new URL(req.url, 'http://x').search || '';
    res.writeHead(302, { Location: '/index.html?v=3' + (search ? '&' + search.slice(1) : ''), 'Cache-Control': 'no-store' });
    res.end();
    return;
  }
  let file = path.normalize(path.join(root, urlPath));
  if (!file.startsWith(root)) { res.writeHead(403); res.end(); return; }
  if (file.endsWith(path.sep) || urlPath === '/') file = path.join(root, 'index.html');
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end('not found'); return; }
    const ext = path.extname(file).toLowerCase();
    res.writeHead(200, {
      'Content-Type': MIME[ext] || 'application/octet-stream',
      'Cache-Control': 'no-cache',
    });
    res.end(data);
  });
}).listen(port, () => {
  console.log(`G1 Boxing (MuJoCo WASM) running at http://localhost:${port}`);
});
