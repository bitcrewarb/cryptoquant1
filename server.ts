import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

const venvUvicorn = path.resolve(process.cwd(), '.venv/bin/uvicorn');
const uvicornCmd = fs.existsSync(venvUvicorn) ? venvUvicorn : 'uvicorn';

const child = spawn(
  uvicornCmd,
  [
    'crypto_quant_lab.app.main:app',
    '--host',
    '0.0.0.0',
    '--port',
    '3000',
    '--proxy-headers',
    '--forwarded-allow-ips=*',
  ],
  {
    stdio: 'inherit',
    env: process.env,
  },
);

function shutdown(signal: NodeJS.Signals): void {
  if (!child.killed) {
    child.kill(signal);
  }
}

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

child.on('exit', (code) => {
  process.exit(code ?? 0);
});
