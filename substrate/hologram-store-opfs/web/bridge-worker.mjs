// Worker-side glue for the hologram OPFS sync/async bridge (architecture §9 G-C2 → B3).
//
// The main thread instantiates a `SyncOpfsBridge` and postMessages the SAB to this Worker.
// The Worker constructs a `BridgeWorker` over the SAB and blocks on `Atomics.waitAsync` until
// the main thread's `Atomics.notify` wakes it. Each request is dispatched via the existing
// async `opfs_*` surface; the response is written back to the SAB and `Atomics.notify`'d.
//
// This file is loaded as a Worker via `new Worker('bridge-worker.mjs', { type: 'module' })`.

import init, {
  BridgeWorker,
  bridge_state_offset,
  bridge_state_idle,
} from './pkg/hologram_store_opfs.js';

let worker = null;
let stateOffset = 0;
let stateIdle = 0;

self.onmessage = async (ev) => {
  const msg = ev.data;
  if (msg && msg.kind === 'init') {
    await init();
    worker = new BridgeWorker(msg.sab);
    stateOffset = bridge_state_offset();
    stateIdle = bridge_state_idle();
    self.postMessage({ kind: 'ready', capacity: worker.capacity() });
    loop();
  }
};

// The dispatch loop: block on Atomics.waitAsync until the main thread posts a request, then
// drive `serve_step()` (one request per wake). When the response is written, the main thread's
// own Atomics.wait wakes and reads it.
async function loop() {
  if (!worker) return;
  const i32 = worker.i32_view();
  while (true) {
    // Suspend the Worker thread asynchronously until the main thread changes STATE != IDLE.
    // `Atomics.waitAsync` returns { async: true, value: Promise } or { async: false, value: ... }
    const wait = Atomics.waitAsync(i32, stateOffset, stateIdle);
    if (wait.async) {
      const _ = await wait.value;
    }
    try {
      const op = await worker.serve_step();
      // BridgeWorker.NO_WORK is u32::MAX — spurious wake; loop again.
      if (op === 0xFFFFFFFF) continue;
    } catch (e) {
      console.error('bridge worker step failed', e);
    }
  }
}
