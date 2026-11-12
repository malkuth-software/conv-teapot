// Test harness: compares the GPU canvas with CPU reference pictures served beside the page
// (gpu/references.sh makes them). Load it in the page and call harness([[name, setup], ...]).
window.harness = async (cases) => {
  const g = window.teapot.gpu, gl = g.gl, e = window.teapot.engine, c = document.getElementById('view');
  c.width = 960; c.height = 600; e._tp_init(960, 600);
  const load = (src) => new Promise((res, rej) => { const i = new Image(); i.onload = () => { const k = document.createElement('canvas'); k.width = 960; k.height = 600; const x = k.getContext('2d'); x.drawImage(i, 0, 0); res(x.getImageData(0, 0, 960, 600).data); }; i.onerror = rej; i.src = src; });
  const out = [];
  for (const [name, setup] of cases) {
    Object.assign(g.options, { material: 0, gloss: 0, texture: 0, wire: 0, roomLight: 0, polished: 0 });
    e._tp_set(36, 0); e._tp_set(24, 1); e._tp_action(4);
    setup(g, e);
    const ref = await load('ref_' + name + '.png?' + Math.random());
    g.upload(); g.draw(window.teapot.state()); g.draw(window.teapot.state());
    const px = new Uint8Array(960 * 600 * 4);
    gl.readPixels(0, 0, 960, 600, gl.RGBA, gl.UNSIGNED_BYTE, px);
    let d8 = 0, d32 = 0, sum = 0;
    for (let y = 0; y < 600; y++) for (let x = 0; x < 960; x++) {
      const a = (y * 960 + x) * 4, b = ((599 - y) * 960 + x) * 4;
      const d = Math.max(Math.abs(ref[a] - px[b]), Math.abs(ref[a + 1] - px[b + 1]), Math.abs(ref[a + 2] - px[b + 2]));
      if (d > 8) d8++; if (d > 32) d32++; sum += d;
    }
    out.push({ name, over8: d8, over32: d32, mean: +(sum / 576000).toFixed(2) });
  }
  return out;
};
