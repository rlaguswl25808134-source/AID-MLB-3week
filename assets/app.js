/* 혼공머신 3장 — 차트 두 개와 k 슬라이더 */
(() => {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const TRAIN = typeof FINCH_TRAIN !== 'undefined' ? FINCH_TRAIN : [];
  const TEST = typeof FINCH_TEST !== 'undefined' ? FINCH_TEST : [];
  const ALL = typeof FINCH_ALL !== 'undefined' ? FINCH_ALL : [];

  const $ = (s, r = document) => r.querySelector(s);

  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  function text(parent, x, y, str, cls, anchor = 'middle') {
    const t = el('text', { x, y, class: cls, 'text-anchor': anchor }, parent);
    t.textContent = str;
    return t;
  }

  /* ---------- 계산 ---------- */

  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;

  function knnNeighbors(x, k) {
    return TRAIN
      .map((p, i) => ({ x: p[0], y: p[1], i, d: Math.abs(p[0] - x) }))
      .sort((a, b) => a.d - b.d || a.i - b.i)
      .slice(0, k);
  }

  function knnPredict(x, k) {
    return mean(knnNeighbors(x, k).map(p => p.y));
  }

  function r2(pairs, predict) {
    const m = mean(pairs.map(p => p[1]));
    let res = 0, tot = 0;
    for (const [x, y] of pairs) {
      res += (y - predict(x)) ** 2;
      tot += (y - m) ** 2;
    }
    return 1 - res / tot;
  }

  const mae = (pairs, predict) => mean(pairs.map(([x, y]) => Math.abs(y - predict(x))));

  function fitLine(pairs) {
    const mx = mean(pairs.map(p => p[0]));
    const my = mean(pairs.map(p => p[1]));
    let num = 0, den = 0;
    for (const [x, y] of pairs) { num += (x - mx) * (y - my); den += (x - mx) ** 2; }
    const a = num / den;
    return x => a * x + (my - a * mx);
  }

  /* 3x3 정규방정식으로 y = c2 x^2 + c1 x + c0 */
  function fitQuad(pairs) {
    const S = [0, 0, 0, 0, 0], T = [0, 0, 0];
    for (const [x, y] of pairs) {
      const p = [1, x, x * x];
      for (let i = 0; i < 3; i++) {
        T[i] += p[i] * y;
        for (let j = 0; j < 3; j++) S[i + j] += p[i] * p[j];
      }
    }
    const A = [[S[0], S[1], S[2]], [S[1], S[2], S[3]], [S[2], S[3], S[4]]];
    const det = m => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
      - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
      + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    const D = det(A);
    const solve = col => det(A.map((row, r) => row.map((v, c) => (c === col ? T[r] : v)))) / D;
    const [c0, c1, c2] = [0, 1, 2].map(solve);
    return x => c2 * x * x + c1 * x + c0;
  }

  /* ---------- 차트 뼈대 ---------- */

  const PAD = { t: 14, r: 16, b: 34, l: 38 };

  function frame(mount, { xd, yd, xticks, yticks, xlabel, ylabel }) {
    mount.innerHTML = '';
    const W = Math.max(240, Math.round(mount.clientWidth || 600));
    const H = Math.round(Math.min(320, Math.max(210, W * 0.52)));
    const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'presentation' }, mount);

    const X = v => PAD.l + (v - xd[0]) / (xd[1] - xd[0]) * (W - PAD.l - PAD.r);
    const Y = v => H - PAD.b - (v - yd[0]) / (yd[1] - yd[0]) * (H - PAD.t - PAD.b);

    for (const t of yticks) {
      el('line', { x1: PAD.l, x2: W - PAD.r, y1: Y(t), y2: Y(t), class: 'grid' }, svg);
      text(svg, PAD.l - 8, Y(t) + 3.5, String(t), 'tick', 'end');
    }
    for (const t of xticks) text(svg, X(t), H - PAD.b + 16, String(t), 'tick');

    el('line', { x1: PAD.l, x2: W - PAD.r, y1: H - PAD.b, y2: H - PAD.b, class: 'axis' }, svg);
    el('line', { x1: PAD.l, x2: PAD.l, y1: PAD.t, y2: H - PAD.b, class: 'axis' }, svg);

    text(svg, (PAD.l + W - PAD.r) / 2, H - 3, xlabel, 'axtitle');
    const cy = (PAD.t + H - PAD.b) / 2;
    const yl = text(svg, 13, cy, ylabel, 'axtitle');
    yl.setAttribute('transform', `rotate(-90 13 ${cy})`);

    const series = el('g', {}, svg);
    return { X, Y, series, W, H };
  }

  const dots = (g, pairs, cls, r = 3.4) =>
    pairs.forEach(([x, y]) => el('circle', { cx: g.X(x), cy: g.Y(y), r, class: cls }, g.series));

  function curve(g, fn, from, to, cls) {
    let d = '';
    for (let x = from; x <= to + 1e-9; x += 0.05) {
      d += (d ? 'L' : 'M') + g.X(x).toFixed(1) + ' ' + g.Y(fn(x)).toFixed(1);
    }
    el('path', { d, class: cls }, g.series);
  }

  /* ---------- 1) KNN 회귀 ---------- */

  const PROBE = 10;

  function drawKnn(k) {
    const mount = $('#chart-knn');
    if (!mount) return;
    const g = frame(mount, {
      xd: [7.8, 16], yd: [2.4, 7.3],
      xticks: [8, 10, 12, 14, 16], yticks: [3, 4, 5, 6, 7],
      xlabel: '부리 길이', ylabel: '부리 높이'
    });

    dots(g, TRAIN, 'pt-a');
    dots(g, TEST, 'pt-b', 3.2);
    curve(g, x => knnPredict(x, k), 8, 15.6, 'curve-a');

    const pred = knnPredict(PROBE, k);
    el('line', {
      x1: g.X(PROBE), x2: g.X(PROBE),
      y1: g.H - PAD.b, y2: g.Y(pred), class: 'probe'
    }, g.series);
    el('circle', { cx: g.X(PROBE), cy: g.Y(pred), r: 4.6, class: 'mark' }, g.series);

    const train = r2(TRAIN, x => knnPredict(x, k));
    const test = r2(TEST, x => knnPredict(x, k));
    const err = mae(TEST, x => knnPredict(x, k));

    $('#kOut').textContent = k;
    $('#knnStats').textContent =
      `훈련 R² ${train.toFixed(4)} · 테스트 R² ${test.toFixed(4)} · MAE ${err.toFixed(4)} · 10.0 예측 ${pred.toFixed(2)}`;
  }

  /* ---------- 2) 세 모델 비교 ---------- */

  function drawCompare() {
    const mount = $('#chart-compare');
    if (!mount) return;
    const g = frame(mount, {
      xd: [7.8, 16], yd: [1, 7.3],
      xticks: [8, 10, 12, 14, 16], yticks: [2, 3, 4, 5, 6, 7],
      xlabel: '부리 길이', ylabel: '부리 높이'
    });

    dots(g, ALL, 'pt-ghost', 2.8);
    curve(g, x => knnPredict(x, 3), 8, 15.6, 'curve-a');
    curve(g, fitLine(TRAIN), 8, 16, 'curve-b');
    curve(g, fitQuad(TRAIN), 8, 16, 'curve-c');
  }

  /* ---------- 시작 ---------- */

  function init() {
    const range = $('#kRange');
    if (!range) return;

    const redraw = () => {
      drawKnn(Number(range.value));
      drawCompare();
    };

    redraw();
    range.addEventListener('input', () => drawKnn(Number(range.value)));

    let t;
    window.addEventListener('resize', () => {
      clearTimeout(t);
      t = setTimeout(redraw, 150);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
