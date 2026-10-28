/* =========================================================
   혼공머신 3장 학습 노트 — interactions & charts
   외부 라이브러리 없이 SVG로 직접 그립니다.
   ========================================================= */
(() => {
  'use strict';

  const SVGNS = 'http://www.w3.org/2000/svg';
  const TRAIN = FINCH_TRAIN;
  const TEST = FINCH_TEST;
  const ALL = FINCH_ALL;

  /* ---------- tiny helpers ---------- */

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function el(tag, attrs = {}, parent) {
    const node = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v !== undefined && v !== null) node.setAttribute(k, String(v));
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function text(parent, x, y, str, cls, anchor = 'middle') {
    const t = el('text', { x, y, class: cls, 'text-anchor': anchor }, parent);
    t.textContent = str;
    return t;
  }

  const fmt = (v, d = 3) => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(d);

  function extentOf(pairs, idx) {
    let lo = Infinity, hi = -Infinity;
    for (const p of pairs) { lo = Math.min(lo, p[idx]); hi = Math.max(hi, p[idx]); }
    return [lo, hi];
  }

  function r2(pairs, predict) {
    const mean = pairs.reduce((s, p) => s + p[1], 0) / pairs.length;
    let ssRes = 0, ssTot = 0;
    for (const p of pairs) {
      const err = p[1] - predict(p[0]);
      ssRes += err * err;
      ssTot += (p[1] - mean) ** 2;
    }
    return 1 - ssRes / ssTot;
  }

  function mae(pairs, predict) {
    let s = 0;
    for (const p of pairs) s += Math.abs(p[1] - predict(p[0]));
    return s / pairs.length;
  }

  function knnPredict(x, k) {
    const near = TRAIN
      .map(p => ({ d: Math.abs(p[0] - x), y: p[1] }))
      .sort((a, b) => a.d - b.d)
      .slice(0, k);
    return near.reduce((s, p) => s + p.y, 0) / near.length;
  }

  function knnNeighbors(x, k) {
    return TRAIN
      .map((p, i) => ({ ...p, i, d: Math.abs(p[0] - x) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, k);
  }

  /* linear least squares: y = a x + b */
  function fitLine(pairs) {
    const n = pairs.length;
    const mx = pairs.reduce((s, p) => s + p[0], 0) / n;
    const my = pairs.reduce((s, p) => s + p[1], 0) / n;
    let num = 0, den = 0;
    for (const p of pairs) { num += (p[0] - mx) * (p[1] - my); den += (p[0] - mx) ** 2; }
    const a = num / den;
    return { a, b: my - a * mx, predict: x => a * x + (my - a * mx) };
  }

  /* quadratic least squares: y = c2 x^2 + c1 x + c0 */
  function fitQuad(pairs) {
    const S = [0, 0, 0, 0, 0]; // Σx^0..Σx^4
    let sy = 0, sxy = 0, sx2y = 0;
    for (const [x, y] of pairs) {
      const x2 = x * x;
      S[0] += 1; S[1] += x; S[2] += x2; S[3] += x2 * x; S[4] += x2 * x2;
      sy += y; sxy += x * y; sx2y += x2 * y;
    }
    const A = [[S[4], S[3], S[2], sx2y], [S[3], S[2], S[1], sxy], [S[2], S[1], S[0], sy]];
    // gaussian elimination with partial pivoting
    for (let i = 0; i < 3; i++) {
      let p = i;
      for (let r = i + 1; r < 3; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
      [A[i], A[p]] = [A[p], A[i]];
      const piv = A[i][i];
      for (let c = i; c < 4; c++) A[i][c] /= piv;
      for (let r = 0; r < 3; r++) {
        if (r === i) continue;
        const f = A[r][i];
        for (let c = i; c < 4; c++) A[r][c] -= f * A[i][c];
      }
    }
    const c2 = A[0][3], c1 = A[1][3], c0 = A[2][3];
    return { c2, c1, c0, predict: x => c2 * x * x + c1 * x + c0 };
  }

  const LINE = fitLine(TRAIN);
  const QUAD = fitQuad(TRAIN);

  /* ---------- plot frame ---------- */

  const M = { top: 18, right: 18, bottom: 42, left: 50 };

  function createPlot(mount, cfg) {
    const W = 760, H = cfg.height || 400;
    const pad = { ...M, ...(cfg.pad || {}) };
    const svg = el('svg', {
      viewBox: `0 0 ${W} ${H}`,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'presentation'
    });
    const iw = W - pad.left - pad.right;
    const ih = H - pad.top - pad.bottom;
    const [x0, x1] = cfg.xd;
    const [y0, y1] = cfg.yd;
    const X = v => pad.left + ((v - x0) / (x1 - x0)) * iw;
    const Y = v => pad.top + ih - ((v - y0) / (y1 - y0)) * ih;

    const grid = el('g', {}, svg);
    (cfg.yticks || []).forEach(t => {
      el('line', { x1: pad.left, x2: W - pad.right, y1: Y(t), y2: Y(t), class: 'grid-line' }, grid);
      text(grid, pad.left - 9, Y(t) + 4, cfg.yLabel ? cfg.yLabel(t) : t, 'axis-label', 'end');
    });
    (cfg.xticks || []).forEach(t => {
      el('line', { x1: X(t), x2: X(t), y1: pad.top, y2: pad.top + ih, class: 'grid-line' }, grid);
      text(grid, X(t), H - pad.bottom + 19, cfg.xLabel ? cfg.xLabel(t) : t, 'axis-label');
    });
    if (cfg.zero) el('line', { x1: pad.left, x2: W - pad.right, y1: Y(0), y2: Y(0), class: 'zero-line' }, grid);
    el('line', { x1: pad.left, x2: W - pad.right, y1: pad.top + ih, y2: pad.top + ih, class: 'axis-line' }, grid);
    el('line', { x1: pad.left, x2: pad.left, y1: pad.top, y2: pad.top + ih, class: 'axis-line' }, grid);
    text(grid, pad.left + iw / 2, H - 4, cfg.xTitle || '', 'axis-title');
    if (cfg.yTitle) {
      const t = el('text', {
        x: 4, y: pad.top + ih / 2, class: 'axis-title',
        transform: `rotate(-90 4 ${pad.top + ih / 2})`, 'text-anchor': 'middle'
      }, grid);
      t.textContent = cfg.yTitle;
    }

    const series = el('g', {}, svg);
    mount.appendChild(svg);
    return { svg, series, grid, X, Y, W, H, pad, iw, ih };
  }

  function dots(group, pairs, cls, r = 4.2) {
    pairs.forEach(([x, y]) => el('circle', { cx: group.X(x), cy: group.Y(y), r, class: cls }, group.series));
  }

  function smoothPath(group, pts) {
    // Catmull-Rom → cubic bezier
    if (pts.length < 2) return '';
    let d = `M ${group.X(pts[0][0])} ${group.Y(pts[0][1])}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i - 1] || pts[i];
      const p1 = pts[i];
      const p2 = pts[i + 1];
      const p3 = pts[i + 2] || p2;
      const c1x = p1[0] + (p2[0] - p0[0]) / 6;
      const c1y = p1[1] + (p2[1] - p0[1]) / 6;
      const c2x = p2[0] - (p3[0] - p1[0]) / 6;
      const c2y = p2[1] - (p3[1] - p1[1]) / 6;
      d += ` C ${group.X(c1x)} ${group.Y(c1y)}, ${group.X(c2x)} ${group.Y(c2y)}, ${group.X(p2[0])} ${group.Y(p2[1])}`;
    }
    return d;
  }

  /* =========================================================
     1) 데이터 미리보기
     ========================================================= */
  function drawDataPreview() {
    const mount = $('#chart-data');
    if (!mount) return;
    const g = createPlot(mount, {
      height: 330,
      xd: [7.8, 16], yd: [2.4, 7.3],
      xticks: [8, 10, 12, 14, 16], yticks: [3, 4, 5, 6, 7],
      xTitle: '부리 길이', yTitle: '부리 높이'
    });
    const small = ALL.filter(p => p[0] < 11);
    const large = ALL.filter(p => p[0] >= 11);
    dots(g, small, 'pt-train');
    large.forEach(([x, y]) => el('circle', { cx: g.X(x), cy: g.Y(y), r: 4.2, class: 'pt-test' }, g.series));

    text(g.series, g.X(8.05), g.Y(7.16), '작은 부리 35마리', 'callout', 'start')
      .setAttribute('fill', '#0f6b52');
    text(g.series, g.X(12.35), g.Y(3.95), '큰 부리 14마리', 'callout', 'start')
      .setAttribute('fill', '#b45309');
  }

  /* =========================================================
     2) KNN 회귀 인터랙티브
     ========================================================= */
  function drawKnn() {
    const mount = $('#chart-knn');
    if (!mount) return;

    const g = createPlot(mount, {
      height: 400,
      xd: [7.9, 16], yd: [2.5, 7.25],
      xticks: [8, 9, 10, 11, 12, 13, 14, 15, 16],
      yticks: [3, 4, 5, 6, 7],
      xTitle: '부리 길이', yTitle: '부리 높이'
    });

    // base layers
    TEST.forEach(([x, y]) => el('circle', { cx: g.X(x), cy: g.Y(y), r: 4.2, class: 'pt-test' }, g.series));
    dots(g, TRAIN, 'pt-train');

    const overlay = el('g', {}, g.series);
    const kIn = $('#kRange'), kOut = $('#kOut');
    const pIn = $('#probeRange'), pOut = $('#probeOut');
    const trainOut = $('#knnTrain'), testOut = $('#knnTest'), maeOut = $('#knnMae');
    const predNote = $('#knnPredNote');

    function render() {
      const k = Number(kIn.value);
      const probe = Number(pIn.value);
      kOut.textContent = k;
      pOut.textContent = probe.toFixed(1);
      overlay.textContent = '';

      // prediction curve
      const pts = [];
      for (let x = 8; x <= 15.6; x += 0.05) pts.push([x, knnPredict(x, k)]);
      el('path', { d: smoothPath(g, pts), class: 'curve curve--knn' }, overlay);

      // neighbors
      const nbs = knnNeighbors(probe, k);
      nbs.forEach(p => {
        el('rect', {
          x: g.X(p[0]) - 4.4, y: g.Y(p[1]) - 4.4, width: 8.8, height: 8.8,
          class: 'pt-nb', transform: `rotate(45 ${g.X(p[0])} ${g.Y(p[1])})`
        }, overlay);
      });

      // probe
      const pred = knnPredict(probe, k);
      const cx = g.X(probe), cy = g.Y(pred);
      el('path', {
        d: `M ${cx} ${cy - 8} L ${cx + 7} ${cy + 5} L ${cx - 7} ${cy + 5} Z`,
        class: 'pt-probe'
      }, overlay);

      // label box
      const label = `${pred.toFixed(2)}`;
      const bx = Math.min(cx + 12, g.W - g.pad.right - 52);
      const by = Math.max(cy - 26, 14);
      el('rect', { x: bx, y: by, width: 44, height: 21, rx: 5, class: 'callout-bg' }, overlay);
      text(overlay, bx + 22, by + 15, label, 'callout');

      // scores
      const r2Train = r2(TRAIN, x => knnPredict(x, k));
      const r2Test = r2(TEST, x => knnPredict(x, k));
      const m = mae(TEST, x => knnPredict(x, k));
      trainOut.textContent = fmt(r2Train, 4);
      testOut.textContent = fmt(r2Test, 4);
      maeOut.textContent = fmt(m, 4);
      predNote.textContent = `k = ${k}, 길이 ${probe.toFixed(1)} → 예측 높이 ${pred.toFixed(2)} · 이웃 ${k}마리의 평균`;
    }

    kIn.addEventListener('input', render);
    pIn.addEventListener('input', render);
    render();
  }

  /* =========================================================
     3) 모델 비교 (KNN / 선형 / 다항)
     ========================================================= */
  function drawModelCompare() {
    const mount = $('#chart-model');
    if (!mount) return;

    const g = createPlot(mount, {
      height: 430,
      xd: [7.5, 31], yd: [-33, 8.5],
      xticks: [8, 10, 12, 14, 16, 20, 25, 30],
      yticks: [-30, -20, -10, 0],
      zero: true,
      xTitle: '부리 길이', yTitle: '부리 높이'
    });

    // training range marker
    el('line', {
      x1: g.X(15.5), x2: g.X(15.5), y1: g.pad.top, y2: g.pad.top + g.ih,
      class: 'range-line'
    }, g.series);
    text(g.series, g.X(15.5) + 6, g.pad.top + 12, '훈련 범위 끝 15.5', 'range-label', 'start');

    dots(g, TRAIN, 'pt-train');

    // negative zone hint
    const negLabel = text(g.series, g.X(22), g.Y(-26), '높이가 0보다 작은 예측', 'callout', 'middle');
    negLabel.setAttribute('fill', '#9a948b');
    negLabel.setAttribute('font-size', '10.5');

    const overlay = el('g', {}, g.series);
    const model = {
      knn: {
        predict: x => knnPredict(x, 3),
        train: r2(TRAIN, x => knnPredict(x, 3)),
        test: r2(TEST, x => knnPredict(x, 3)),
        cls: 'curve--knn',
        note: 'KNN은 15.5 밖을 알지 못해 마지막 이웃들의 평균(2.8)에 머뭅니다.'
      },
      lin: {
        predict: LINE.predict,
        train: r2(TRAIN, LINE.predict),
        test: r2(TEST, LINE.predict),
        cls: 'curve--lin',
        note: '직선은 범위 밖까지 내려갑니다. 30.0에서 높이가 음수(-6.46)가 됩니다.'
      },
      poly: {
        predict: QUAD.predict,
        train: r2(TRAIN, QUAD.predict),
        test: r2(TEST, QUAD.predict),
        cls: 'curve--poly',
        note: '곡선도 30.0에서는 더 크게 음수(-30.94)가 됩니다. 데이터 범위 밖 외삽은 여전히 위험합니다.'
      }
    };

    function render(which) {
      const m = model[which];
      overlay.textContent = '';

      const pts = [];
      for (let x = 8; x <= 30.001; x += 0.1) pts.push([x, m.predict(x)]);
      el('path', { d: smoothPath(g, pts), class: `curve ${m.cls}` }, overlay);

      // probe at 30
      const px = g.X(30), py = g.Y(m.predict(30));
      el('line', { x1: px, x2: px, y1: py, y2: g.pad.top + g.ih, class: 'grid-line' }, overlay);
      el('circle', { cx: px, cy: py, r: 5, class: 'pt-probe' }, overlay);

      const val = m.predict(30);
      const bx = g.X(30) - 96, by = Math.min(py + 12, g.pad.top + g.ih - 26);
      el('rect', { x: bx, y: by, width: 74, height: 22, rx: 5, class: 'callout-bg' }, overlay);
      text(overlay, bx + 37, by + 15, val.toFixed(2), 'callout');

      $('#modelTrain').textContent = fmt(m.train, 4);
      $('#modelTest').textContent = fmt(m.test, 4);
      $('#modelPred').textContent = val.toFixed(2);
      $('#modelNote').textContent = m.note;

      $$('[data-model]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.model === which)));
    }

    $$('[data-model]').forEach(btn => {
      btn.addEventListener('click', () => render(btn.dataset.model));
    });
    render('knn');
  }

  /* =========================================================
     4) 과대/과소적합 개념도
     ========================================================= */
  const SYNTH = [
    [0.8, 2.1], [1.9, 3.4], [3.0, 3.0], [4.2, 5.2],
    [5.3, 4.4], [6.4, 6.6], [7.6, 6.0], [8.7, 8.4]
  ];

  function drawFit(kind, mountSel) {
    const mount = $(mountSel);
    if (!mount) return;
    const g = createPlot(mount, {
      height: 200,
      pad: { top: 10, right: 12, bottom: 10, left: 12 },
      xd: [0, 10], yd: [0, 10],
      xticks: [], yticks: []
    });
    el('circle', { cx: g.X(0.4), cy: g.Y(9.6), r: 2.6, class: 'pt-train' }, g.series);

    if (kind === 'under') {
      const line = fitLine(SYNTH);
      el('path', {
        d: `M ${g.X(0.2)} ${g.Y(line.predict(0.2))} L ${g.X(9.8)} ${g.Y(line.predict(9.8))}`,
        class: 'curve curve--lin', 'stroke-width': 2.4
      }, g.series);
    } else {
      const pts = SYNTH.map(([x, y], i) => [x, y + (i % 2 === 0 ? 0.55 : -0.5) * 0.9]);
      el('path', { d: smoothPath(g, pts), class: 'curve curve--poly', 'stroke-width': 2.4 }, g.series);
    }
    dots(g, SYNTH, 'pt-train', 3.6);
  }

  /* =========================================================
     5) alpha 개념도
     ========================================================= */
  function drawAlpha() {
    const mount = $('#chart-alpha');
    if (!mount) return;
    const g = createPlot(mount, {
      height: 330,
      pad: { top: 16, right: 20, bottom: 44, left: 54 },
      xd: [-3, 2], yd: [0.9, 1.005],
      xticks: [-3, -2, -1, 0, 1, 2],
      yticks: [0.92, 0.95, 0.98, 1],
      xLabel: t => `10^${t}`,
      xTitle: 'alpha (로그 스케일)', yTitle: 'R²'
    });

    const trainPts = [], testPts = [];
    for (let t = -3; t <= 2.001; t += 0.05) {
      const train = 0.9997 - 0.0023 * Math.pow(t + 3, 1.55);
      const d = t + 1;
      const test = 0.9828 - (d < 0 ? 0.0135 : 0.0105) * d * d;
      trainPts.push([t, Math.max(0.9, train)]);
      testPts.push([t, Math.max(0.9, test)]);
    }
    el('path', { d: smoothPath(g, trainPts), class: 'curve', stroke: '#0f6b52' }, g.series);
    el('path', { d: smoothPath(g, testPts), class: 'curve', stroke: '#b45309' }, g.series);

    el('line', {
      x1: g.X(-1), x2: g.X(-1), y1: g.pad.top + 6, y2: g.pad.top + g.ih,
      class: 'range-line'
    }, g.series);
    text(g.series, g.X(-1) + 6, g.pad.top + 12, 'α = 0.1', 'range-label', 'start');

    text(g.series, g.X(-2.75), g.pad.top + g.ih - 10, '과대적합', 'axis-label', 'start')
      .setAttribute('fill', '#9a948b');
    text(g.series, g.X(1.15), g.pad.top + g.ih - 10, '과소적합', 'axis-label', 'start')
      .setAttribute('fill', '#9a948b');
  }

  /* =========================================================
     UI: progress, TOC, reveal
     ========================================================= */
  function initScrollUI() {
    const bar = $('#progressBar');
    const sections = $$('main section[id]');
    const links = $$('.toc a');
    const onScroll = () => {
      const h = document.documentElement;
      const max = h.scrollHeight - h.clientHeight;
      const p = max > 0 ? (h.scrollTop / max) * 100 : 0;
      if (bar) bar.style.width = `${p}%`;

      let active = sections[0];
      for (const s of sections) {
        if (s.getBoundingClientRect().top <= 160) active = s;
      }
      links.forEach(a => a.classList.toggle('is-active', a.getAttribute('href') === `#${active.id}`));
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    onScroll();
  }

  function initReveal() {
    const items = $$('.reveal');
    if (!('IntersectionObserver' in window) || matchMedia('(prefers-reduced-motion: reduce)').matches) {
      items.forEach(i => i.classList.add('is-in'));
      return;
    }
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) { e.target.classList.add('is-in'); io.unobserve(e.target); }
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.04 });
    items.forEach(i => io.observe(i));
  }

  /* =========================================================
     UI: tabs, copy, quiz, lightbox
     ========================================================= */
  function initTabs() {
    const tabs = $$('.tablist button');
    tabs.forEach(btn => {
      btn.addEventListener('click', () => {
        tabs.forEach(b => b.setAttribute('aria-selected', String(b === btn)));
        $$('.tabpanel').forEach(p => { p.hidden = p.id !== btn.getAttribute('aria-controls'); });
      });
    });
  }

  function initCopy() {
    $$('.copy').forEach(btn => {
      btn.addEventListener('click', async () => {
        const pre = btn.closest('.code').querySelector('pre');
        const txt = pre.innerText;
        try {
          await navigator.clipboard.writeText(txt);
        } catch {
          const ta = document.createElement('textarea');
          ta.value = txt;
          ta.style.position = 'fixed';
          ta.style.opacity = '0';
          document.body.appendChild(ta);
          ta.select();
          document.execCommand('copy');
          ta.remove();
        }
        btn.textContent = 'copied';
        btn.classList.add('is-done');
        setTimeout(() => { btn.textContent = 'copy'; btn.classList.remove('is-done'); }, 1600);
      });
    });
  }

  function initQuiz() {
    const list = $('#quizList');
    if (!list) return;
    const scoreEl = $('#quizScore');
    const questions = $$('.q', list);
    let answered = 0, score = 0;

    questions.forEach(q => {
      const correct = Number(q.dataset.answer);
      $$('.opt', q).forEach((opt, idx) => {
        opt.addEventListener('click', () => {
          if (q.classList.contains('is-answered')) return;
          q.classList.add('is-answered');
          answered++;
          if (idx + 1 === correct) {
            opt.classList.add('is-right');
            score++;
          } else {
            opt.classList.add('is-wrong');
            $$('.opt', q)[correct - 1].classList.add('is-right');
          }
          $$('.opt', q).forEach(b => { b.disabled = true; });
          scoreEl.textContent = score;
        });
      });
    });

    $('#quizReset')?.addEventListener('click', () => {
      answered = 0; score = 0;
      scoreEl.textContent = '0';
      questions.forEach(q => {
        q.classList.remove('is-answered');
        $$('.opt', q).forEach(b => {
          b.disabled = false;
          b.classList.remove('is-right', 'is-wrong');
        });
      });
    });
  }

  function initLightbox() {
    const frame = $('#noteFrame');
    const box = $('#lightbox');
    if (!frame || !box) return;
    const closeBtn = $('#lightboxClose');
    const open = () => {
      box.classList.add('is-open');
      document.body.style.overflow = 'hidden';
      closeBtn.focus();
    };
    const close = () => {
      box.classList.remove('is-open');
      document.body.style.overflow = '';
      frame.focus();
    };
    frame.addEventListener('click', open);
    frame.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); }
    });
    closeBtn.addEventListener('click', close);
    box.addEventListener('click', e => { if (e.target === box) close(); });
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && box.classList.contains('is-open')) close(); });
  }

  /* ---------- boot ---------- */
  function boot() {
    drawDataPreview();
    drawKnn();
    drawModelCompare();
    drawFit('under', '#chart-under');
    drawFit('over', '#chart-over');
    drawAlpha();
    initScrollUI();
    initReveal();
    initTabs();
    initCopy();
    initQuiz();
    initLightbox();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
