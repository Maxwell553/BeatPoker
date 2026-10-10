const $ = (id) => document.getElementById(id);

let state = null;
let syncing = false;
let busy = false;

async function api(path, body) {
  const opts = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function cardEl(c, faceDown) {
  const d = document.createElement("div");
  if (faceDown || !c) {
    d.className = "card back";
    d.setAttribute("aria-label", "facedown");
    return d;
  }
  d.className = "card" + (c.red ? " red" : "");
  d.innerHTML = `<span class="rank">${c.rank}</span><span class="sym">${c.sym}</span>`;
  d.title = c.code;
  return d;
}

function renderCards(el, cards, facedownCount) {
  el.innerHTML = "";
  if (cards && cards.length) {
    cards.forEach((c) => el.appendChild(cardEl(c, false)));
    return;
  }
  for (let i = 0; i < (facedownCount || 0); i++) el.appendChild(cardEl(null, true));
}

function betText(p) {
  if (p.folded) return "folded";
  if (p.allin) return "all-in";
  if (p.bet_bb > 0) return p.bet_bb.toFixed(1) + " bb";
  return "";
}

function clamp(n, lo, hi) {
  return Math.max(lo, Math.min(hi, n));
}

function applySlidersFromState() {
  const bb = $("bbSlider");
  bb.min = state.min_to_bb;
  bb.max = Math.max(state.min_to_bb, state.max_to_bb);
  bb.step = "0.5";
  if (Number(bb.value) < Number(bb.min) || Number(bb.value) > Number(bb.max)) {
    bb.value = bb.min;
  }
  $("bbVal").textContent = Number(bb.value).toFixed(1);
  const increment = Number(bb.value) - state.hero_bet_bb - state.to_call_bb;
  const matchedPot = state.pot_bb + state.to_call_bb;
  const pct = matchedPot > 0 ? (increment / matchedPot) * 100 : 66;
  $("pctSlider").value = clamp(Math.round(pct), 25, 200);
  $("pctVal").textContent = $("pctSlider").value;
}

function sizeFromPotPct(pct) {
  const increment = ((state.pot_bb + state.to_call_bb) * pct) / 100;
  return clamp(state.hero_bet_bb + state.to_call_bb + increment, state.min_to_bb, state.max_to_bb);
}

function render(s) {
  if (state === null) {
    $("stackSlider").value = s.stack_bb;
    $("stackVal").textContent = s.stack_bb;
    $("rakeSel").value = s.rake_preset || (s.rake === "off" ? "none" : "casino");
  }
  state = s;
  $("handNo").textContent = s.hand_no;
  $("session").textContent = (s.session_bb >= 0 ? "+" : "") + s.session_bb.toFixed(1) + " bb";
  $("rakeLabel").textContent = s.rake;
  $("pot").textContent = s.pot_bb.toFixed(1);
  $("street").textContent = s.street.replace("_", " ");

  $("heroPos").textContent = s.you.pos;
  $("botPos").textContent = s.villain.pos;
  $("heroStack").textContent = s.you.stack_bb.toFixed(1);
  $("botStack").textContent = s.villain.stack_bb.toFixed(1);
  $("heroBet").textContent = betText(s.you);
  $("botBet").textContent = betText(s.villain);

  $("seatYou").classList.toggle("to-act", s.you.to_act);
  $("seatBot").classList.toggle("to-act", s.villain.to_act);
  $("seatYou").classList.toggle("folded", s.you.folded);
  $("seatBot").classList.toggle("folded", s.villain.folded);

  renderCards($("heroCards"), s.you.hole, 2);
  renderCards($("botCards"), s.villain.hole, s.villain.folded ? 0 : 2);
  renderCards($("board"), s.board, 0);

  const dlr = $("dealer");
  dlr.className = "dealer " + (s.button === 0 ? "hero" : "bot");

  const log = $("log");
  log.innerHTML = "";
  s.log.forEach((row) => {
    const li = document.createElement("li");
    li.className = row.who;
    li.textContent = row.text;
    log.appendChild(li);
  });
  log.scrollTop = log.scrollHeight;

  const res = $("result");
  if (s.result) {
    res.hidden = false;
    const delta = s.result.hero_delta_bb;
    const sign = delta >= 0 ? "+" : "";
    const hands = [s.result.hero_hand, s.result.bot_hand].filter(Boolean).join(" vs ");
    res.textContent = `${s.result.winners.join(" & ")}  ${sign}${delta.toFixed(1)} bb` + (hands ? " · " + hands : "");
  } else {
    res.hidden = true;
  }

  const over = s.finished;
  $("btnNext").hidden = !over;
  ["btnFold", "btnCheck", "btnCall", "btnMin", "btnPot", "btnBet", "btnAllin"].forEach((id) => {
    $(id).hidden = over;
  });

  $("btnFold").disabled = !s.can_fold;
  $("btnCheck").disabled = !(s.your_turn && s.can_check);
  $("btnCall").disabled = !(s.your_turn && s.to_call_bb > 0);
  $("btnCall").textContent = s.to_call_bb > 0 ? `Call ${s.to_call_bb.toFixed(1)}` : "Call";
  $("btnMin").disabled = !(s.can_bet || s.can_raise);
  $("btnPot").disabled = !(s.can_bet || s.can_raise);
  $("btnBet").disabled = !(s.can_bet || s.can_raise);
  $("btnBet").textContent = s.can_raise ? "Raise" : "Bet";
  $("btnAllin").disabled = !s.your_turn;

  $("bbSlider").disabled = !(s.can_bet || s.can_raise);
  $("pctSlider").disabled = !(s.can_bet || s.can_raise);
  if (s.can_bet || s.can_raise) applySlidersFromState();
}

async function requestAction(path, body) {
  if (busy) return;
  busy = true;
  $("thinking").hidden = false;
  document.querySelectorAll(".controls button, .controls input, .controls select").forEach(el => el.disabled = true);
  try {
    state = await api(path, body);
  } catch (e) {
    alert(e.message);
  } finally {
    busy = false;
    $("thinking").hidden = true;
    document.querySelectorAll(".controls button, .controls input, .controls select").forEach(el => el.disabled = false);
    if (state) render(state);
  }
}

function send(kind, extra) {
  return requestAction("/api/act", { action: kind, ...extra });
}

$("bbSlider").addEventListener("input", () => {
  if (!state) return;
  $("bbVal").textContent = Number($("bbSlider").value).toFixed(1);
  const increment = Number($("bbSlider").value) - state.hero_bet_bb - state.to_call_bb;
  const matchedPot = state.pot_bb + state.to_call_bb;
  const pct = matchedPot > 0 ? (increment / matchedPot) * 100 : 66;
  syncing = true;
  $("pctSlider").value = clamp(Math.round(pct), 25, 200);
  $("pctVal").textContent = $("pctSlider").value;
  syncing = false;
});

$("pctSlider").addEventListener("input", () => {
  if (!state || syncing) return;
  $("pctVal").textContent = $("pctSlider").value;
  const to = sizeFromPotPct(Number($("pctSlider").value));
  $("bbSlider").value = to;
  $("bbVal").textContent = Number(to).toFixed(1);
});

$("stackSlider").addEventListener("input", () => {
  $("stackVal").textContent = $("stackSlider").value;
});

$("btnFold").onclick = () => send("fold");
$("btnCheck").onclick = () => send("check");
$("btnCall").onclick = () => send("call");
$("btnAllin").onclick = () => send("allin");
$("btnMin").onclick = () => send(state.can_raise ? "raise" : "bet", { to_bb: state.min_to_bb });
$("btnPot").onclick = () => {
  const to = sizeFromPotPct(100);
  send(state.can_raise ? "raise" : "bet", { to_bb: to });
};
$("btnBet").onclick = () => send(state.can_raise ? "raise" : "bet", { to_bb: Number($("bbSlider").value) });
$("btnNext").onclick = () => requestAction("/api/next", {});
$("btnNew").onclick = () => requestAction("/api/new", {
  stack_bb: Number($("stackSlider").value), rake: $("rakeSel").value,
});

api("/api/state").then(render).catch((e) => {
  document.body.innerHTML = "<p style='padding:24px'>Could not reach the table: " + e.message + "</p>";
});
