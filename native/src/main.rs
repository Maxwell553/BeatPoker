//! Local JSON-lines interface to the exact-card postflop solver.
use postflop_solver::*;
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::time::Instant;

#[derive(Clone, Deserialize)]
struct Move { kind: String, #[serde(default)] amount: i32, card: Option<String> }

impl Move {
    fn action(&self) -> Result<Action, String> {
        Ok(match self.kind.as_str() {
            "fold" => Action::Fold, "check" => Action::Check, "call" => Action::Call,
            "bet" => Action::Bet(self.amount), "raise" => Action::Raise(self.amount),
            "allin" => Action::AllIn(self.amount),
            "chance" => Action::Chance(card_from_str(self.card.as_deref().ok_or("missing card")?)?),
            _ => return Err("unknown action".into()),
        })
    }
}

#[derive(Deserialize)]
struct Request {
    cmd: String,
    #[serde(default)] board: Vec<String>,
    ranges: Option<[String; 2]>,
    pot: Option<i32>, stack: Option<i32>,
    #[serde(default)] history: Vec<Move>,
    #[serde(default)] lines: Vec<Vec<Move>>,
    #[serde(default = "default_iters")] iterations: u32,
    #[serde(default = "default_seconds")] seconds: f64,
    #[serde(default = "default_target")] target: f32,
    #[serde(default = "default_memory")] memory_mb: u64,
    #[serde(default)] rake_rate: f64,
    #[serde(default)] rake_cap: f64,
    #[serde(default)] compact: bool,
    #[serde(default)] wide_root: bool,
    #[serde(default)] compression: Option<bool>,
    #[serde(default = "default_min_bet")] min_bet: i32,
    #[serde(default = "default_max_bets")] max_bets: usize,
    hole: Option<[String; 2]>,
}
fn default_iters() -> u32 { 1000 }
fn default_seconds() -> f64 { 30.0 }
fn default_target() -> f32 { 0.005 }
fn default_memory() -> u64 { 4096 }
fn default_min_bet() -> i32 { 1 }
fn default_max_bets() -> usize { 2 }

fn action_json(a: Action) -> Value {
    match a {
        Action::Fold => json!({"kind":"fold"}), Action::Check => json!({"kind":"check"}),
        Action::Call => json!({"kind":"call"}),
        Action::Bet(n) => json!({"kind":"bet", "amount":n}),
        Action::Raise(n) => json!({"kind":"raise", "amount":n}),
        Action::AllIn(n) => json!({"kind":"allin", "amount":n}),
        _ => json!({"kind":"chance"}),
    }
}

// Bound the ordinary raise abstraction while keeping explicit all-ins. Actual
// off-tree reraises are inserted after pruning and are never silently snapped.
fn prune_raises(tree: &mut ActionTree, bets: usize, required: &[Vec<Action>], min_bet: i32, max_bets: usize) -> Result<(), String> {
    if tree.is_terminal_node() { return Ok(()); }
    let bets = if tree.is_chance_node() { 0 } else { bets };
    for a in tree.available_actions().to_vec() {
        let mut prefix = tree.history().to_vec();
        prefix.push(a);
        let preserve = required.iter().any(|line| line.starts_with(&prefix));
        if (bets >= max_bets && matches!(a, Action::Raise(_)) && !preserve)
            || matches!(a, Action::Bet(n) if n < min_bet) {
            tree.remove_action(a)?;
            continue;
        }
        tree.play(a)?;
        let next = bets + usize::from(matches!(a, Action::Bet(_) | Action::Raise(_) | Action::AllIn(_)));
        prune_raises(tree, next, required, min_bet, max_bets)?;
        tree.undo()?;
    }
    Ok(())
}

fn query(game: &mut PostFlopGame, req: &Request) -> Result<Value, String> {
    game.back_to_root();
    for (i, mv) in req.history.iter().enumerate() {
        let a = mv.action()?;
        if let Action::Chance(card) = a {
            if !game.is_chance_node() || game.possible_cards() & (1u64 << card) == 0 {
                return Err(format!("invalid chance at {i}"));
            }
            game.play(card as usize);
        } else {
            let actions = game.available_actions();
            let index = actions.iter().position(|&x| x == a)
                .ok_or_else(|| format!("off_tree:{i}:{a:?}"))?;
            game.play(index);
        }
    }
    let ranges: Vec<Value> = (0..2).map(|p| {
        json!({"hands": holes_to_strings(game.private_cards(p)).unwrap(), "weights":game.weights(p)})
    }).collect();
    if game.is_terminal_node() || game.is_chance_node() {
        return Ok(json!({"terminal":game.is_terminal_node(), "chance":game.is_chance_node(), "ranges":ranges}));
    }
    let player = game.current_player();
    let hands = game.private_cards(player);
    let n = hands.len();
    let strategy = game.strategy();
    let actions: Vec<_> = game.available_actions().into_iter().map(action_json).collect();
    let mut out = json!({"player":player,"actions":actions,"ranges":ranges});
    if let Some(hole) = &req.hole {
        let a = card_from_str(&hole[0])?;
        let b = card_from_str(&hole[1])?;
        let index = hands.iter().position(|&(c,d)| (a==c && b==d) || (a==d && b==c))
            .ok_or("hero hand absent from public range")?;
        out["probabilities"] = json!((0..actions.len()).map(|i| strategy[i*n+index]).collect::<Vec<_>>());
    }
    Ok(out)
}

fn build(req: &Request) -> Result<(PostFlopGame, Value), String> {
    let begin = Instant::now();
    if !(3..=5).contains(&req.board.len()) { return Err("board must have 3–5 cards".into()); }
    if !req.seconds.is_finite() || req.seconds <= 0.0 || req.iterations == 0 {
        return Err("positive solve budget required".into());
    }
    let ranges = req.ranges.as_ref().ok_or("missing ranges")?;
    let card_config = CardConfig {
        range: [ranges[0].parse()?, ranges[1].parse()?],
        flop: flop_from_str(&req.board[..3].concat())?,
        turn: if req.board.len() >= 4 { card_from_str(&req.board[3])? } else { NOT_DEALT },
        river: if req.board.len() == 5 { card_from_str(&req.board[4])? } else { NOT_DEALT },
    };
    let wide_flop = !req.compact || (req.wide_root && req.board.len()==3);
    let wide_turn = !req.compact || (req.wide_root && req.board.len()==4);
    let wide_river = !req.compact || (req.wide_root && req.board.len()==5);
    let flop = BetSizeOptions::try_from((if wide_flop { "33%, 75%" } else { "66%" }, "3x"))?;
    let turn = BetSizeOptions::try_from((if wide_turn { "50%, 100%" } else { "75%" }, "3x"))?;
    let river = BetSizeOptions::try_from((if wide_river { "33%, 75%, 150%, e" } else { "75%, e" }, "2.5x"))?;
    let cfg = TreeConfig {
        initial_state: match req.board.len() { 3=>BoardState::Flop, 4=>BoardState::Turn, _=>BoardState::River },
        starting_pot:req.pot.ok_or("missing pot")?, effective_stack:req.stack.ok_or("missing stack")?,
        rake_rate:req.rake_rate, rake_cap:req.rake_cap,
        flop_bet_sizes:[flop.clone(),flop], turn_bet_sizes:[turn.clone(),turn],
        river_bet_sizes:[river.clone(),river], turn_donk_sizes:None, river_donk_sizes:None,
        add_allin_threshold:1.5, force_allin_threshold:0.15, merging_threshold:0.1,
    };
    let target = cfg.starting_pot as f32 * req.target;
    let mut tree = ActionTree::new(cfg)?;
    prune_raises(&mut tree, 0, &[], req.min_bet, req.max_bets)?;
    let mut required = Vec::new();
    for line in &req.lines {
        let mut prefix = Vec::new();
        tree.back_to_root();
        for mv in line {
            let a = mv.action()?;
            if matches!(a, Action::Chance(_)) { continue; }
            prefix.push(a);
            if !tree.available_actions().contains(&a) { tree.add_line(&prefix)?; }
            tree.play(a)?;
        }
        required.push(prefix);
    }
    tree.back_to_root();
    prune_raises(&mut tree, 0, &required, req.min_bet, req.max_bets)?;
    tree.back_to_root();
    let mut game = PostFlopGame::with_config(card_config, tree)?;
    let compression = req.compression.unwrap_or(true);
    let (uncompressed_bytes, compressed_bytes) = game.memory_usage();
    let bytes = if compression { compressed_bytes } else { uncompressed_bytes };
    if bytes > req.memory_mb * 1024 * 1024 { return Err(format!("memory_limit:{bytes}")); }
    game.allocate_memory(compression);
    let mut gap = compute_exploitability(&game);
    let setup_seconds = begin.elapsed().as_secs_f64();
    let search_begin = Instant::now();
    let mut completed = 0;
    for i in 0..req.iterations {
        // Tree allocation and the initial value pass must not consume the
        // search budget. Two steps let the average include an updated policy;
        // a caller explicitly requesting one step still receives one step.
        if completed >= req.iterations.min(2) && search_begin.elapsed().as_secs_f64() >= req.seconds { break; }
        solve_step(&game, i);
        completed = i+1;
        if completed % 10 == 0 {
            gap = compute_exploitability(&game);
            if gap <= target { break; }
        }
    }
    let search_seconds = search_begin.elapsed().as_secs_f64();
    let finalize_begin = Instant::now();
    if completed % 10 != 0 { gap = compute_exploitability(&game); }
    finalize(&mut game);
    let stats = json!({"iterations":completed, "exploitability_chips":gap,
        "target_chips":target, "converged":gap<=target, "seconds":begin.elapsed().as_secs_f64(),
        "setup_seconds":setup_seconds, "search_seconds":search_seconds,
        "finalize_seconds":finalize_begin.elapsed().as_secs_f64(),
        "budget_scope":"search iterations and periodic convergence checks; setup and final measurement excluded",
        "memory_bytes":bytes, "compact":req.compact, "wide_root":req.wide_root, "max_bets":req.max_bets,
        "compression":compression,
        "scope":"specified ranges and abstract betting tree"});
    Ok((game, stats))
}

fn main() {
    let threads = std::env::var("HU_SOLVER_THREADS").ok().and_then(|x| x.parse().ok()).unwrap_or(6);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    let mut game: Option<PostFlopGame> = None;
    let mut stats = json!(null);
    for line in io::stdin().lock().lines() {
        let answer = (|| -> Result<Value,String> {
            let line = line.map_err(|e|e.to_string())?;
            let req: Request = serde_json::from_str(&line).map_err(|e|e.to_string())?;
            if req.cmd == "clear" { game=None; return Ok(json!({"ok":true})); }
            if req.cmd == "solve" {
                game = None; // release old allocation before building the replacement
                let (g,s) = build(&req)?;
                game=Some(g); stats=s;
            } else if req.cmd != "query" { return Err("unknown command".into()); }
            let mut out = query(game.as_mut().ok_or("no solved game")?, &req)?;
            out["solve"] = stats.clone();
            Ok(out)
        })();
        let out = answer.unwrap_or_else(|e|json!({"error":e}));
        println!("{out}");
        io::stdout().flush().unwrap();
    }
}
