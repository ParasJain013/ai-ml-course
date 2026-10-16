# Part H. Reinforcement learning — detailed syllabus

Course weeks 32–33. Weekly detail for Part H of [`COURSE_OUTLINE.md`](COURSE_OUTLINE.md), which stays
the source of truth: the topic ids, Core/Depth labels, project P11 and the DQN paper below are taken
from it unchanged. Written 2026-10-16.

**Goal:** learn the RL toolkit from MDPs to PPO and see exactly how it is used to tune LLMs.

Part H is the last content part before the capstone, and most of it is Depth for an applied ML
engineer. Two things in it are Core for interviews and for LLM work: the vocabulary of MDPs, value
functions and Q-learning (H1.1–H1.3), and the full RLHF pipeline (H2.3), which closes the loop F3.2
opened. The order follows the outline: H1 learns values (first with the model of the world known, then
from experience, then with a neural network), and H2 learns the policy directly, builds up to PPO, and
then runs PPO on your own tiny GPT with a reward model and a KL penalty, so you can compare it with the
DPO you implemented in F3.2.

Supervised learning has a fixed dataset and a correct answer per example. RL has neither: the agent's
own actions decide which data it sees, and the only signal is a reward that may arrive many steps
later. Almost every difficulty in this part (exploration, unstable targets, high-variance gradients,
reward hacking) comes from those two facts. Each lesson names which one it is fighting.

## At a glance

| Week | Course wk | Theme | Topics | Hours | Project | GPU |
|---|---|---|---|---|---|---|
| H1 | 32 | Value-based RL | H1.1–H1.4 | 12.5 | P11 M1 | no |
| H2 | 33 | Policy-based RL | H2.1–H2.4 | 13 | **P11 due** | helpful (H2.3 `GPU_SCALE` only) |

About 25.5 hours in total. Each week's hours include about 1 hour for the weekly quiz and cheat sheet.
Topic hours cover reading the lesson and doing its lab and exercises.

**How the hours are counted.** *Reading* is working through the lessons and the paper, including the
worked examples by hand. *Coding* is the labs and the code exercises. *Project* is P11.

**If a week runs long,** follow the outline's catch-up plan: drop the Depth topics H1.4 (DQN), H2.2
(actor–critic and PPO) and H2.4 (reward hacking). Part H then fits in one week with H1.1, H1.3, H2.1
and H2.3, and P11 becomes tabular Q-learning plus REINFORCE on CartPole. Two consequences to know
before cutting: H2.3 runs PPO, so on the cut path its lesson opens with a one-hour "PPO in brief"
section (the clipped objective and a value baseline, without GAE derivations); and the DQN reading
guide's summary is still worth 15 minutes, because "replay buffer and target network" is a common
interview question.

**Libraries this part needs** (Step 2 pins the versions): `numpy`, `matplotlib`, `torch` (CPU build),
`gymnasium` with the `toy-text` and `classic-control` environments, and optionally `gymnasium[box2d]`
for LunarLander. `stable-baselines3` is used only to check your DQN and PPO against a reference
implementation, never as the solution. H2.3 reuses your F1.4 GPT and F3.1–F3.2 code; `trl` is
optional there and only for reading its PPO and reward-model code, because its RL trainer APIs have
changed repeatedly. Gymnasium's API (`reset` returning `(obs, info)`, `step` returning
`terminated` and `truncated` separately), Box2D's Windows install (it has needed SWIG and build tools
in the past) and the `trl` and `stable-baselines3` APIs all move: VERIFY (as of 2026-10).

**Environments** (no datasets to download; nothing goes in `data/` this part except saved result
tables): your own gridworld written in H1.2; `FrozenLake-v1` (4×4, slippery and not), `CliffWalking-v0`
and `Taxi-v3` from toy-text; `CartPole-v1` from classic-control; `LunarLander` (the version suffix
changes between Gymnasium releases: VERIFY) as the optional harder target for P11. Environment version
suffixes and their reward thresholds: VERIFY when the labs are generated.

**Hardware:** every lab runs on the laptop CPU at default settings. RL on small networks is limited by
stepping the environment in Python, not by matrix maths, so a GPU barely speeds up H1 and most of H2;
what a free Colab or Kaggle session gives you is a second machine to run more seeds and longer
training while the laptop does something else. The one place a GPU helps for real is the H2.3
`GPU_SCALE` run on a small pretrained language model. Intel integrated graphics are not used. How to
use the free GPUs is under "Running Part H on Colab or Kaggle" at the end.

**One rule for every lab in this part:** RL results vary a lot between seeds. Every learning curve is
the mean over at least 3 seeds (5 in `GPU_SCALE`) with a shaded band, and a claim that method X beats
method Y needs that band (A6.2). A single lucky seed is not a result.

---

## H1 · Value-based RL (course week 32)

**Learning objectives**

- I can write an RL problem as an MDP `(S, A, P, R, γ)`, defining each symbol, and model a small
  gridworld that way, including which states are terminal.
- I can compute the discounted return `G_t = r_{t+1} + γ r_{t+2} + γ² r_{t+3} + …` by hand for a short
  episode, and explain what `γ` trades off and why `γ < 1` keeps an infinite sum finite.
- I can define the state value `V^π(s)` and the action value `Q^π(s, a)`, write the Bellman expectation
  and Bellman optimality equations, and check by hand that a given `V` satisfies them for a 3-state
  example.
- I can implement policy evaluation, policy iteration and value iteration from scratch on a gridworld
  with slippery moves, show they reach the same optimal policy, and explain why value iteration
  converges (each sweep is a contraction by a factor of `γ`).
- I can explain the difference between planning (the transition model `P` is known) and learning from
  experience (it is not), and why model-free methods are needed for most real problems.
- I can implement tabular Q-learning with an epsilon-greedy policy and a decaying epsilon, show it
  recovers the value-iteration policy on FrozenLake, and explain why it is off-policy, using SARSA on
  CliffWalking as the on-policy contrast.
- I can explain the exploration–exploitation trade-off and show, with curves over seeds, what happens
  with too little and too much exploration.
- (Depth) I can explain why naive Q-learning with a neural network is unstable (correlated samples, a
  target that moves with every update), and how a replay buffer and a target network fix each problem.
- (Depth) I can implement DQN in PyTorch from scratch, solve CartPole with it, and match the learning
  curve of a reference implementation to within the seed band.

**Topics** (in teaching order)

| Id | Topic | Level | Hours |
|---|---|---|---|
| H1.1 | MDPs: states, actions, rewards, returns, value functions, the Bellman equations | Core | 2.5 |
| H1.2 | Dynamic programming: value iteration and policy iteration on a gridworld | Core | 2 |
| H1.3 | Q-learning: tabular, epsilon-greedy exploration | Core | 2.5 |
| H1.4 | DQN: function approximation, replay buffer, target network (+ the DQN paper) | Depth | 3 |

**Hours:** reading 3.5 · coding 6.5 · project 1.5 · quiz and cheat sheet 1 · **total 12.5**

**Labs**

- **H1.1:** a minimal environment interface in plain Python (`reset() -> state`,
  `step(action) -> (next_state, reward, done)`), then a 4×4 gridworld behind it with a goal, a pit and
  walls, and an optional "slip" probability that moves the agent sideways. Roll out a random policy,
  compute returns for several values of `γ`, and estimate `V^π` of the random policy by Monte Carlo
  averaging of returns (A5.4). Then solve the Bellman expectation equation for the same policy exactly
  as a linear system, `V = (I − γ P^π)⁻¹ R^π` with shapes `(|S|, |S|)` and `(|S|,)` (A2.3; use
  `np.linalg.solve`, not `inv`, per A3.4), and show the Monte Carlo estimate converging to it. Finally
  wrap the same gridworld in the Gymnasium API (`reset` returns `(obs, info)`; `step` returns
  `(obs, reward, terminated, truncated, info)`) and explain why `terminated` and `truncated` must be
  handled differently when bootstrapping later (a time limit is not a real end state). Bridge: an MDP
  is a state machine whose transitions are probabilistic and whose edges carry a reward; the analogy
  breaks because the agent does not know the transition table and has to discover it by acting.
- **H1.2:** policy evaluation by iterating the Bellman expectation update until the largest change is
  below `1e-8`, then policy iteration (evaluate, act greedily, repeat until the policy stops changing)
  and value iteration (apply the Bellman optimality update directly). Plot the value function as a
  heatmap and the policy as arrows after each sweep. Check that all three agree with the H1.1 linear
  solve and with each other. Run both algorithms on `FrozenLake-v1` using its transition table
  (`env.unwrapped.P`; VERIFY the attribute) and record the optimal success rate of the slippery lake,
  which is well below 100%: that number is the ceiling for H1.3. Experiment: vary `γ` and the slip
  probability and explain how the optimal policy changes (with high slip it pays to walk away from the
  pit). Bridge: value iteration is the same relaxation idea as Bellman–Ford shortest paths, which you
  may have met in routing; with deterministic moves and a reward of −1 per step it computes exactly
  shortest distances. The analogy breaks once moves are random: the agent optimizes expected return,
  not one path.
- **H1.3:** tabular Q-learning from scratch: a `Q` array of shape `(|S|, |A|)`, epsilon-greedy action
  selection with ties broken at random, the update
  `Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') − Q(s, a)]` with no bootstrap on `terminated`, and a
  linearly decaying epsilon. Train on non-slippery then slippery FrozenLake and on `Taxi-v3`, evaluate
  the greedy policy separately from training (exploration off), and compare its success rate with the
  H1.2 optimum. Then SARSA on `CliffWalking-v0`, with the classic result that Q-learning learns the
  short path along the cliff edge while SARSA learns the safer path, because SARSA's targets include
  its own exploratory actions. Experiments over 5 seeds: epsilon fixed at 0, 0.1 and 0.5, and a decaying
  schedule; and the learning rate `α`. CPU default: a few thousand episodes, seconds per run.
- **H1.4:** DQN in PyTorch on `CartPole-v1`, built in three steps so each fix is visible. (1) Naive
  online Q-learning with a small MLP (`(4,) → 128 → 128 → (2,)`), updating on every transition: show it
  learns then collapses. (2) Add a replay buffer (a fixed-size ring buffer of
  `(s, a, r, s', terminated)` tuples sampled uniformly in mini-batches). (3) Add a target network
  copied from the online network every N steps, and use the Huber loss and gradient clipping (E2.1).
  Plot the return per episode and the mean predicted Q-value for each variant. Then implement Double
  DQN (one line: choose `a'` with the online network, score it with the target network) and show it
  reduces Q-value overestimation. Check against `stable-baselines3`'s DQN with matched settings
  (VERIFY its argument names). CPU default: about 30,000 environment steps, 3 seeds, a few minutes.
  `GPU_SCALE`: 100,000 steps, 5 seeds. Bridge: the replay buffer is a ring buffer you sample from,
  and the target network is a periodically refreshed snapshot, a little like reading from a replica
  that lags the primary on purpose; the analogy breaks because here the lag is the point, it keeps the
  regression target still long enough to fit.

**Paper:** *Human-level control through deep reinforcement learning* ("DQN", Mnih et al., 2015), with
its reading guide in `papers/`. Read the abstract, the methods section on experience replay and the
target network, and the main results figure. The reading guide explains why the Atari setup
(frame stacking, reward clipping, frame skipping) exists, which you do not need for CartPole, and notes
that the Nature version may be behind a paywall, while the earlier workshop paper "Playing Atari with
Deep Reinforcement Learning" (Mnih et al., 2013) covers the same core idea on arXiv. About 1 hour,
counted in H1.4.

**Project milestone: P11 M1, tabular agent and evaluation harness (1.5 h).** Write the harness P11 is
judged with: a function that trains an agent for a given seed, a separate greedy evaluation over 100
episodes, and a plot of the mean and spread across at least 5 seeds. Use it for your H1.3 Q-learning
agent on slippery FrozenLake (or Taxi) and report its success rate next to the H1.2 optimum. Pick the
deep-RL target for M2: DQN or PPO, on CartPole or LunarLander, and write down the reward threshold you
will call "solved" (VERIFY the environment's own threshold).

**Prerequisites:** A5.1 (expectation, the definition of a value function), A5.2 (categorical
distributions for stochastic transitions and policies), A5.4 (Monte Carlo estimation), A2.3 and A3.4
(the linear solve for policy evaluation, without `inv()`), A6.2 (confidence bands across seeds), A4.4
(the step-size idea behind `α`); B1.1 (baselines, and keeping evaluation separate from training);
C2.1–C2.3 (`nn.Module`, training loop, seeds and device handling), C3.1 (Adam), C4.3 (debugging
training runs, watching for collapse); E2.1 (gradient clipping). Within the week: H1.2 needs H1.1;
H1.3 needs H1.1 and uses H1.2's optimum as its reference; H1.4 needs H1.3.

**Resources** (optional)

- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd edition, 2018), free from the
  authors; chapters 3, 4 and 6 match H1.1–H1.3: http://incompleteideas.net/book/the-book-2nd.html
  (VERIFY link)
- David Silver, UCL Course on RL, lectures 1–6 (MDPs, dynamic programming, model-free control, value
  function approximation): https://www.davidsilver.uk/teaching/ (VERIFY link)
- Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013), the free precursor of the DQN
  paper: https://arxiv.org/abs/1312.5602 (VERIFY link)

**GPU:** no. Tabular methods take seconds and DQN on CartPole with a small MLP takes a few minutes on
the laptop CPU; a GPU would sit idle waiting for the environment. If you want the 5-seed `GPU_SCALE`
DQN run without tying up the laptop, run it in a Kaggle or Colab CPU session.

**Used later in:** the MDP vocabulary, returns and the advantage `Q − V` in every H2 topic (H1.1);
the value baseline and critic in H2.1–H2.2 (H1.1–H1.3); the replay-free, on-policy contrast that makes
PPO's design make sense in H2.2 (H1.4); the RLHF framing of a generated response as an episode in H2.3
(H1.1); implementing value iteration or Q-learning under time pressure in L3.1; P11.

---

## H2 · Policy-based RL (course week 33)

**Learning objectives**

- I can explain when learning a policy directly beats learning values: continuous or large action
  spaces, stochastic policies, and (for LLMs) an action space that is the whole vocabulary.
- I can derive the policy-gradient estimator `∇_θ J = E[Σ_t ∇_θ log π_θ(a_t | s_t) G_t]` with the
  log-derivative trick in a few lines, defining every symbol, and check it numerically on a two-armed
  bandit against a finite-difference gradient (A4.1).
- I can implement REINFORCE from scratch on CartPole, add a baseline, and show with curves over seeds
  that subtracting a baseline lowers the variance without changing the expected gradient.
- (Depth) I can implement an actor–critic with a value head and compute advantages with generalized
  advantage estimation (GAE), and explain what `λ` trades off between bias and variance.
- (Depth) I can write PPO's clipped surrogate objective
  `L = E[min(ρ_t A_t, clip(ρ_t, 1 − ε, 1 + ε) A_t)]` with `ρ_t = π_θ(a_t|s_t) / π_old(a_t|s_t)`, say what
  the clip prevents, and implement PPO with vectorized environments, minibatch epochs, an entropy bonus
  and a value loss, logging approximate KL and the clip fraction.
- I can draw the full RLHF pipeline (SFT model, preference data, reward model, PPO policy with a value
  head, frozen reference model) and say what each part is for and what it costs in memory.
- I can train a Bradley–Terry reward model from preference pairs, then run PPO on a tiny language model
  where the episode is one generated response, the reward is the reward-model score minus
  `β · KL(π_θ ‖ π_ref)`, and report reward and KL together.
- I can explain why DPO (F3.2) needs no reward model, no sampling during training and no value head,
  compare my PPO and DPO results at a similar KL from the reference, and say when each is used.
- (Depth) I can define reward hacking and specification gaming, show both in a lab, and name the
  mitigations used in practice (KL penalties, reward-model ensembles and refreshes, held-out evals,
  human review) and what each one cannot catch.

**Topics** (in teaching order)

| Id | Topic | Level | Hours |
|---|---|---|---|
| H2.1 | Policy gradients: REINFORCE and baselines | Core | 2.5 |
| H2.2 | Actor–critic and PPO: advantage estimation and the clipped objective | Depth | 3 |
| H2.3 | RLHF in full: reward model, PPO with a KL penalty, and why DPO (F3.2) removes the RL loop | Core | 2.5 |
| H2.4 | `[RAI]` Reward hacking and specification gaming | Depth | 1.5 |

**Hours:** reading 3.5 · coding 6 · project 2.5 · quiz and cheat sheet 1 · **total 13**

**Labs**

- **H2.1:** start with a two-armed bandit and a softmax policy over two logits: compute the
  policy-gradient estimate from samples and check it against a finite-difference gradient of the exact
  expected reward. Then REINFORCE on `CartPole-v1` in PyTorch: an MLP policy that outputs logits,
  `torch.distributions.Categorical` for sampling and log-probabilities (C1.4 log-softmax underneath),
  one episode per update, returns computed backwards with `γ`. Show three variants over 5 seeds: raw
  returns, reward-to-go (only rewards after step `t` count), and reward-to-go minus a baseline (first
  the batch mean, then a learned value network, H1.1). Plot the learning curves and the variance of
  the gradient estimate for each. Note the most common bug: forgetting the minus sign, so the
  optimizer minimizes reward. CPU default: 500 episodes, about a minute per seed.
- **H2.2:** actor–critic, then PPO, on `CartPole-v1` (and optionally LunarLander). A shared or split
  MLP with a policy head and a value head; GAE from scratch on a rollout buffer, with a hand-checked
  example of 4 steps and small numbers; advantage normalization. Then PPO: collect a rollout from
  several environments with Gymnasium's vector API (VERIFY its current interface), run several epochs
  of minibatch updates on the clipped objective plus a value loss and an entropy bonus, and log the
  approximate KL between old and new policy and the fraction of clipped ratios. Experiments over 3
  seeds: `ε` of 0.1, 0.2 and 0.3, and no clipping at all, where you should see the policy take a step
  too large and collapse. Check against `stable-baselines3`'s PPO with matched settings (VERIFY its
  defaults). CPU default: about 50,000 environment steps with 4 environments, a few minutes.
  `GPU_SCALE`: LunarLander, 500,000 steps, 5 seeds (on a CPU session too; see GPU below). The lesson
  ends with "the PPO implementation details that matter" (advantage normalization, value clipping,
  learning-rate annealing, orthogonal initialization), each switched off once to see its effect.
- **H2.3:** the full RLHF pipeline on your own models, small enough to run on the CPU, so you can see
  every number. (1) Start from your F3.1 SFT tiny GPT and the synthetic preference task from F3.2,
  where you know which answer is correct. (2) Reward model: the same GPT backbone with a scalar head on
  the last token, trained with the Bradley–Terry loss `−log σ(r_chosen − r_rejected)` on the F3.2 pairs;
  report its accuracy on held-out pairs. (3) PPO: sample a response per prompt from the policy, score
  it with the reward model, subtract a per-token KL penalty `β (log π_θ − log π_ref)` against a frozen
  copy of the SFT model, compute advantages with GAE over the response tokens using a value head, and
  update with the H2.2 clipped objective. Log mean reward, KL from the reference, response length and
  held-out accuracy on the true task. (4) Compare with your F3.2 DPO run on the same pairs: true-task
  accuracy against KL from the reference for both methods, and the count of models held in memory
  during training (policy, reference, reward model and value head for PPO; policy and reference for
  DPO). (5) Set `β = 0` and watch the reward climb while true-task accuracy stalls or falls, which is
  the bridge to H2.4. Optionally, read how `trl` structures the same loop and compare it with yours
  (VERIFY which PPO-style trainers it currently ships). CPU default: about 1–2M parameter models, 64
  prompts, responses up to 32 tokens, a few dozen PPO iterations, under 5 minutes. `GPU_SCALE`: the
  same pipeline on the F3.4 small pretrained model starting from your F3.1 SFT checkpoint, with a few
  hundred public preference pairs.
- **H2.4:** reward hacking in two places. First a gridworld whose designer meant "reach the goal" but
  rewarded "touch the coin" with a coin that respawns: Q-learning (H1.3) learns to circle the coin
  forever. Fix the reward and show the intended behaviour returns, then show a second, subtler
  misspecification (a step penalty so harsh that walking into the pit early is optimal). Second, return
  to the `β = 0` run from H2.3 and find what the policy exploited in the reward model (for example
  longer answers, a repeated token, or a pattern the reward model over-rewards); measure how the gap
  between reward-model score and true-task accuracy grows during training, which is Goodhart's law in
  one plot. Try two mitigations, the KL penalty and an ensemble of two reward models trained on
  different splits, and report how much each closes the gap. Write a half-page "known failure modes"
  section for the H2.3 model in the model-card style used since B4.6.

**Project milestone: P11 due, RL agent in Gymnasium (2.5 h this week, plus 1.5 h in H1).** From the
outline: tabular Q-learning on a small environment (your M1 result), then DQN or PPO (your H1.4 or
H2.2 code, not `stable-baselines3`) on CartPole or LunarLander, with learning curves across seeds.
Deliver: the M1 harness extended to the deep agent; curves over at least 5 seeds with a shaded band;
greedy evaluation over 100 episodes against your stated "solved" threshold; one ablation that matters
for your method (replay buffer or target network for DQN, clipping or GAE `λ` for PPO); a comparison
with `stable-baselines3` under matched settings; and a short README covering what failed first and how
you debugged it (C4.3), and the honest limits of the result. The brief is in `projects/` and the rubric
in `answers/projects/`.

**Prerequisites:** H1.1–H1.3 (returns, value functions, the advantage, the evaluation harness), H1.4
(for P11 if you choose DQN); A4.1 (finite-difference gradient checks), A5.1 (expectation, the
log-derivative trick), A5.4 (Monte Carlo gradient estimates), A6.2 (seed bands), A6.4 (KL divergence,
for PPO's approximate KL and the RLHF penalty); C1.4 (log-softmax), C2.1–C2.2 (training loop),
C3.1 (Adam), C3.3 (initialization), C4.3 (debugging training runs); F1.4 (your tiny GPT), F2.4
(sampling with temperature, used to generate responses), F3.1 (SFT, the starting policy and the response
masking), F3.2 (the RLHF overview, the Bradley–Terry model and DPO, compared here); B4.6 (model cards).
Within the week: H2.2 needs H2.1; H2.3 needs H2.2 (or the "PPO in brief" section on the cut path);
H2.4 uses H1.3 and the H2.3 run.

**Resources** (optional)

- OpenAI, *Spinning Up in Deep RL*, especially "Intro to Policy Optimization" and the PPO page:
  https://spinningup.openai.com/ (VERIFY link)
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017): https://arxiv.org/abs/1707.06347
  (VERIFY link)
- Huang et al., "The N Implementation Details of RLHF with PPO" (Hugging Face blog, 2023), a
  line-by-line account of the H2.3 pipeline at real scale: https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo
  (VERIFY link)

**GPU:** helpful for the H2.3 `GPU_SCALE` run only. REINFORCE and PPO with small MLPs are bound by the
environment and run as fast on the laptop CPU as on a free GPU; use a free session for its time, not
its GPU, when you want 5 seeds of LunarLander. H2.3 at its CPU default uses your 1–2M parameter
models. The `GPU_SCALE` version holds four copies of a small pretrained model's weights (policy,
reference, reward model, value head), which is where a free T4 helps: load in fp16 (the T4 has no bf16
support; VERIFY the GPU you are given), keep the reference and reward model frozen and in `eval` mode,
and cut response length and batch size before anything else if memory runs out.

**Used later in:** the RLHF and DPO explanation behind every "how was this model trained" question,
and choosing between PPO-style RLHF and DPO in L3.2 (H2.3); reward hacking as one more failure mode in
capstone model cards (L2.1) and in eval design, where any automated metric you optimize against, such
as an LLM-as-judge score (I2.4), can be gamed the same way (H2.4); implementing REINFORCE under time
pressure in L3.1 (H2.1); P11 as a portfolio piece in L3.3.

---

## Running Part H on Colab or Kaggle

Free-tier GPUs, quotas and session limits change often: VERIFY (as of 2026-10).

- **Why a free session at all:** for H1, H2.1 and H2.2 the bottleneck is Python stepping the
  environment, so pick a CPU runtime and use it to run 5 seeds in parallel processes (or 5 notebooks)
  while the laptop stays free. Save each seed's results as a small CSV and plot them together at home.
- **Colab:** open the lab, then Runtime → Change runtime type → CPU for the RL runs, or a GPU (usually
  a T4) for the H2.3 `GPU_SCALE` run. Install the environments with `pip install "gymnasium[box2d]"`
  if you use LunarLander (Box2D installs more easily on Colab's Linux than on Windows). Save results to
  Google Drive as you go, because the VM is wiped when the session ends.
- **Kaggle:** create a notebook, Settings → Accelerator → none or GPU, and turn Internet on to install
  packages and download the small pretrained model (Kaggle asks for phone verification first). Save
  outputs to `/kaggle/working`. Kaggle's longer sessions suit the 500,000-step LunarLander PPO runs.
- **Both:** every lab picks `device = "cuda" if torch.cuda.is_available() else "cpu"` (C2.3) and sets
  the seed of NumPy, PyTorch and the environment (`env.reset(seed=...)`), so the same file runs in all
  three places and each seed is reproducible. Bring back only results (CSV of returns, plots, eval
  tables), never model weights or checkpoints: they are not committed.

---

## You're ready for Part L when…

Part L (Capstone and interview prep, course weeks 34–36) comes next. It needs no RL, but interviewers
ask about RL fundamentals and RLHF, and L3.1 may ask you to code one of these algorithms. Answer each
item without notes; the answers belong in your quiz reviews, not here.

1. You can define an MDP and every symbol in it, compute a discounted return by hand, and write the
   Bellman optimality equation for `Q*`.
2. You can implement value iteration on a small gridworld from a blank file in 20 minutes, and explain
   why it needs the transition model and Q-learning does not.
3. You can write the tabular Q-learning update, explain each term and why it is off-policy, and say
   what goes wrong with too little exploration.
4. You can explain the two problems a replay buffer and a target network solve in DQN.
5. You can derive the REINFORCE gradient in a few lines and explain why a baseline reduces variance
   without adding bias.
6. You can write PPO's clipped objective and say in one sentence what the clip prevents.
7. You can draw the RLHF pipeline, name the four models PPO-based RLHF keeps in memory, and explain why
   the KL penalty to the reference model is there.
8. You can explain the practical differences between PPO-style RLHF and DPO (data, sampling, memory,
   stability), and say when a team would pick each.
9. You can give one example of reward hacking from your own labs and one mitigation, and explain why an
   automated eval score you optimize against can be gamed the same way.
10. P11 is at 70% or more on its rubric, with curves over at least 5 seeds and an evaluation kept
    separate from training.

If two or more items fail, revisit the topic ids your quiz reviews flagged before L1, but do not let
Part H delay the capstone: on the fast path, items 1, 3, 5, 7 and 8 are the ones to fix first.
