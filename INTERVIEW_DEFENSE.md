# PA-GRPO: Technical Design Tradeoffs & Systems × RL Deep Dive

This document addresses key **ML Systems, Hardware Roofline, and Reinforcement Learning Theory** questions regarding **Precision-Asymmetric & Self-Speculative Rollout GRPO (`PA-GRPO`)**, linking empirical observations from `pa_grpo_experiment.py` to distributed frontier RL post-training.

---

## Q1. Roofline Physics & Arithmetic Intensity Crossover

> **Question:** *Why are autoregressive GRPO rollouts memory-bandwidth bound while the policy gradient update is compute bound? What is the exact arithmetic intensity (FLOPs/byte) of each phase, and at what batch size $M = P \times G$ does this assumption shift?*

### Answer
Consider a linear projection layer of shape $[d_{\text{in}}, d_{\text{out}}]$ stored in `BF16`/`FP16` ($2$ bytes/param):

1. **Autoregressive Rollout Decoding ($T = 1$, batch size $M = P \times G$):**
   - **FLOPs per step:** $2 M d_{\text{in}} d_{\text{out}}$
   - **Bytes transferred per step:** $2(d_{\text{in}} d_{\text{out}} + M d_{\text{in}} + M d_{\text{out}})$
   - **Arithmetic Intensity (Weight Streaming):**
     $$\text{AI}_{\text{rollout}} = \frac{M d_{\text{in}} d_{\text{out}}}{d_{\text{in}} d_{\text{out}} + M(d_{\text{in}} + d_{\text{out}})} \approx M \text{ FLOPs/byte} \quad (\text{when } M \ll d)$$
   - Because modern accelerators (Apple M-series Metal, NVIDIA H100/B200) have a Roofline ridge point of $150\text{--}300$ FLOPs/byte, rollout decoding at $M \in [8, 128]$ is deeply **memory-bandwidth bound**, streaming the entire model weights from memory $T_{\text{gen}}$ sequential times.

2. **Parallel Policy Gradient Update ($T_{\text{total}} = T_{\text{prompt}} + T_{\text{gen}}$, flattened token batch $N_{\text{tok}} = M \times T_{\text{total}}$):**
   - All completed rollout tokens are flattened into a single parallel GEMM of batch dimension $N_{\text{tok}} = M \times T_{\text{total}} \gg 256$.
   - **Arithmetic Intensity:** $\text{AI}_{\text{learner}} \approx \min(M \cdot T_{\text{total}},\, d/2)$ FLOPs/byte, placing the forward-backward learner pass firmly in the **compute-bound** Roofline regime while loading model weights only once.

3. **The Large-Batch Crossover ($M \ge 512$):**
   - When rollout batch size $M$ exceeds the hardware ridge point ($M \ge 256\text{--}512$) on long Chain-of-Thought sequences ($S \ge 4096\text{--}32768$), linear weight streaming becomes compute-bound, **but self-attention over the KV cache remains strictly memory-bandwidth bound** ($\mathcal{O}(1)$ FLOP/byte, reading $2 \cdot M \cdot S \cdot d_{\text{kv}}$ bytes from HBM at every token step).
   - **Why PA-GRPO accelerates both regimes:** At small-to-medium $M$, skipping upper refinement layers (`4 -> 8L`) cuts weight-streaming memory traffic by **33.4%**. At massive $M$ and long context $S$, PA-GRPO's **Cross-Layer KV Anchor (50% fewer KV layers cached) + INT4 KV quantization ($4\times$ bitwidth reduction)** slashes KV-cache HBM traffic and capacity by **$87.5\%$ ($8\times$)**, allowing $8\times$ larger rollout batch sizes before hitting HBM capacity limits.

---

## Q2. Process-Verified State Resets vs. Free-Trajectory Rollouts

> **Question:** *In `pa_grpo_experiment.py` (line 334), evaluation (`greedy=True`) runs 100% free autoregressive generation (`prev_tok = generated[-1]`), whereas training rollouts advance the KV cache along `verified_targets` after sampling each action $a_t \sim \pi_{\text{rollout}}(\cdot|s_t)$. Why did you design the training loop this way, and how does PA-GRPO extend to open-ended CoT where intermediate ground-truth prefixes aren't available?*

### Answer
1. **Why We Isolated Single-Step Off-Policy Drift in the 24-Token Multi-Hop Benchmark:**
   - Our benchmark task (`a,b,c> -> [s1:X,Y,Z]>[r2:Z,Y,X]=Z.`) contains an explicit **multi-hop KV-retrieval circuit**: Step 2 (`[r2:Z,Y,X]`) must attend back to the KV cache of Step 1 (`[s1:X,Y,Z]`) and copy those tokens in reverse order.
   - In our ablation without prefix resets, we uncovered a subtle RL failure mode (**Prefix-Induced Circuit Un-training**): if a rollout stochastically samples a wrong digit $X' \neq X$ at Step 1 (`index 4`), then the KV cache at `index 4` literally stores $X'$. If the outcome verifier at Step 2 (`index 19`) rewards the policy only when it outputs the ground-truth $X$ (despite the KV cache containing $X'$!), the policy gradient actively **trains the Step-2 attention head to ignore its own KV cache**.
   - By sampling actions $a_t \sim \pi_{\text{rollout}}(\cdot|s_t)$ from the quantized/draft KV cache and advancing along verified causal prefixes during training, we isolate the exact off-policy discrepancy $\Delta_t = \log \pi_\theta^{(8)}(a_t|s_t) - \log \pi_{\text{rollout}}(a_t|s_t)$ at all 7 reasoning steps simultaneously without confounding prefix corruption. (And crucially, all reported accuracy metrics in Table 1 and Panel A are evaluated under **100% unguided, free autoregressive generation**).

2. **Extending PA-GRPO to Free-Trajectory Open-Ended CoT (No Step-Level Ground Truth):**
   - When intermediate ground-truth states are unavailable, trajectories roll out freely ($s_{t+1} = s_t \circ a_t$) and we combine two mechanisms:
     1. **Causal Prefix-Validity Truncation / Process Verification:** In verifiable domains (code, formal math, symbolic execution), gradients are masked after the first syntax/verifier violation ($\text{active}_{t} = \prod_{k < t} \mathbb{I}[\text{valid}(s_k, a_k)]$) so off-manifold suffixes never corrupt attention circuits.
     2. **Chunked Self-Speculative Draft-and-Verify Rollouts:** Instead of accepting unverified 4-layer draft tokens for an entire 1,000-token chain, the 4-layer anchor prefix drafts a short window of $K=4$ tokens, and the upper layers (`4..7`) verify all $K$ tokens in a single parallel $T=K$ micro-batch pass, rejecting or correcting draft tokens whose log-ratio exceeds $\tau_{\text{trust}}$ online during generation.

---

## Q3. Retrofitting Off-the-Shelf Pretrained Checkpoints (Without Re-Pretraining)

> **Question:** *In `pa_grpo_experiment.py`, the 8-layer model is trained with Cross-Layer KV Sharing (`owns_kv = (i < 4)`) from initialization. In production post-training, we start from an existing pretrained base checkpoint where every layer $l \in \{0\dots L-1\}$ has its own distinct $K_l, V_l$ projections. How do you avoid KV-cache holes on an off-the-shelf checkpoint?*

### Answer
Zero-shot substitution of Layer $L/2$'s KV cache into Layers $(L/2+1)\dots L$ on an unmodified checkpoint fails because each layer's key/value projections live in distinct learned subspaces. There are two practical ways to deploy PA-GRPO on an existing pretrained base model:

1. **Zero-Retrofit Path — Chunked Self-Speculative Execution (No Architecture Change):**
   - Keep all $L$ layers' native $K_l, V_l$ projections intact.
   - During rollout decoding, layers $0\dots (L/2-1)$ draft $K=4$ tokens sequentially ($T=1$ per step).
   - Every $K$ steps (or when draft predictive entropy $H(p_t) > \tau$), layers $(L/2)\dots (L-1)$ run **one parallel forward pass over the chunk of $K$ tokens ($T=K$)**.
   - This populates the exact native KV caches of layers $(L/2)\dots (L-1)$ with **zero KV-cache holes**, converts $K$ memory-bound $T=1$ upper-layer launches into a single $T=K$ pass ($K\times$ higher arithmetic intensity), and still pairs directly with PA-GRPO's A-TIS and Single-Pass Dual Co-Alignment objective.

2. **Lightweight SFT Adapter Path — Cross-KV LoRA Projection Alignment:**
   - During the standard SFT cold-start / instruction-tuning phase that precedes RL post-training, freeze the upper layers' $K_l, V_l$ caches and train lightweight low-rank query/key adapters ($W_Q^{(l)} + \Delta W_Q^{(l)}$) on layers $(L/2)\dots (L-1)$ to attend to the anchor layer's KV cache, alongside the auxiliary layer-$L/2$ dual LM head loss.

---

## Q4. Off-Policy Bias-Variance Tradeoff & "Negative-Gradient Support Leakage"

> **Question:** *(1) Truncating the importance ratio $\rho_t \in [0.65, 1.45]$ and masking tokens with $|\Delta_t| \ge 0.85$ introduces bias into the policy gradient—why doesn't this prevent convergence? (2) Why did standard negative REINFORCE advantages cause sudden policy collapse in your earlier ablations, and how does your Positive-Support Advantage relate to Reward-Weighted Regression (RWR)?*

### Answer

1. **Bias-Variance Tradeoff of Asymmetric TIS (A-TIS) + Trust-Region Masking:**
   - Unclipped off-policy importance sampling $\hat{g} = \rho_t A_t \nabla_\theta \log \pi_\theta(a_t|s_t)$ is unbiased in expectation, but its variance scales with $\text{Var}(\rho_t) = \mathbb{E}_{\pi_{\text{rollout}}}[\rho_t^2] - 1 = \chi^2(\pi_\theta \,\|\, \pi_{\text{rollout}})$, which grows **exponentially** in the log-prob discrepancy caused by `INT4`/`INT2` KV quantization.
   - By bounding $w_t \in [0.65, 1.45]$ and down-weighting outlier tokens where $|\Delta_t| \ge 0.85$ nats, A-TIS enforces a **proximal trust region around the rollout behavior policy** (analogous to V-trace in IMPALA and Retrace($\lambda$)). As LONG as $\text{sign}(w_t A_t) = \text{sign}(A_t)$ and $w_t > 0$, the clipped gradient retains a positive inner product with the true policy gradient ($\langle \hat{g}_{\text{A-TIS}}, \nabla J(\theta) \rangle > 0$), guaranteeing ascent while bounding per-step gradient variance.

2. **Negative-Gradient Support Leakage & Connection to RWR-GRPO:**
   - In our early ablations (`task-71` and `task-80`), applying unclipped negative advantages ($A_t < 0$) when the rollout sampled a wrong digit $a_t \in \{0, 1, 2, 3\} \setminus \{y_t^*\}$ caused exact accuracy to collapse from `50.0% -> 0.0%` within 10–15 steps.
   - **Mechanism ("Negative-Gradient Support Leakage"):** For softmax logits $z \in \mathbb{R}^{|V|}$, the gradient of $-\log \pi_\theta(a_t) \cdot (-|A_t|)$ pushes down $z_{a_t}$ and pushes UP the log-partition function $\log \sum_{v \in V} e^{z_v}$ across all $v \neq a_t$. Because out-of-support punctuation tokens (`]`, `:`, `=`) are never sampled at digit positions, they never receive negative pushes—so repeated negative updates on wrong digits steadily inflate untouched punctuation logits until an out-of-support punctuation token overtakes the digit support under greedy decoding, breaking sequence syntax.
   - **Solution:** Combining **Positive-Support Variance-Filtered Advantages** ($A_{i,g,t}^+ = r_{i,g,t} \cdot (\frac{1 - \mu_{i,t}}{\sigma_{i,t} + 0.15} + 0.12)$ on reasoning forks, which prioritizes unsolved prompts by $10\times$ while assigning $0$ gradient to wrong samples) with **Exact Full-Vocabulary Reference KL Anchoring** ($D_{\text{KL}}(\pi_{\text{ref}}(\cdot|s_t) \,\|\, \pi_\theta(\cdot|s_t))$) forms a group-normalized **Reward-Weighted Regression (RWR / RLOO-RWR)** objective that is provably immune to out-of-support logit drift.

---

## Q5. Scaling PA-GRPO to Disaggregated Actor-Learner Clusters

> **Question:** *In frontier RL systems, Rollout Actors (vLLM/SGLang inference workers) are disaggregated from Learner Nodes (FSDP/Megatron trainers). How does PA-GRPO map onto a disaggregated actor-learner architecture?*

### Answer
Disaggregated RL pipelines face three primary systems bottlenecks where PA-GRPO provides compounding leverage:

1. **Actor HBM Capacity & KV-Cache Bandwidth:**
   - Rollout actors generate long reasoning traces ($8\text{k}–64\text{k}$ tokens) where KV caches saturate GPU HBM and limit concurrent batch size $M$.
   - PA-GRPO's Cross-Layer KV Anchor + `INT4` KV cache cuts actor KV memory by **$50\%\text{--}87.5\%$**, allowing actors to pack **$2\times\text{--}8\times$ more concurrent rollouts per GPU** without preemption or host-swap stalls.
2. **Long-Tail CoT Straggler Mitigation:**
   - Synchronous or semi-asynchronous GRPO barriers are bottlenecked by the longest rollout in a group. Speculative branch routing (`4 -> 8L` or chunked self-speculative drafting) accelerates deterministic boilerplate/formatting segments while reserving full depth for high-entropy reasoning forks, compressing P99 rollout completion latency.
3. **Learner-to-Actor Weight Synchronization Bandwidth & Staleness Absorption:**
   - Broadcasting updated FP16/BF16 weights from FSDP Learner Nodes to hundreds of Rollout Actor GPUs after every step is a major interconnect bottleneck.
   - Under PA-GRPO, Learner Nodes can broadcast **quantized (`FP8`/`INT4`) weight deltas** or synchronize actor weights every $K$ steps: because PA-GRPO's **A-TIS ratio clipping + KL-Trust-Region Token Masking + Single-Pass Dual Co-Alignment** explicitly models and corrects for the discrepancy $\Delta_t = \log \pi_{\theta}^{\text{learner}}(a_t|s_t) - \log \pi_{\text{actor}}^{\text{approx}}(a_t|s_t)$, the exact same trust-region machinery seamlessly absorbs **both** precision asymmetry (INT4 actor vs. BF16/FP32 learner) **and** $K$-step actor weight staleness.
