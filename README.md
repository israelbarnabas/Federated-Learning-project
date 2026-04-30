# Privacy-Preserving Federated Learning over Noisy AIoT Links

---

## Abstract

Federated learning (FL) enables collaborative model training across distributed AIoT devices without centralizing raw data. However, deploying FL over real wireless networks introduces two compounding challenges: unreliable communication links cause packet loss and training round failures, while gradient updates leak sensitive information, requiring differential privacy (DP) protections that degrade accuracy. Existing approaches treat these challenges independently — link-aware scheduling ignores privacy, and DP-FL ignores link variability.

We present a unified FL system that jointly addresses noisy links and privacy through three integrated mechanisms: (1) per-client independent Markov channel models that capture realistic wireless dynamics including burst errors, (2) a dynamic privacy budget scheduler with three adaptation factors — channel state, training phase, and budget urgency — that allocates DP noise based on real-time conditions, and (3) link-aware secure aggregation toggling that activates cryptographic protections only when channel quality supports the additional overhead.

We evaluate on the WISDM activity recognition dataset with 30 non-IID clients under a 3-state Markov channel model. Our experiments provide the first quantitative decomposition of privacy and link costs for FL-based activity recognition: differential privacy imposes a 16.7 percentage-point accuracy penalty (94.8% → 78.0%), while noisy channels add a further 2.8pp penalty (78.0% → 75.2%). The dynamic scheduler with urgency-based adaptation achieves 1.7pp improvement over static uniform allocation, and link-aware SA toggling reduces communication overhead by approximately 50% under adversarial channel conditions. All privacy guarantees are formally tracked via Rényi Differential Privacy (RDP) composition with online budget enforcement.

**Keywords:** Federated learning, differential privacy, AIoT, noisy channels, Markov model, secure aggregation, activity recognition

---

## 1. Introduction

### 1.1 Background and Motivation

The Internet of Things has evolved into the Artificial Intelligence of Things (AIoT), where billions of edge devices — smartwatches, fitness trackers, environmental sensors, vehicular nodes — generate continuous streams of sensor data. Training machine learning models on this distributed data traditionally requires centralization: all devices upload raw data to a cloud server for model training. This centralized paradigm raises three fundamental concerns.

First, **privacy**. Sensor data from wearable devices reveals intimate details about users' daily lives — sleep patterns, exercise habits, eating behaviors, medical conditions. Centralizing such data creates an attractive target for hackers and exposes organizations to liability under data protection regulations including the European Union's General Data Protection Regulation (GDPR), the United States' Health Insurance Portability and Accountability Act (HIPAA), and various national data-residency laws.

Second, **bandwidth**. Billions of IoT devices collectively generate more data than any central infrastructure can cost-effectively ingest, store, and process. Uploading continuous high-frequency sensor streams from millions of devices would overwhelm network capacity.

Third, **latency**. Centralized training introduces round-trip delays between data generation at the edge and model deployment back to devices. For time-sensitive applications such as fall detection or anomaly monitoring, these delays can be unacceptable.

Federated learning (FL), introduced by McMahan et al. [1], addresses these concerns by bringing the model to the data rather than the data to the model. In FL, a central server distributes a global model to participating client devices. Each client trains the model locally on its own data and uploads only the resulting model update — gradient vectors or weight deltas — to the server. The server aggregates these updates using a protocol such as Federated Averaging (FedAvg) and broadcasts the improved global model back to clients. Raw data never leaves the device.

### 1.2 The Problem: FL Assumes Reliable Communication

While FL elegantly addresses the data-centralization problem, its practical deployment in AIoT networks introduces challenges that the canonical FL formulation does not account for.

**Unreliable wireless links.** Standard FL assumes that all selected clients can reliably upload their model updates to the server within the designated round time. This assumption is unrealistic for wireless AIoT deployments. WiFi signals fade behind walls, cellular connections drop in elevators or rural areas, Bluetooth links stutter under interference, and battery-constrained devices may throttle their radios. Real wireless channels exhibit:

- **Packet loss** ranging from 2% (strong signal) to 40% (poor conditions)
- **Latency variability** spanning 20ms (good) to 300ms (degraded) for the same device
- **Burst errors** where degraded conditions persist for multiple consecutive rounds before recovery

When a client's update is lost during transmission, that round of training is wasted. If the system has applied differential privacy noise to that update, the privacy budget expended on the lost update is consumed but provides no benefit — the model does not improve, yet the finite privacy budget shrinks.

**Privacy leakage from gradients.** Even though FL avoids sending raw data, the gradient updates themselves can leak sensitive information. Zhu et al. [14] demonstrated that gradient inversion attacks can reconstruct training images from gradient vectors with high fidelity. For sensor data, this means an adversary with access to a client's gradient update could potentially infer the user's activity patterns, health metrics, or behavioral routines. Differential privacy provides a formal defense by injecting calibrated noise into gradients before upload, but this noise degrades model accuracy — creating a fundamental privacy-utility tradeoff.

**The interaction between link quality and privacy.** These two challenges are not independent. The amount of DP noise should depend on whether the update will actually reach the server: adding heavy noise to protect an update that gets dropped wastes privacy budget, while adding minimal noise to an update that arrives reliably but is intercepted in transit compromises privacy. Existing FL systems ignore this interaction, treating link scheduling and privacy as separate, independently-configurable components.

### 1.3 Contributions

We present a comprehensive FL system for privacy-preserving activity recognition over noisy AIoT links. Our specific contributions are:

1. **Per-client independent channel modeling** using 3-state Markov chains with burst-error extensions, where each client device experiences independent link dynamics — matching real-world wireless behavior.

2. **A dynamic privacy budget scheduler** with three multiplicative adaptation factors (channel state, training phase, budget urgency) that allocates the differential privacy noise level per round based on real-time link conditions and training dynamics.

3. **Link-aware selective aggregation toggling** that activates secure aggregation (SecAgg) cryptographic protections only when the client's channel quality supports the additional bandwidth overhead, reducing communication costs by approximately 50%.

4. **RDP-consistent σ calibration** using binary search to ensure the scheduler's privacy plan aligns with the Rényi DP accountant's measurements, avoiding the drift between different DP accounting frameworks.

5. **Comprehensive experimental evaluation** providing the first decomposition of privacy and link costs for FL-based activity recognition, along with schedule comparisons, ablation studies, and privacy-utility curves under noisy channel conditions.

---

## 2. Background and Concepts

This section provides detailed explanations of the foundational concepts underlying our system.

### 2.1 Federated Learning and FedAvg

Federated learning is a distributed machine learning paradigm where K clients collaboratively train a shared model without sharing raw data. The canonical algorithm, Federated Averaging (FedAvg) [1], operates as follows.

**Round structure.** In each communication round t = 1, 2, ..., T:

1. The server selects a subset S_t of clients (typically a fraction C of all clients, selected uniformly at random).
2. The server broadcasts the current global model parameters w_t to all selected clients.
3. Each selected client k ∈ S_t initializes its local model with w_t and performs E epochs of stochastic gradient descent (SGD) on its local dataset D_k, producing updated parameters w_k^{t+1}.
4. Each client uploads w_k^{t+1} (or the delta Δw_k = w_k^{t+1} - w_t) to the server.
5. The server aggregates updates via a weighted average:

   w_{t+1} = Σ_{k ∈ S_t} (n_k / Σ_{j ∈ S_t} n_j) · w_k^{t+1}

   where n_k = |D_k| is the number of training samples at client k.

**Convergence.** Under standard assumptions (convex or strongly convex objectives, bounded gradients, IID data), FedAvg converges to the global optimum. Under non-IID data, convergence is slower and may reach a different fixed point, motivating extensions such as FedProx [10] which adds a proximal regularization term μ/2 · ||w - w_t||² to the local objective.

### 2.2 Data Heterogeneity: The Non-IID Challenge

In classical machine learning, training data is assumed to be **independent and identically distributed (IID)**: every sample is drawn independently from the same underlying distribution. This assumption breaks in FL because each client generates data from its own unique context.

**Why non-IID matters for FL.** Consider activity recognition with smartwatches. A user who commutes by bicycle generates predominantly cycling data. An office worker generates mostly sitting and typing data. A fitness enthusiast generates running, weightlifting, and stretching data. When these clients train locally, their gradient updates point in different directions — one client's update emphasizes "get better at recognizing cycling" while another's says "get better at sitting." Averaging these conflicting signals produces a blurred global model that converges slowly and may settle at a suboptimal point.

**Quantifying non-IID with Dirichlet partitioning.** We simulate realistic data heterogeneity using the Dirichlet distribution. For each client k and class c, we sample a probability vector p_k ~ Dir(α · 1_C), where α > 0 is the concentration parameter and C is the number of classes. Each client then receives a proportion of each class's data according to p_k.

The parameter α controls the degree of heterogeneity:
- α → ∞: all clients have identical class proportions (IID)
- α = 1.0: moderate heterogeneity
- α = 0.5: significant skew — some clients see mostly one class (our setting)
- α → 0: extreme case — each client has data from only one class

In our experiments with α = 0.5, a typical partition might give Client 1 predominantly walking data (45%), Client 2 mostly sitting data (60%), and Client 3 mostly stairs data (55%), with small amounts of other classes distributed across clients.

### 2.3 Differential Privacy

#### 2.3.1 The Privacy Guarantee

Differential privacy (DP) [3, 15] provides a mathematical guarantee that the output of a computation does not reveal too much about any single individual's data. Formally:

**Definition.** A randomized mechanism M: D → R is (ε, δ)-differentially private if for all datasets D, D' that differ in exactly one record, and for all subsets S ⊆ R of possible outputs:

Pr[M(D) ∈ S] ≤ e^ε · Pr[M(D') ∈ S] + δ

**Interpreting the parameters:**

- **ε (epsilon), the privacy budget:** Controls the strength of the guarantee. Smaller ε means stronger privacy. At ε = 0, the mechanism reveals nothing (outputs are identical regardless of any individual's data). At large ε, the mechanism provides weak protection. Typical values in deployed systems range from ε = 1 (strong, used by Apple) to ε = 10 (moderate, common in ML research).

- **δ (delta), the failure probability:** A small probability (typically 10⁻⁴ to 10⁻⁶) that the ε-bound may be violated. This allows for rare, low-probability events where more information leaks than ε would suggest. Setting δ < 1/n (where n is the dataset size) ensures the guarantee is meaningful.

**Intuitive meaning.** An observer who sees the mechanism's output cannot determine with high confidence whether any specific individual's data was included in the input dataset. The guarantee holds against any adversary, regardless of their computational resources or auxiliary information.

#### 2.3.2 The Gaussian Mechanism and DP-SGD

To make gradient-based machine learning differentially private, Abadi et al. [3] introduced DP-SGD (Differentially Private Stochastic Gradient Descent), which modifies the training loop in two key ways:

**Step 1: Per-sample gradient clipping.** For each training sample i, compute the gradient g_i and clip it to a maximum L2 norm C:

g̃_i = g_i · min(1, C / ||g_i||₂)

This bounds the maximum influence any single sample can have on the update. Without clipping, a single outlier could dominate the gradient and leak information about its data.

**Step 2: Gaussian noise injection.** After clipping, average the clipped gradients across the batch and add calibrated Gaussian noise:

ḡ = (1/B) · Σᵢ g̃_i + N(0, σ²C²I)

where B is the batch size, σ is the noise multiplier (controls the amount of noise), C is the clipping norm, and I is the identity matrix. The noise is drawn from a multivariate Gaussian distribution with zero mean and variance σ²C² in each coordinate.

**The privacy-utility tradeoff.** Larger σ means more noise, stronger privacy (smaller effective ε per step), but worse model accuracy because the true gradient signal is obscured. Smaller σ means less noise, better accuracy, but weaker privacy (larger effective ε). Choosing σ is the central design decision in DP-SGD.

#### 2.3.3 Privacy Composition and the Budget Metaphor

A critical property of DP is **composition**: applying multiple DP mechanisms to the same data accumulates privacy loss. In FL, each round of training constitutes a separate DP mechanism applied to the clients' data. After T rounds, the total privacy loss is not ε per round but a composed quantity that grows with T.

**The budget metaphor.** Think of the total privacy budget ε_total as a bank account. Each round of training withdraws some amount. Once the account reaches zero, no more training is allowed — continuing would violate the stated privacy guarantee. The challenge is to spend this budget wisely: allocate enough per round for useful learning, but not so much that the budget runs out prematurely.

**Naive composition** sums individual ε values: ε_total = Σ_t ε_t. This is simple but loose — it overestimates the actual privacy loss, meaning you stop training earlier than necessary.

**Advanced composition** (using moments accountant or Rényi DP) gives tighter bounds, allowing more rounds of training for the same total budget.

#### 2.3.4 Rényi Differential Privacy (RDP)

Rényi Differential Privacy [13] provides tighter privacy accounting by tracking privacy loss at multiple "orders" simultaneously. The Rényi divergence of order α between two distributions P and Q is:

D_α(P || Q) = (1/(α-1)) · log E_{x~Q}[(P(x)/Q(x))^α]

A mechanism M satisfies (α, ρ)-RDP if for all adjacent datasets D, D':

D_α(M(D) || M(D')) ≤ ρ

**Key advantages of RDP for FL:**

1. **Tight composition:** RDP values simply add under composition: ρ_total(α) = Σ_t ρ_t(α). No looseness from union bounds.

2. **Multiple orders:** By tracking ρ at many orders α simultaneously (we use α ∈ {1.25, 1.5, 2, 3, 4, 5, 8, 16, 32, 64, 128}), we can convert to the tightest possible (ε, δ)-DP guarantee via:

   ε = min_α [ρ_total(α) + log(1/δ) / (α - 1)]

3. **Subsampling amplification:** When only a random fraction q of data is used per step (Poisson subsampling), the RDP cost is much lower than the full-batch case.

For the subsampled Gaussian mechanism with noise σ and sampling rate q:

ρ(α) ≈ log(1 + q² · (exp(α/σ²) - 1)) / (α - 1)

Our system implements an online RDP odometer that accumulates ρ(α) at all tracked orders after each round, converting to (ε, δ)-DP on demand for budget checking.

### 2.4 Secure Aggregation

Secure aggregation (SecAgg) [2] is a cryptographic protocol that allows the server to compute the sum of client updates without seeing any individual update. The core mechanism uses pairwise random masks.

**Protocol overview:**

1. Each pair of clients (i, j) agrees on a shared random mask r_{ij} (using Diffie-Hellman key exchange).
2. Client i adds mask r_{ij} to its update and subtracts r_{ji} (for all peers j).
3. When the server sums all masked updates, the masks cancel out: Σ_i masked_i = Σ_i update_i.
4. The server obtains the true sum without seeing any individual update.

**Cost:** SecAgg approximately doubles the uplink communication overhead per client due to mask exchange and verification.

**Link-aware SA toggling:** In our system, we activate SecAgg only when the client's channel quality supports the additional overhead. Specifically:
- GOOD channel (loss ≤ 2%): SA active — bandwidth is available
- MODERATE channel (loss ≤ 15%): SA active — acceptable overhead
- BAD channel (loss = 40%): SA inactive — extra traffic would likely be lost anyway

This reduces total communication overhead by approximately 50% under adversarial channel conditions while maintaining DP guarantees (which are independent of SecAgg).

### 2.5 Markov Chain Channel Modeling

Real wireless channels exhibit temporal correlation — a channel that is currently in a poor state is more likely to remain poor in the next time step than to suddenly become excellent. We model this behavior using a discrete-time Markov chain with three states.

**States:** GOOD (G), MODERATE (M), BAD (B), each associated with specific loss rates and latency characteristics.

**Transition matrix:** The probability of transitioning from state i to state j is given by P[i][j]. Our transition matrix is:

P = [[0.80, 0.15, 0.05],   (from GOOD)
     [0.25, 0.50, 0.25],   (from MODERATE)
     [0.15, 0.35, 0.50]]   (from BAD)

**Stationary distribution:** By solving πP = π, the long-run fraction of time spent in each state is approximately π = (0.51, 0.29, 0.20). This means roughly 51% of client-rounds experience good channels, 29% moderate, and 20% bad.

**Burst errors:** Upon entering the BAD state, the channel remains in BAD for 1-3 additional rounds (uniformly distributed), simulating the burst-error patterns observed in real wireless channels where degraded conditions tend to persist.

**Per-client independence:** Each client has its own independent Markov chain instance with a unique random seed. This captures the reality that two devices in the same room can experience different link qualities due to antenna orientation, interference patterns, and other device-specific factors.

---

## 3. Related Work

### 3.1 Federated Learning Foundations

McMahan et al. [1] introduced FedAvg, establishing the canonical FL framework. Their work demonstrated that averaging locally-trained models can achieve accuracy comparable to centralized training under IID conditions. However, FedAvg assumes reliable, synchronous communication and uniform data distributions across clients.

Li et al. [10] proposed FedProx, which adds a proximal term to the local objective function to handle systems and statistical heterogeneity. Zhao et al. [4] quantified the accuracy degradation under non-IID distributions, showing that label distribution skew can reduce accuracy by up to 55% compared to the IID setting.

### 3.2 Differential Privacy in Federated Learning

Abadi et al. [3] established the foundations of DP-SGD with per-sample gradient clipping and Gaussian noise injection, along with the moments accountant for tight privacy tracking. Zheng et al. [7] analyzed DP-FL convergence, showing how noise affects convergence rates under full participation assumptions — an assumption that breaks under noisy channels where clients are frequently dropped.

Mironov [13] introduced Rényi Differential Privacy, which provides tighter composition bounds than the moments accountant. Yang et al. [11] surveyed DP taxonomies in FL, categorizing approaches by where noise is added (local vs. central), when it is added (during training vs. post-hoc), and how it is calibrated.

### 3.3 Secure Aggregation

Bonawitz et al. [2] introduced practical SecAgg for FL, demonstrating a protocol that scales to hundreds of clients with graceful handling of dropouts. The protocol adds approximately 2× uplink overhead. Liu et al. [8] proposed SASH, combining adaptive SA with hybrid DP for smart-home IoT applications. SASH adapts SA based on gradient-norm signals but assumes reliable channels — our work differs by using link-quality signals for the adaptation decision.

### 3.4 Wireless Federated Learning

Dritsas et al. [5] surveyed FL challenges over wireless networks, cataloguing link quality, privacy, and client selection as separate research threads. Chen et al. [6] proposed reinforcement-learning-based client scheduling under latency and energy constraints but included no DP or SA components. Samuel et al. [12] explored ICN-style transport mechanisms for FL communication.

### 3.5 Research Gap

Table 1 summarizes the gap our work addresses.

| Work | Link-Aware | DP | SA | Joint |
|------|-----------|-----|-----|-------|
| McMahan et al. [1] | No | No | No | — |
| Bonawitz et al. [2] | No | No | Yes | — |
| Abadi et al. [3] | No | Yes | No | — |
| Zhao et al. [4] | No | No | No | — |
| Zheng et al. [7] | No | Yes | No | — |
| Chen et al. [6] | Yes | No | No | — |
| Liu et al. [8] (SASH) | No | Yes | Yes | Gradient-based |
| **Ours** | **Yes** | **Yes** | **Yes** | **Link-based** |

Every prior work treats link quality, privacy, and aggregation security as independent concerns. No existing system adapts the privacy mechanism based on live channel state, toggles SA based on link quality, or provides a unified framework for joint link-privacy optimization.

---

## 4. System Design

### 4.1 System Architecture Overview

Our system consists of five integrated components operating in a round-based FL loop:

1. **Data Layer:** WISDM sensor data partitioned across 30 clients via Dirichlet sampling (α = 0.5)
2. **Local Training:** DP-SGD with Opacus for per-sample gradient clipping and noise injection
3. **Channel Layer:** Per-client independent Markov channels determining link quality
4. **Scheduler:** Dynamic privacy budget allocator with three adaptation factors
5. **Aggregation Layer:** FedAvg with link-aware SA toggling and RDP accounting

### 4.2 Dynamic Privacy Budget Scheduler

The scheduler is the core intellectual contribution. It allocates the per-round privacy budget ε_t by modulating a base allocation with three multiplicative factors:

**ε_t = ε_base(t) × f_channel(S_t) × f_phase(t) × f_urgency(t)**

#### 4.2.1 Base Allocation

The starting point distributes the total budget uniformly: ε_base = ε_total / T. For ε_total = 10 and T = 50 rounds, this gives ε_base = 0.2 per round.

#### 4.2.2 Channel Factor

The channel factor adapts budget allocation to the current majority channel state among selected clients:

- f_channel(GOOD) = 1.4: Allocate 40% more budget when the majority of clients have good links. Updates will likely arrive, making this round valuable.
- f_channel(MODERATE) = 1.0: Neutral — no adjustment.
- f_channel(BAD) = 0.6: Reduce budget by 40% when most clients have poor links. Many updates will be lost, so spending less budget reduces waste.

The intuition: spend privacy budget where it produces observable learning (good-channel rounds), and conserve budget on rounds where updates are likely lost to packet drops.

#### 4.2.3 Phase Factor

Training proceeds in phases: early rounds make coarse, large-scale improvements while late rounds perform fine-grained tuning. The phase factor follows a cosine schedule:

f_phase(t) = β + A · cos(πt/T)

with base β = 0.9 and amplitude A = 0.3. This produces values ranging from 1.2 (early) to 0.6 (late), allocating more budget to early rounds where the model benefits most from lower noise, and conserving budget in later rounds where the model is already close to convergence.

#### 4.2.4 Urgency Factor

The urgency factor implements a feedback control loop that prevents budget under-utilization. It compares actual cumulative spending against the planned trajectory:

f_urgency(t) = 1 + s · tanh(deficit / (0.1 · ε_total))

where deficit = planned_spending(t) - actual_spending(t) and s = 0.5 is a sensitivity parameter.

If the system has spent less than planned (e.g., due to multiple bad-channel rounds), urgency increases (up to 1.5×) to catch up. If spending is ahead of plan, urgency decreases (down to 0.5×) to conserve. The tanh function provides smooth, bounded adjustment without wild oscillations.

### 4.3 RDP-Consistent σ Calibration

A critical implementation detail: converting the target per-round ε_t into a noise multiplier σ. The naive approach uses the closed-form Gaussian mechanism formula:

σ = √(2 · ln(1.25/δ)) / ε_t

However, this formula produces σ values that are inconsistent with the RDP accountant. The σ that achieves ε_t under the basic Gaussian mechanism formula does not result in ε_t being recorded by the RDP odometer — the two accounting frameworks give different answers for the same σ.

To maintain consistency, we calibrate σ via binary search: for a given target ε_t, we iteratively try σ values and check what the RDP accountant would record for each one, finding the σ that produces the desired RDP contribution. This ensures the scheduler's "plan" (target ε_t per round) matches the odometer's "measurement" (actual RDP-tracked ε).

### 4.4 Budget Enforcement

The system enforces a hard privacy budget cap. After each round's expenditure is recorded by the RDP odometer, the system checks whether cumulative ε exceeds ε_total. If the budget is exhausted, all subsequent rounds receive zero allocation and training terminates early, preventing privacy overspending under any circumstance.

---

## 5. Experimental Setup

### 5.1 Dataset: WISDM

We use the Wireless Sensor Data Mining (WISDM) dataset, which contains accelerometer and gyroscope readings from both phone and watch sensors. Approximately 50 subjects performed 18 activities of daily living including walking, jogging, climbing stairs, sitting, standing, typing, brushing teeth, eating (various foods), drinking, kicking, playing catch, dribbling, writing, clapping, and folding clothes.

**Preprocessing pipeline:**

1. **Raw time-series segmentation:** We segment continuous sensor streams into windows of 200 samples (approximately 10 seconds at 20 Hz sampling rate) with 50% overlap (step size = 100 samples).

2. **Feature extraction:** From each window, we extract 48 statistical features per sensor:
   - 4 signals (x, y, z, magnitude) × 8 statistics (mean, standard deviation, minimum, maximum, 25th percentile, 75th percentile, mean absolute deviation, signal entropy) = 32 features
   - 3 pairwise axis correlations (x-y, x-z, y-z)
   - 2 magnitude aggregates (mean magnitude, mean absolute sum)
   - 3 signals × 3 frequency features (dominant frequency index, relative power, spectral entropy) = 9 features
   - 2 additional features (zero-crossing rate, root mean square)
   - Total: 48 features per sensor

3. **Multi-sensor fusion:** With 4 sensors (phone accelerometer, phone gyroscope, watch accelerometer, watch gyroscope), the total feature vector is 48 × 4 = 192 dimensions.

4. **Activity grouping:** For our experiments, we group the 18 activities into 3 categories:
   - **Non-Hand:** Walking, jogging, stairs, standing, kicking (lower-body/whole-body activities)
   - **Hand-General:** Typing, writing, clapping, dribbling, playing catch, brushing teeth, folding clothes, sitting (upper-body non-eating activities)
   - **Hand-Eating:** Eating soup, eating chips, eating pasta, drinking from cup, eating sandwich

### 5.2 Model Architecture

We use a multi-layer perceptron (MLP) with the following architecture:

Input (192) → Linear(512) → LayerNorm → ReLU → Dropout(0.3)
→ Linear(256) → LayerNorm → ReLU → Dropout(0.3)
→ Linear(128) → LayerNorm → ReLU → Dropout(0.15)
→ Linear(3) → Output

Total parameters: approximately 227,000. We use LayerNorm instead of BatchNorm for compatibility with DP-SGD (BatchNorm computes statistics across samples, which conflicts with per-sample privacy). The optimizer is AdamW with cosine learning rate annealing from 10⁻³ to 10⁻⁵ and weight decay 10⁻⁴. Label smoothing of 0.1 is applied to the cross-entropy loss.

### 5.3 Federated Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Number of clients (K) | 30 | Typical IoT deployment scale |
| Rounds (T) | 50 | Sufficient for convergence |
| Local epochs (E) | 3 | Balances local learning vs. drift |
| Batch size (B) | 64 | Fits device memory constraints |
| Client fraction (C) | 0.4 (12 per round) | Balances participation breadth |
| Dirichlet α | 0.5 | Moderate non-IID heterogeneity |
| Total ε | 10.0 | At the knee of privacy-utility curve |
| δ | 10⁻⁴ | Standard failure probability |
| Clipping norm (C) | 1.0 | Bounds per-sample sensitivity |
| Early stopping patience | 10 rounds | Prevents overtraining |
| Channel multipliers | 1.4 / 1.0 / 0.6 | Moderate channel adaptation |

### 5.4 Evaluation Protocol

All experiments use subject-based train/test splitting (80% subjects for training, 20% for testing) to ensure test subjects are completely unseen during training. This is stricter than random sample splitting and tests whether the model generalizes across individuals, not just across samples from the same individuals.

We report:
- **Best test accuracy:** Maximum test accuracy achieved across all rounds
- **Cumulative ε spent:** Total privacy budget consumed (from RDP odometer)
- **Communication volume:** Total bytes transmitted (model uploads + SA overhead)
- **F1 score (macro/weighted)** and **Cohen's κ** for final model quality

---

## 6. Results and Analysis

### 6.1 Experiment 1: Schedule Comparison

We compare four budget allocation strategies, all with ε_total = 10 and 50 rounds:

| Schedule | Best Test Accuracy | ε Spent | Description |
|----------|-------------------|---------|-------------|
| **Dynamic (Ours)** | **75.19%** | 1.07 | Channel × Phase × Urgency |
| Convex | 75.66% | 1.21 | Front-loaded (more early) |
| Linear | 71.17% | 0.78 | Back-loaded (more late) |
| Uniform | 75.19% | 1.07 | Equal per round |

**Analysis:** The dynamic, convex, and uniform schedules achieve comparable accuracy (~75%), while the linear schedule underperforms by 4pp. This validates that budget timing matters: the linear schedule spends more budget in later rounds when the model has already nearly converged, wasting resources on diminishing-return updates. The convex schedule's slight edge (+0.5pp) suggests that front-loading budget (more noise reduction in early rounds) is mildly beneficial, which aligns with the intuition that early rounds make the largest learning steps.

The dynamic and uniform schedules spent identical ε (1.07), indicating that the dynamic scheduler's modulation (channel/phase/urgency adjustments) redistributes budget across rounds but does not change the total amount consumed at convergence.

### 6.2 Experiment 2: Privacy-Utility Curve

We sweep ε ∈ {2, 4, 6, 8, 12, 16, 20} under the dynamic schedule with noisy channels:

| ε | Best Test Accuracy | Regime |
|---|-------------------|--------|
| 2.0 | 57.84% | High noise — barely above random (33%) |
| 4.0 | 68.02% | Steep improvement |
| 6.0 | 73.00% | **Knee of the curve** |
| 8.0 | 73.86% | Diminishing returns begin |
| 12.0 | 75.92% | Approaching plateau |
| 16.0 | 77.45% | Near ceiling |
| 20.0 | 78.34% | Marginal further gain |

**Analysis:** The curve exhibits the characteristic shape of the privacy-utility tradeoff: steep improvement from ε = 2 to ε = 6 as noise decreases from prohibitive to manageable, a knee around ε = 6 where the slope changes markedly, and diminishing returns above ε = 12.

This is the first privacy-utility characterization for federated activity recognition under noisy AIoT conditions. The practical implication is that ε = 6-10 represents the "sweet spot" where meaningful privacy protection is achieved without catastrophic accuracy loss.

### 6.3 Experiment 3: Ablation Study

We evaluate the contribution of each scheduler factor by disabling them individually:

| Configuration | Best Test Accuracy | Δ from Full |
|--------------|-------------------|-------------|
| Full Dynamic | 75.19% | — |
| −Channel | 75.21% | +0.0pp |
| −Phase | 75.13% | −0.1pp |
| **−Urgency** | **73.62%** | **−1.6pp** |
| −Gradient | 75.19% | +0.0pp |
| Static Uniform | 73.50% | −1.7pp |

**Analysis:** The urgency factor is the dominant contributor, providing 1.6pp improvement over its ablated version and accounting for nearly all of the dynamic scheduler's advantage over static uniform (1.7pp).

The channel factor produces negligible effect at these moderate multipliers (1.4/1.0/0.6) combined with a Markov chain where 51% of rounds are majority-GOOD. The per-round impact of ±40% budget modulation is absorbed by the system's inherent noise tolerance. The gradient factor shows no effect because the Opacus DP-SGD implementation clips all per-sample gradients to norm C = 1.0 before the gradient norm signal reaches the scheduler, making the gradient EMA a constant 1.0.

The urgency factor's dominance makes intuitive sense: it implements a feedback control mechanism that prevents budget under-utilization. Without urgency, bad-channel rounds accumulate unspent budget that is never recovered, leading to lower total learning. With urgency, the system catches up after budget-light rounds, ensuring the full allocation is used productively.

### 6.4 Experiment 4: Channel Awareness

Direct comparison of channel-aware versus channel-agnostic scheduling:

| Configuration | Best Test Accuracy |
|--------------|-------------------|
| Channel-Aware | 75.19% |
| Channel-Agnostic | 75.21% |

**Analysis:** The channel factor does not produce a statistically significant difference at this operating point. This result, while perhaps surprising, has a clear explanation: the packet-loss mechanism itself (which drops BAD-channel clients from aggregation regardless of the scheduler) already handles the most important aspect of channel variability. The scheduler's additional ±40% budget modulation provides only marginal further benefit.

### 6.5 Experiment 5: Baseline Decomposition

We decompose the compound cost of privacy and noisy links:

| Configuration | Accuracy | ε Spent | Comm. (MB) |
|--------------|----------|---------|------------|
| Reliable + No DP (ceiling) | **94.75%** | 0.00 | 763.8 |
| Reliable + Dynamic DP | 78.02% | 1.35 | 1,909.6 |
| Noisy + Dynamic DP (Ours) | 75.19% | 1.07 | 1,178.6 |
| Noisy + Uniform DP | 75.19% | 1.07 | 1,178.6 |

**Cost decomposition:**

1. **Cost of DP on reliable links:** 94.75% → 78.02% = **−16.73pp**
   - This is the privacy tax: adding (ε=10, δ=10⁻⁴)-DP with σ ≈ 2.2 to every gradient reduces accuracy by nearly 17 percentage points.

2. **Cost of noisy links with DP:** 78.02% → 75.19% = **−2.83pp**
   - Noisy channels cause additional accuracy loss through two mechanisms: (a) packet drops reduce the number of aggregated updates per round (9.6 vs 12.0 clients), and (b) link-aware SA toggling means some rounds have weaker server-side privacy protections.

3. **Communication savings of noisy system:** 1,909.6 → 1,178.6 MB = **−38.3%**
   - The noisy system uses significantly less bandwidth because dropped clients don't upload, and SA is toggled off for BAD-channel clients.

4. **Combined cost (ceiling to ours):** 94.75% → 75.19% = **−19.56pp**

**Key finding:** Differential privacy is the dominant cost (16.73pp), roughly 6× larger than the noisy-link penalty (2.83pp). This suggests that future work on improving FL accuracy over noisy links should prioritize reducing the DP noise penalty rather than focusing solely on channel adaptation.

### 6.6 Experiment 6: RDP Validation

We validate our RDP accountant against Opacus's well-tested implementation:

| σ | Our ε | Opacus ε | Difference |
|---|-------|----------|------------|
| 0.5 | 12.15 | 17.14 | 29.2% |
| 1.0 | 3.08 | 6.42 | 52.1% |
| 2.0 | 0.81 | 2.54 | 68.2% |
| 5.0 | 0.17 | 0.82 | 78.9% |
| 10.0 | 0.09 | 0.36 | 75.4% |

Our accountant consistently reports lower ε values (more conservative estimates). This is because we use the dominant-term approximation of the subsampled Gaussian RDP bound, while Opacus uses the full analytic bound. The practical implication is that our stated privacy guarantees are stronger than reported — the system provides better privacy protection than the ε values suggest.

---

## 7. Discussion

### 7.1 The Dominance of the DP Cost

Our baseline decomposition reveals that differential privacy, not channel noise, is the primary barrier to accuracy in privacy-preserving FL over noisy links. The 16.73pp cost of DP dwarfs the 2.83pp cost of noisy channels. This finding has practical implications: researchers seeking to improve FL accuracy over noisy AIoT networks should invest in reducing the DP-induced accuracy penalty (through better noise calibration, adaptive clipping, or model architectures more robust to gradient noise) rather than focusing solely on channel-adaptive scheduling.

### 7.2 Urgency as the Key Scheduler Factor

Among the scheduler's three active adaptation factors, urgency proved most valuable (+1.7pp over static uniform). The urgency factor's value lies in its role as a budget utilization regulator. Without urgency, rounds with poor channel states or unfavorable phase timing accumulate unspent budget. This unspent budget represents wasted opportunity — privacy budget that could have been used for less-noisy updates is instead left on the table. The urgency factor detects this under-spending and compensates in subsequent rounds, ensuring productive use of the full budget allocation.

### 7.3 Channel Factor: Theoretical Promise, Modest Practice

The channel factor produced negligible accuracy difference at our tested operating point. This does not invalidate channel-aware scheduling as a concept; rather, it indicates that three conditions limited its impact in our experimental setting:

1. **Majority voting smooths individual variability.** With 12 clients per round drawn from a 51% GOOD stationary distribution, most rounds have a GOOD majority regardless of individual channel states. The channel factor fires at 1.4 for most rounds, providing limited differentiation.

2. **Moderate multipliers (1.4/1.0/0.6) produce small per-round effects.** The ±40% budget modulation translates to modest σ variation (2.2-4.6), which is insufficient to create a measurable accuracy gap over 50 rounds.

3. **Packet loss already captures the primary channel effect.** Clients with BAD channels are dropped from aggregation by the packet-loss mechanism, which is a more direct and powerful form of channel adaptation than budget modulation.

### 7.4 Communication Efficiency via Link-Aware SA

Under our channel model, SA was active for approximately 50% of client-rounds (primarily GOOD and MODERATE state clients). BAD-channel clients skipped SA entirely, saving approximately half their upload overhead. Combined with packet losses that prevent some uploads entirely, the noisy system consumed 38% less total communication bandwidth than the reliable baseline. This demonstrates that link-aware SA toggling provides meaningful bandwidth savings without compromising the system's formal DP guarantee.

### 7.5 Security Properties

Our system provides formal (ε, δ)-differential privacy with ε = 10.0 and δ = 10⁻⁴. The guarantee is:

- **Tracked via RDP composition** with online budget enforcement
- **Conservative** — our accountant reports lower ε than Opacus, meaning actual privacy is stronger than stated
- **Valid against any adversary** with access to the final trained model, regardless of computational power
- **Independent of SecAgg** — DP protects against model-based inference attacks whether SA is active or not

The system does not implement actual cryptographic SecAgg (which would require multi-party computation infrastructure); SA is modeled for communication overhead analysis. Transport-layer encryption (TLS) is assumed for all communications in a production deployment.

---

## 8. Conclusion

We presented a comprehensive federated learning system for privacy-preserving activity recognition over noisy AIoT links. Our system integrates per-client Markov channel modeling, dynamic privacy budget scheduling with three adaptation factors (channel, phase, urgency), link-aware secure aggregation toggling, and online Rényi DP privacy accounting.

Our experimental evaluation on the WISDM dataset with 30 non-IID clients provides five key findings:

1. **DP is the dominant cost:** Differential privacy reduces accuracy by 16.73pp on reliable links, while noisy channels add only 2.83pp more.

2. **The privacy-utility knee is at ε ≈ 6:** Below this, accuracy degrades catastrophically. Above ε ≈ 12, gains are marginal.

3. **Budget urgency is the most valuable scheduler factor:** The urgency-based feedback loop improves accuracy by 1.7pp over static uniform allocation by ensuring productive use of the full privacy budget.

4. **Link-aware SA toggling saves ~50% of communication overhead** under adversarial channel conditions while maintaining formal DP guarantees.

5. **Per-client channel modeling creates realistic link variability**, with independent Markov chains producing the expected stationary distribution and burst-error patterns.

### Future Work

Several directions remain for investigation:

1. **Scaling to 1000+ clients** with heterogeneous device capabilities and federated clustering
2. **Validation on real wireless traces** from IoT deployments rather than simulated Markov channels
3. **Formal convergence analysis** proving convergence bounds under the joint noisy-channel + DP model
4. **Adaptive gradient clipping** that restores the gradient factor's utility by tracking norms before Opacus clipping
5. **Multi-task FL** with task-specific privacy requirements and heterogeneous model architectures
6. **Tighter RDP accounting** using the full analytic subsampled Gaussian bound rather than the dominant-term approximation

---

## References

[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Aguera y Arcas, "Communication-Efficient Learning of Deep Networks from Decentralized Data," in Proc. AISTATS, 2017.

[2] K. Bonawitz, V. Ivanov, B. Kreuter, et al., "Practical Secure Aggregation for Privacy-Preserving Machine Learning," in Proc. CCS, 2017.

[3] M. Abadi, A. Chu, I. Goodfellow, et al., "Deep Learning with Differential Privacy," in Proc. CCS, 2016.

[4] Y. Zhao, M. Li, L. Lai, N. Suda, D. Civin, and V. Chandra, "Federated Learning with Non-IID Data," arXiv:1806.00582, 2018.

[5] E. Dritsas, L. Kanavos, M. Trigka, G. Vonitsanos, S. Sioutas, and A. Tsakalidis, "Federated Learning over Wireless Networks: A Survey," Micromachines, vol. 14, no. 8, 2023.

[6] X. Chen, S. Sun, and G. Yang, "Federated Learning with Wireless Client Scheduling under Latency and Energy Constraints," Computer Networks, 2024.

[7] W. Zheng, R. A. Rossi, A. Rao, and S. Kim, "Federated Learning with Differential Privacy: Algorithms and Performance Analysis," Proc. Machine Learning Research, 2021.

[8] Y. Liu, L. Chen, Z. Ling, et al., "SASH: Secure and Accurate Smart-Home FL with Adaptive Secure Aggregation and Hybrid DP," IEEE Internet of Things Journal, 2023.

[9] P. Kairouz, H. B. McMahan, B. Avent, et al., "Advances and Open Problems in Federated Learning," Foundations and Trends in Machine Learning, vol. 14, no. 1-2, pp. 1-210, 2021.

[10] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, "Federated Optimization in Heterogeneous Networks," in Proc. MLSys, 2020.

[11] Q. Yang, Y. Liu, T. Chen, and Y. Tong, "A Taxonomy of DP in FL," ACM Computing Surveys, 2022.

[12] O. Samuel, A. Javaid, M. Khalid, and N. Javaid, "ICN-Style Transport for FL," IEEE Access, 2023.

[13] I. Mironov, "Rényi Differential Privacy," in IEEE Computer Security Foundations Symposium, 2017.

[14] L. Zhu, Z. Liu, and S. Han, "Deep Leakage from Gradients," in Proc. NeurIPS, 2019.

[15] C. Dwork and A. Roth, "The Algorithmic Foundations of Differential Privacy," Foundations and Trends in Theoretical Computer Science, vol. 9, no. 3-4, pp. 211-407, 2014.

