# 🔐📡 Delivery-Aware Privacy Control for Federated Learning over Noisy AIoT Links

> **A Privacy–Communication Co-Design Framework for Wireless Federated Learning**

---

## 🌍 Overview

Federated Learning (FL) enables distributed AIoT devices to collaboratively train machine learning models **without centralizing raw data**, improving privacy and reducing communication load. However, real-world deployment over wireless networks introduces two major challenges:

* **📶 Unreliable communication links** causing packet loss, latency spikes, and failed training rounds.
* **🔐 Privacy leakage from gradients**, requiring Differential Privacy (DP), which introduces noise and reduces model accuracy.

Most existing approaches address these challenges separately. This project proposes a **unified delivery-aware privacy framework** that jointly considers:

✅ wireless channel reliability
✅ differential privacy budget allocation
✅ secure aggregation overhead
✅ communication efficiency
✅ privacy-utility tradeoffs

---

# 🚨 Problem Statement

In practical AIoT deployments:

* Clients communicate over **WiFi / Cellular / Bluetooth / LPWAN**
* Links experience **fading, interference, burst loss, and instability**
* Model updates may be dropped before reaching the server

At the same time:

* Gradient updates leak private information
* Differential Privacy protects gradients by adding calibrated noise
* Privacy budget is finite

### Core inefficiency:

> **Current FL systems spend privacy budget equally—even on updates unlikely to arrive.**

That means:

❌ privacy budget gets consumed
❌ communication overhead is incurred
❌ noisy gradients are generated
❌ model receives no update

**Learning utility = zero**

This is wasteful.

---

# 🎯 Research Question

> **Can privacy spending be conditioned on update delivery probability in noisy wireless federated learning?**

This project answers:

> **Yes—through delivery-aware privacy scheduling.**

---

# 💡 Proposed Solution

We introduce a unified FL framework with three integrated mechanisms.

---

## 1️⃣ Independent Markov Wireless Channel Modeling 📶

Each client has its own 3-state wireless channel:

* 🟢 GOOD
* 🟡 MODERATE
* 🔴 BAD

Transition matrix:

$$
P=
\begin{bmatrix}
0.80 & 0.15 & 0.05\\
0.25 & 0.50 & 0.25\\
0.15 & 0.35 & 0.50
\end{bmatrix}
$$

This captures:

✅ temporal channel correlation
✅ burst errors
✅ packet loss dynamics
✅ realistic per-client variability

---

## 2️⃣ Delivery-Aware Privacy Controller 🔐

Per-round privacy allocation is dynamically adjusted:

$$
\epsilon_t=
\epsilon_{base}(t)
\cdot
f_{channel}(t)
\cdot
f_{phase}(t)
\cdot
f_{urgency}(t)
$$

### Channel factor

$$
f_{channel}\in{1.4,;1.0,;0.6}
$$

GOOD → spend more privacy budget
BAD → conserve budget

---

### Phase factor

$$
f_{phase}(t)=
\beta+A\cos\left(\pi t/T\right)
$$

This front-loads budget toward earlier rounds.

---

### Urgency factor ⭐

Closed-loop control:

$$
f_{urgency}(t)=
1+s\tanh\left(
\frac{deficit}{0.1\epsilon_{total}}
\right)
$$

This ensures:

✅ no under-utilization of privacy budget
✅ dynamic catch-up after poor rounds
✅ efficient spending

**Experimental finding:** urgency is the dominant contributor.

---

## 3️⃣ Link-Aware Secure Aggregation 🔒

Secure aggregation overhead is adaptively modeled:

| Channel     | Secure Aggregation |
| ----------- | ------------------ |
| 🟢 GOOD     | ON                 |
| 🟡 MODERATE | ON                 |
| 🔴 BAD      | OFF                |

Benefits:

✅ reduced bandwidth usage
✅ avoids wasted cryptographic traffic
✅ preserves DP guarantees

---

# 🧠 Privacy Efficiency Principle

Expected learning utility:

$$
U_t=P(delivery)\times G_t
$$

Privacy cost:

$$
C_t=\epsilon_t
$$

Optimization objective:

$$
\max \frac{U_t}{C_t}
$$

Meaning:

> **maximize learning gain per unit privacy spent**

---

# 🏗️ System Pipeline

```text
WISDM Dataset
     ↓
Dirichlet Non-IID Partitioning
     ↓
Local DP-SGD Training
     ↓
Markov Channel Simulation
     ↓
Delivery-Aware Privacy Scheduling
     ↓
Adaptive Aggregation
     ↓
RDP Privacy Accounting
     ↓
Global Model Update
```

Frameworks:

* PyTorch
* Opacus

Dataset:

* WISDM Dataset

---

# 📊 Key Experimental Findings

## Privacy cost dominates

Reliable + No DP:

$$
94.75%
$$

Reliable + DP:

$$
78.02%
$$

Noisy + DP:

$$
75.19%
$$

### DP penalty

$$
16.73%
$$

### Link penalty

$$
2.83%
$$

### Insight

> **Differential Privacy contributes ~6× more accuracy loss than noisy wireless links.**

---

## Urgency dominates scheduling

Removing urgency:

$$
75.19%\rightarrow73.62%
$$

Gain:

$$
+1.57%
$$

### Insight

> **Closed-loop budget control matters more than heuristic channel weighting.**

---

## Communication savings

Adaptive secure aggregation:

$$
38.3%
$$

less bandwidth consumption.

---

# 🚀 Contributions

✅ Delivery-aware privacy scheduling
✅ Closed-loop privacy budget controller
✅ Independent wireless channel modeling
✅ Adaptive aggregation overhead control
✅ Privacy-link cost decomposition
✅ AIoT federated learning validation

---

# 🔮 Future Work

* Full cryptographic secure aggregation
* Continuous channel scoring
* Real wireless trace validation
* Adaptive clipping
* CNN / Transformer baselines
* Larger-scale deployments (1000+ clients)

---

# 🏁 Conclusion

> **Not every update deserves equal privacy budget.**

By conditioning privacy spending on delivery likelihood, federated learning can become:

🔐 more privacy efficient
📡 more communication aware
🧠 more learning effective
⚖️ better balanced in privacy vs utility

---

**Author:** Israel Barnabas\\
**Research Area:** Federated Learning • Differential Privacy • Wireless AIoT • Edge Intelligence
