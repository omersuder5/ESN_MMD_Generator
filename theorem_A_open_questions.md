# Open questions after the Ortega meeting

Scope: make Theorem A perfect before moving to C. B is essentially fine.

---

## 0. The decision everything hangs on: OPEN LOOP vs OUTPUT FEEDBACK

> **SUPERSEDED. See A1 below.** The framing in this section was wrong: RC12 §6 is written for output-feedback ESNs, so the choice was never open loop versus closed loop. The real change is scalar feedback to full-window feedback, and functional approximation to state-map approximation. Section kept for the record only.

Current architecture is output feedback ($\xi_j := \hat x_{j-1}$), with $k+1$ readouts.
The alternative is open loop: an ESN driven directly by the i.i.d. noise $u_t$, no feedback.

This is not a stylistic choice. It determines the answers to Q3, Q5, Q6, Q7 below.

- **Output feedback:** no closed-loop ESP, so over $\mathbb{Z}_-$ there may be no filter and no pushforward representation; burn-in cannot be removed; boundary readouts are needed.
- **Open loop:** deterministic ESP + FMP available from $\|A\|_2 L_\sigma < 1$; output law is a genuine pushforward; stationary; no burn-in; no boundary readouts.
- **Cost of going open loop:** it is a different object from the autoregressive generative model, and the thesis narrative currently commits to output feedback. Decide with Ortega before rewriting.

---

## Theorem A

**Q1. Can GO2018 Theorem 3.1 (internal approximation property) give a better proof of Theorem A?**
Motivation: the BKM triangular / conditional-quantile construction already has the form of a state-space system, with state = lagged window and input = i.i.d. uniform.

**Q2. If not, why exactly not?**
Identify precisely which hypothesis of 3.1(iii) fails and whether the technique (as opposed to the theorem) still applies.

**Q3. Should the proof approximate the state equation instead of the functional?**
i.e. approximate the reservoir map $F^*(x,u)$ on a compact finite-dimensional domain, rather than approximating the quantile functionals $\Phi_j$ via filter universality (current Lemma .5). What does this buy and what does it cost?

**Q4. Randomly drawn $(A,C,\zeta)$: RC12 IS the tool. Remark .15 must be rewritten.**

> **KEY POINT. RC12 (Gonon-Grigoryeva-Ortega, AAP 33(1), 2023) makes random reservoirs usable. The current Remark .15 ("random reservoirs are not covered") is too pessimistic and needs replacing.** Random draw of $(A,C,\zeta)$, only the readout $W$ trained, which is how ESNs are actually used and what the implementation chapter already does.

Two distinct routes, keep them separate:

- **Theorem C, directly via RC12 Thm 4.** Its hypothesis is that $F^*(\cdot,v)$ is an $r$-contraction *for the target's own reservoir map*, not for the ESN. Under (C1) ($\sum_i \ell_i < 1$) the quantile companion map $F^*$ is exactly such a contraction. So Thm 4 applies as stated and gives $O(N^{-1/2})$ plus a burn-in term $r^{T+1}$, with only $W$ trained. This is the quantitative theorem the thesis currently lacks.
- **Theorem A, via RC12's static machinery.** Once Q3 is adopted and the object to approximate is the state map $F^*$ on the compact finite-dimensional domain $X'^k\times[0,1]$, RC12 Prop. 2 (integral representation for ReLU) + Thm 1 apply. The endogenous-input-law objection in Remark .15 dissolves outright, since the domain is fixed independently of $W$.

Remaining caveats, do not lose these:

- RC12 Thm 1 is in $L^2(\mathcal X,\mu_Z)$. The sup-norm version needed by Step 3 is **RC12 Remark 5**, which points to Gonon (2021), *Random feature neural networks learn Black-Scholes type PDEs without curse of dimensionality*, Thm 1, valid for $\mathcal X$ finite-dimensional under condition (i) of RC12 Thm 1. Check that theorem's hypotheses directly.
- Requires a Fourier/Barron-type condition (RC12 Assumption 2) on $q^*$. This is a genuine assumption on the target density, not free.
- The conclusion becomes: with probability $\ge 1-\delta$ over the draw, there exists $W$ with $W_1^\rho \le \varepsilon$. The draw is independent of $u$, so Step 3 runs unchanged conditionally on the good event and the bound stays sure in $u$.

**Q5. Is it legitimate to use GO2018 for a finite horizon?**
The theorem is stated for left-infinite inputs $z \in K_M \subset (\mathbb{R}^n)^{\mathbb{Z}_-}$. Probably minor, but the padding / initial-state convention needs to be made explicit. [depends on Q3]

---

## Theorem C

**Q6. Without a closed-loop ESP, is stochastic causality even well posed?**
If there is no deterministic filter, the output cannot be framed as a pushforward of the noise law. Rossmannek's work (RC28, and *Stochastic dynamics learning with state-space systems*) shows the stochastic ESP can hold without the deterministic one, with causality imposed at the level of measures. Which notion should Theorem C be stated in? [depends on open/closed loop]

**Q7. Can Theorem C be stated without a burn-in?**
Burn-in should be an implementation artifact, not part of the theory. Is there a formulation where the ESN's own semi-infinite solution is compared to the target's, with the burn-in relegated to a corollary? [depends on open/closed loop]

---

## Literature and positioning

**Q8. RC20 and RC12: what is usable?**
Full theorem inventory of both, then filter for what bears on A.

**Q9. Backhoff-Bartl-Beiglbock-Wiesel, *Estimating processes in adapted Wasserstein distance*: any transferable technique?**
Also raises a question of its own: is $AW$ (bi-causal couplings) the right metric rather than $W_1$, given that $W_1$ ignores conditional structure? The shared-noise coupling is causal but not obviously anti-causal, because the ESN's hidden state is not recoverable from its own past outputs.

**Q10. Priority scan.**
Bartl's own record, plus anything on Theorem B or the others. Specifically: RC28's conclusion announces this exact question as "ongoing work by the authors" (Ortega and Rossmannek). Clarify the division of labour before writing.

---

---
---

# ANSWERS AND IMPROVEMENTS

Resolved items. Numbered A1, A2, ... independently of the Q numbering.

---

## A1. Lemma .5 is replaced. New architecture, new lemma, proof.

*Supersedes Section 0. Answers Q1, Q2, Q3, Q5.*

### A1.1 Architecture

Feed the whole window in as input, and assemble it from past outputs rather than from the reservoir's internal memory:

$$h_j=\sigma\big(B\,\hat y_{j-1}+c\,u_j+\zeta\big),\qquad \hat x_j=W_{\min(j,k+1)}h_j,\qquad \hat y_j:=(\hat x_j,\dots,\hat x_{j-k+1}),$$

with $\hat y_0:=(x^*,\dots,x^*)$, $B\in M_{N_r,k}$, $c,\zeta\in\mathbb R^{N_r}$, $W_i\in\mathbb R^{1\times N_r}$.

Two things to state explicitly rather than leave implicit:

- **There is no recurrent term $Ah_{j-1}$.** The model is a feedforward network applied to (window, noise) and iterated. This is RC12 eq. (77) plus an explicit tapped delay line on the scalar output. Cite RC12 §6's own sentence for the pedigree: these are *"also referred to as Jordan recurrent neural networks (with random internal weights)"* and *"a popular modification of the echo state networks considered in Section 5"*. That sentence is the difference between "I dropped the reservoir" and "I am using the architecture RC12 §6 is written for".
- **Do not feed back $\hat y_{j-1}$ as a $k$-dimensional readout**, i.e. RC12 (77) literally. Then the shift coordinates are only approximated, each picks up $\delta$ per step, and Step 3 has to be redone as a companion-matrix vector recursion. Assembling $\hat y_j$ from stored past outputs makes the shift **exact** and leaves Step 3 verbatim.

### A1.2 New Lemma

**Lemma .5′ (Simultaneous uniform approximation).** Let $\sigma:\mathbb R\to\mathbb R$ be continuous, bounded and non-constant, and let $\Phi_1,\dots,\Phi_m:[0,1]\times X'^k\to\mathbb R$ be continuous. Then for every $\delta>0$ there exist $N_r\in\mathbb N$, $B\in M_{N_r,k}$, $c,\zeta\in\mathbb R^{N_r}$ and $W_1,\dots,W_m\in\mathbb R^{1\times N_r}$ such that

$$\max_{1\le i\le m}\ \sup_{(u,y)\in[0,1]\times X'^k}\big|\,W_i\,\sigma(By+cu+\zeta)-\Phi_i(u;y)\,\big|<\delta .$$

### A1.3 Proof

**(1)** $K:=[0,1]\times X'^k\subset\mathbb R^{k+1}$ is a product of compact intervals, hence compact, and each $\Phi_i\in C(K)$.

**(2)** Since $\sigma$ is continuous, bounded and non-constant, Hornik (1991, Theorem 2) gives that single-hidden-layer networks with activation $\sigma$ are dense in $C(K)$ for the uniform norm. Applying this to each $\Phi_i$ yields $N_i\in\mathbb N$, $V_i\in M_{N_i,k+1}$, $\zeta_i\in\mathbb R^{N_i}$, $w_i\in\mathbb R^{1\times N_i}$ with

$$\sup_{(u,y)\in K}\big|w_i\,\sigma\big(V_i(u,y)^\top+\zeta_i\big)-\Phi_i(u;y)\big|<\delta .$$

Write $V_i=[\,c_i\mid B_i\,]$ with $c_i\in\mathbb R^{N_i}$ acting on $u$ and $B_i\in M_{N_i,k}$ acting on $y$.

**(3)** Concatenate the hidden layers: $N_r:=\sum_i N_i$, $B:=[B_1;\cdots;B_m]$, $c:=(c_1,\dots,c_m)$, $\zeta:=(\zeta_1,\dots,\zeta_m)$, and $W_i:=(0\ \cdots\ w_i\ \cdots\ 0)\in\mathbb R^{1\times N_r}$ supported on the $i$-th block. Because $\sigma$ acts componentwise, $\sigma(By+cu+\zeta)$ is the concatenation of the vectors $\sigma(B_iy+c_iu+\zeta_i)$, so $W_i\sigma(By+cu+\zeta)=w_i\sigma(B_iy+c_iu+\zeta_i)$ and the bound of (2) holds for every $i$ simultaneously. $\square$

### A1.4 How it is applied

Apply with $m=k+1$ and

$$\Phi_i(u;y):=q^{(i)}(u;y_1,\dots,y_{i-1})\ (1\le i\le k),\qquad \Phi_{k+1}(u;y):=q(u;y).$$

All are continuous on $[0,1]\times X'^k$ by Lemma .4(2) after the McShane extension. Fix a convention for the coordinates the boundary maps ignore, e.g. set them to $x^*$.

### A1.5 What this deletes from the draft

- Preliminaries **Theorem .1** (GO2018 Thm 4.1) and **Corollary .2** (GO2018 Cor 3.2). Used nowhere else.
- The hypothesis $\|A\|_2L_\sigma<1$ throughout. The old concatenation was a direct sum of *reservoir systems*, so the ESP had to be preserved; the new one concatenates feedforward hidden units, where nothing has to be preserved.
- **Assumption 3** weakens from "differentiable squashing with $L_\sigma=\sup|\sigma'|<\infty$" to "continuous, bounded, non-constant". tanh and logistic qualify. ReLU does not; for it, cite Leshno-Lin-Pinkus-Schocken (1993) instead.
- **Step 1**'s fading-memory-functional discussion, replaced by "each $\Phi_i$ is continuous on a compact set" (Lemma .4(2)).
- **Footnote 3** collapses to one sentence: the supremum in Lemma .5′ is over $X'^k$, and the induction in Step 3 keeps $\hat y_{j-1}$ there. No open-loop versus closed-loop distinction is needed, because no filter appears anywhere in the argument.

**Steps 3 and 4 are unchanged, character for character, including $C_N$.**

### A1.6 References to add

- Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks* **4**, 251-257. **Theorem 2.**
- Gonon, L., Grigoryeva, L., Ortega, J.-P. (2023). Approximation bounds for random neural networks and reservoir systems. *Ann. Appl. Probab.* **33**(1), 28-69. **Section 6, eq. (77)**, for the architecture.
- Optional, if the activation class is to be left open: Leshno, M., Lin, V. Ya., Pinkus, A., Schocken, S. (1993). *Neural Networks* **6**, 861-867.

---

## A2. Do not assume the contraction in Theorem A. Add a corollary instead.

*Answers the follow-up to Q4.*

Assuming $\sum_i\ell_i<1$ in A does not merely restrict it, it makes it **redundant**. Remark .10 already computes that under contraction $C_N\le(1-Lk)^{-1}$ uniformly in $N$, so the finite-horizon restriction stops doing any work and A becomes C restricted to finite $N$. The class lost is exactly the non-contractive $k$-Markov targets, which is where the interesting persistence lives.

Two further reasons:

- RC12 Assumption 2 has two parts and the contraction is the easy one. **Definition 2's Fourier condition (eq. 80, $C_f<\infty$) is the hard one** to verify for $q^*$ built from BKM. Assuming contraction alone does not unlock Theorem 3.
- The condition is currently stated too crudely. Lemma .4 gives $L\le L_F/c$, an *upper bound* on the true per-lag Lipschitz constants, and it can be far from sharp: for a truncated Gaussian AR(1), $q(u;x)=\phi x+\sigma\Phi^{-1}(u)$ has $\ell_1=|\phi|$, so the sharp condition is $|\phi|<1$ while $L_F/c$ is much larger. **State the contraction hypothesis in terms of the actual $\ell_i$, or the spectral radius of the companion matrix with first row $(\ell_1,\dots,\ell_k)$, not as $Lk<1$ with $L=L_F/c$.** Strictly weaker, and worth doing wherever the hypothesis ends up.

**Do this instead.** Keep Theorem A contraction-free; add:

> **Corollary A′.** Under $\sum_i\ell_i<1$: (i) $C_N\le(1-r)^{-1}$ uniformly in $N$, where $r$ is the companion spectral radius realized in a suitable weighted norm; (ii) subject to RC12 Definition 2's smoothness condition on $q$, RC12 Theorem 3 yields $O(N_r^{-1/2})$ with randomly drawn $(B,c,\zeta)$ and only the readouts trained.

This buys RC12's benefits without paying for them in the main statement, and is the honest bridge to Theorem C.

---
---

## Standing flags

- **RC12 unlocks random $(A,C,\zeta)$ (see Q4).** Remark .15 is to be rewritten, not defended. The obstruction was never randomness; it was the choice of norm, and that is fixable.
- Sup-norm vs $L^p$: Step 3 needs a sure, uniform bound at the realized history. Any tool that only gives an average over a fixed input law is unusable there. This is the one live constraint on the random-reservoir route.
- $C_N$ vs $1/(1-r)$: the finite-horizon accumulation constant collapses to a horizon-free one exactly when $\sum_i \ell_i < 1$. That condition is the bridge from A to C.
- Theorem B's priority context is classical ergodic theory (Parthasarathy, Sigmund, entropy density), not the RC literature.
