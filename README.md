# DSP Assignment – Audio Processing Toolkit

This repository contains my complete solutions for four signal-processing problems involving **frequency translation & mixing**, **DFT/IDFT from first principles**, **channel equalization**, and **single-channel speech de-noising**.
Everything is implemented in Python with NumPy/SciPy/Matplotlib, and designed to be **reproducible, well-commented, and listenable** (with WAV I/O and plots).

## Table of Contents

* [Environment & Setup](#environment--setup)
* [Input Files](#input-files)
* [Repository Layout](#repository-layout)
* [How to Run](#how-to-run)
* [Problem 1 — Frequency Translation & Selective Mixing](#problem-1--frequency-translation--selective-mixing)
* [Problem 2 — DFT/IDFT, Validation, and Sampling Experiments](#problem-2--dftidft-validation-and-sampling-experiments)
* [Problem 3 — Channel Equalization (LTI System Inversion)](#problem-3--channel-equalization-lti-system-inversion)
* [Problem 4 — Signal De-Noising (STFT Noise Tracking)](#problem-4--signal-de-noising-stft-noise-tracking)
* [Reproducibility Notes](#reproducibility-notes)
* [FAQ](#faq)
* [License](#license)

---

## Environment & Setup

### Python

* Python **3.9+** (tested with 3.10/3.11)

### Minimal dependencies

```bash
pip install numpy scipy matplotlib soundfile ipython
```

If you’d like to view tables or re-run the denoising metrics, also:

```bash
pip install pandas
```

> `soundfile` is preferred for WAV I/O; if it’s not available, the code falls back to `scipy.io.wavfile`.

### Optional (for interactive exploration)

* JupyterLab / Notebook or VS Code with a Python notebook extension.

---

## Input Files

Place the following WAV files in the project root (same folder as the notebooks/scripts):

* **Problem 1**: `music1.wav`, `music2.wav`
  *(If these are missing, the code generates realistic demo signals so the pipeline still runs.)*
* **Problem 3**: `clean1.wav`, `distorted1.wav`, `distorted2.wav`
* **Problem 4**: `noisy1.wav`

**All WAVs** are converted to **mono, float32** internally. Sample-rate mismatches are handled with rational resampling so everything runs on a common grid.

---

## Repository Layout

*If you’re reading this inside the actual repo, filenames may differ slightly; the code is self-contained and can live either in notebooks or in `.py` scripts. The README describes exactly what each part does and the files it writes.*

Recommended structure:

```
.
├── README.md
├── docs/
│   ├── problem1.png
│   ├── problem2.png
│   ├── problem3.png
│   └── problem4.png
├── music1.wav            # (Problem 1)
├── music2.wav            # (Problem 1)
├── clean1.wav            # (Problem 3)
├── distorted1.wav        # (Problem 3)
├── distorted2.wav        # (Problem 3)
├── noisy1.wav            # (Problem 4)
├── notebooks/            # (optional)
│   ├── Problem_1.ipynb
│   ├── Problem_2.ipynb
│   ├── Problem_3.ipynb
│   └── Problem_4.ipynb
└── src/                  # (optional if you prefer .py)
    ├── problem_1.py
    ├── problem_2.py
    ├── problem_3.py
    └── problem_4.py
```

> You can keep everything in a single notebook as well; the code blocks in the sections below are stand-alone.

---

## How to Run

### Notebooks (recommended)

1. `jupyter lab` (or `jupyter notebook`)
2. Open the problem’s notebook and run all cells in order.

### Python scripts (if you prefer `.py`)

```bash
python src/problem_1.py
python src/problem_2.py
python src/problem_3.py
python src/problem_4.py
```

Each script writes its outputs (WAVs/plots) alongside the script or inside a small output folder (see per-problem notes).

> 🔊 **Volume caution:** Audio files are peak-normalized for listening widgets, but always start playback at a moderate volume.

---

## Problem 1 — Frequency Translation & Selective Mixing

**Goal.** Combine two audio signals into a single file such that, by shifting frequency bands, you can **listen selectively** to either source later.

**Key idea.** Create an **analytic signal** via the Hilbert transform and multiply by a complex exponential to **shift** one signal’s spectrum out of baseband:

$$
y(t) = \Re\{\,x_a(t)\,e^{j2\pi f_\text{shift} t}\,\}
$$

This avoids negative-frequency mirroring and yields a clean translation by $f_\text{shift}$.

### What I implemented

* Robust WAV I/O (`soundfile` with SciPy fallback), mono conversion, safe resampling to a **single** sampling rate (default 44.1 kHz).
* Helper utilities for **normalization**, **same-length trimming**, and **waveform/spectrum** plotting.
* `analytic_frequency_shift(x, sr, shift_hz)` using `scipy.signal.hilbert`.
* Optional **FIR low-pass** to isolate baseband after inverse shifting.

### Pipeline

1. **Load** `music1.wav` and `music2.wav`. If missing, generate demo “speech-like” and “music-like” signals.
2. **Shift up** `music1` by e.g. **+18.5 kHz** (safe given Nyquist and the content bandwidth).
3. **Mix** `mix = normalize(y1_shifted + y2)` → largely sounds like `music2` alone (shifted `music1` sits near 18.5 kHz).
4. **Shift the mixture back down** by **−18.5 kHz** → `music1` comes back to baseband while `music2` is moved up; the ear now hears **music1**.
5. (Optional) **Low-pass** to further suppress the shifted copy of `music2`.

### Files written

* `part_a_music1_shifted.wav`
* `part_b_mix.wav`
* `part_c_mix_shifted_back.wav`
* `part_c_recovered_m1_lowpass.wav`

### Why the listening trick works (linearity)

* In frequency: $\mathcal{F}\{\text{mix}\} = Y_2(f) + Y_1(f-f_s)$
* Shift by $-f_s$: $\mathcal{F}\{\text{mix}_\downarrow\} = Y_2(f+f_s) + Y_1(f)$
  → `music1` returns to baseband; `music2` is pushed high.

> **Aliasing note:** Ensure $f_\text{shift} + f_\text{max} < f_\mathrm{Nyq}$.
> For broader `music1`, reduce the shift (e.g., 12–16 kHz).

---

## Problem 2 — DFT/IDFT, Validation, and Sampling Experiments

**Goal.** Move from continuous-time FT to a practical **DFT**; derive **IDFT**, implement both transforms in $O(N^2)$, validate against NumPy, and study sampling effects using $x(t)=\cos(\pi t)$.

### Definitions (conventions used)

$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-j 2\pi kn/N}, \qquad
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{+j 2\pi kn/N}.
$$

All $1/N$ scaling is in the **IDFT** (matches `numpy.fft` default).

### Implementations

* `fft(x)` — vectorized $O(N^2)$ DFT via an explicit DFT matrix.
* `ifft(X)` — vectorized $O(N^2)$ IDFT via the conjugate DFT matrix divided by $N$.

Both functions are compact, loop-free (NumPy broadcasting), and **numerically match** `np.fft.fft/ifft` to machine precision.

### Validations & Demos

* **Numerical checks:** for multiple $N$, `ifft(fft(x)) ≈ x` and `fft(x)` ≈ `np.fft.fft(x)` (max abs error \~1e-13…1e-15).
* **Bin-aligned demo:** mixture of tones exactly landing on bins → **no spectral leakage**; reconstruction error \~2e-14.
* **Sampling $x(t)=\cos(\pi t)$:** with $T=\tfrac12$, $F_s=2$ Hz → $x[n]=\cos(\frac{\pi}{2}n)$; DFT lines at $\omega=\pm \pi/2$ (or $f=\pm 0.5$ Hz) match the CTFT impulses.
* **Changing $N$:** locations fixed; **resolution** improves as $\Delta f = F_s/N$ shrinks; peak heights scale with $N/2$ for a unit cosine.
* **Changing $T$:** $\omega_0=\pi T$ modulo $2\pi$; shows **Nyquist** collapse at $T=1$ and **aliasing** for $T>1$.
* **Leakage demo:** choosing $N$ not multiple of the signal period + windowing (Hann vs. rectangular) illustrates main-lobe width vs. sidelobes.

---

## Problem 3 — Channel Equalization (LTI System Inversion)

**Goal.** Given:

* `clean1.wav` (**system input**)
* `distorted1.wav` (**system output** of the same content)
* `distorted2.wav` (another output with **unknown** original input)

Estimate the **channel $H(f)$** and build a **regularized inverse** $G(f)$ to recover a clean version of `distorted2`.

### Workflow

1. **Pre-process**: mono, resample, peak-normalize (for plotting), and **time-align** `clean1`/`distorted1` via **FFT cross-correlation**.

2. **Estimate channel** (LS in the STFT domain):

   $$
   H(f) = \frac{\sum_t Y(f,t)\,X^*(f,t)}{\sum_t |X(f,t)|^2 + \varepsilon}
   $$

3. **Build inverse** with **Tikhonov regularization** and a **gain cap** (e.g., +18 dB):

   $$
   G(f) = \frac{H^*(f)}{|H(f)|^2 + \lambda}
   $$

4. **Recover** `distorted2` by applying $G$ frame-wise in STFT and ISTFT back to time.

5. **Validate** with waveform overlays, absolute spectra, and a **ratio plot**
   $20\log_{10}(|\mathrm{Rec2}|/|\mathrm{Dist2}|)$ vs. $20\log_{10}|G(f)|$.

### Files written

* `recovered2.wav` (peak-normalized for listening)
* `recovered2_without_normalization.wav` (raw level, float if possible)

### Notes

* STFT: Hann, $N_\text{FFT}=4096$, **75% overlap**.
* Regularization $\lambda$ and **gain cap** prevent instability/noise blow-up in channel notches.
* The **measured ratio** tracks the expected gain $|G|$ in passbands and backs off in deep notches (by design).

---

## Problem 4 — Signal De-Noising (STFT Noise Tracking)

**Goal.** Recover a cleaner speech waveform from `noisy1.wav` using a **time–frequency suppression mask** with **noise PSD tracking** (noise-only tail + minimum statistics), **SNR-adaptive oversubtraction**, and **temporal/frequency smoothing**.

### Design (high-level)

1. **Pre-emphasis** ($1 - \alpha z^{-1}$, $\alpha=0.97$).
2. **STFT** (Hann, $N_\text{FFT}\in\{2048,4096\}$, hop $=$ $N_\text{FFT}/4$).
3. **Noise PSD** $N_\text{est}(f,t)$:
   **Tail median** $N_0(f)$ + **minimum statistics** over \~1 s; use $\max(N_\text{ms}, N_0)$.
4. **A-posteriori SNR** $\gamma = P_{yy}/N_\text{est}$; map SNR(dB) → **oversubtraction** $\alpha(f,t)$.
   Power-domain subtraction with amplitude floor $\beta$ (to avoid musical noise):

   $$
   G_p = \max\{\beta^2,\ 1 - \alpha\,N_\text{est}/P_{yy}\},\qquad G=\sqrt{G_p}
   $$

   Plus an **aggressive** variant: $\alpha \ge \alpha_\text{tail}$.
5. **Stability smoothing**: temporal IIR + small frequency moving average.
6. **iSTFT + De-emphasis** → denoised waveform.
7. **Metrics & visualization**: tail RMS reduction, low-/high-energy frame behavior, spectral flatness, waveforms/spectrograms, mean gain vs. frequency.

### Parameters (typical)

* Tail start: **9.5 s**
* $N_\text{FFT}=$ 2048 (≤32 kHz) or 4096 (>32 kHz), hop $= N_\text{FFT}/4$
* Pre-emphasis $\alpha=0.97$
* Amplitude floor $\beta\approx 0.06$ (≈ −24.4 dB)
* Temporal smoothing $0.6$, frequency smoothing **7 bins**
* Minimum-statistics window ≈ **1 s**
* Aggressive oversubtraction floor $\alpha_\text{tail}=4.0$

### Files written

* `denoise_outputs/denoise1.wav` (moderate)
* `denoise_outputs/denoise1_listen.wav` (moderate, peak-normalized for playback)
* `denoise_outputs/denoise1_all_aggressive.wav` (aggressive)
* `denoise_outputs/denoise1_all_aggressive_listen.wav` (aggressive, normalized)
* `denoise_outputs/denoise_metrics.csv` (summary table)

### Objective results (example run)

| Metric                                |   Moderate | Aggressive |  Original |
| ------------------------------------- | ---------: | ---------: | --------: |
| Tail RMS improvement (dB)             | **+19.78** | **+19.83** |         — |
| Attenuation on low-energy frames (dB) | **−20.69** | **−21.07** |         — |
| Mean gain on high-energy frames (dB)  |  **−6.86** |  **−8.51** |         — |
| Spectral flatness (median)            |  **0.106** |  **0.077** | **0.273** |

**Interpretation**

* Both variants remove ≈ **20 dB** of noise in the **tail/noise-only** segment.
* Aggressive setting gives **slightly** more attenuation overall, but **reduces** high-band speech energy more than moderate.
* Spectrograms show a much cleaner background; formants and harmonics remain intact, especially with the moderate mask.

### Baselines (appendix, also implemented)

* **Notch bank + HPF** (time-domain) → removes hum/rumble only.
* **Decision-Directed Wiener** (STFT) → conservative, low artifacts, more residual hiss.
* **Log-MMSE (LSA)** → smoother but still leaves more high-freq noise than the oversubtractive mask here.

---

## Reproducibility Notes

* **Determinism**: Any randomness is limited to synthetic demo generation (Problem 1 fallback). A fixed RNG seed is used.
* **Numerical stability**: All divisions guarded with small epsilons; normalization prevents clipping; regularization and gain caps control ill-conditioned inverses.
* **STFT consistency**: Hann + 75% overlap satisfies COLA for artifact-free iSTFT.

---

## FAQ

**Q: I don’t have `music1.wav`/`music2.wav`. Can I still run Problem 1?**
A: Yes. The code auto-generates demo signals (speech-like band-limited noise and a sum of tones) so you can reproduce the mixing/translation behavior.

**Q: Why shift by \~18.5 kHz in Problem 1?**
A: It cleanly moves one track out of the audible baseband at 44.1 kHz sampling without breaching Nyquist, so the mixture largely sounds like the unshifted track. Shift amount can be reduced if the source has wider bandwidth.

**Q: Your DFT grows with $N$; is that a bug?**
A: No—my convention puts the **$1/N$** factor entirely in the **IDFT**, so a unit-amplitude cosine places \~$N/2$ in each of the two bins. This matches NumPy’s default scaling.

**Q: Why regularize $1/H(f)$ in Problem 3?**
A: Real channels have notches. A naïve inverse would blow up noise. Tikhonov regularization + a gain cap produces a safe, realizable equalizer.

**Q: What causes “musical noise” in denoisers (Problem 4)?**
A: Hard, unsmoothed masks with deep nulls. I use time/frequency smoothing and a gain floor to avoid isolated TF bins with extreme suppression.

---

## License

This repository is released under the **MIT License**. See `LICENSE` if present; otherwise you may reuse the code with attribution.

---

## Acknowledgements

* NumPy/SciPy/Matplotlib teams for foundational scientific Python tools.
* `soundfile` for robust float WAV I/O with graceful fallback to SciPy.
* Classic references behind spectral subtraction, min-statistics noise tracking, DD-Wiener, and Log-MMSE denoising (implemented here in simplified, assignment-friendly form).

---

### Quick Reference – Produced Audio Files

* **Problem 1**

  * `part_a_music1_shifted.wav`
  * `part_b_mix.wav`
  * `part_c_mix_shifted_back.wav`
  * `part_c_recovered_m1_lowpass.wav`
* **Problem 3**

  * `recovered2.wav`
  * `recovered2_without_normalization.wav`
* **Problem 4**

  * `denoise_outputs/denoise1.wav`
  * `denoise_outputs/denoise1_listen.wav`
  * `denoise_outputs/denoise1_all_aggressive.wav`
  * `denoise_outputs/denoise1_all_aggressive_listen.wav`
  * `denoise_outputs/denoise_metrics.csv`

> If you add the four problem images to `docs/`, link them in this README to mirror the assignment statements visually.
