"""
HMM perplexity convergence figure.

A 3-state HMM with transition-tied emissions over alphabet {A, B, C}.
Generate 40 sequences of length 20 from the HMM. Iteratively grow a GDC's
training set one sequence at a time. After each sequence, compute the
ratio of the GDC's per-step predictive perplexity to the true HMM's
per-step predictive perplexity (geometric-mean form, base e).

Layout:
    +---------------------+----------------------------+
    |                     |  Perplexity ratio          |
    |  HMM diagram        |                            |
    |                     +----------------------------+
    |                     |  5-step prefix + GDC       |
    |                     |  5-step forecast (heatmap) |
    +---------------------+----------------------------+
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyArrowPatch, Arc
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)            # paper_figures/
ROOT = os.path.dirname(PARENT)            # repo root
sys.path.insert(0, PARENT)
sys.path.insert(0, ROOT)

from style import COLORS, apply_style  # noqa: E402
from generative_dense_chain import GenerativeDenseChain  # noqa: E402


# ---- HMM definition --------------------------------------------------------

# joint[i, j, l] = P(transit i -> j, emit l | state i)
# States: S1=0, S2=1, S3=2
# Emissions: A=0, B=1, C=2
JOINT = np.array([
    [[0.770, 0.070, 0.070],   # S1 -> S1
     [0.003, 0.040, 0.003],   # S1 -> S2
     [0.003, 0.003, 0.040]],  # S1 -> S3
    [[0.040, 0.003, 0.003],   # S2 -> S1
     [0.070, 0.770, 0.070],   # S2 -> S2
     [0.003, 0.003, 0.040]],  # S2 -> S3
    [[0.040, 0.003, 0.003],   # S3 -> S1
     [0.003, 0.040, 0.003],   # S3 -> S2
     [0.070, 0.070, 0.770]],  # S3 -> S3
])
# Each row sums to 1.002 due to listed precision; renormalize to be exact.
JOINT = JOINT / JOINT.sum(axis=(1, 2), keepdims=True)

INIT_DIST = np.ones(3) / 3
N_EMIT = 3
EMIT_LABELS = ['A', 'B', 'C']
STATE_LABELS = [r'$S_1$', r'$S_2$', r'$S_3$']


# ---- Configuration ---------------------------------------------------------

N_SEQ = 400
SEQ_LEN = 40
N_RUNS = 50
GDC_ALPHA = 0.9
GDC_BETA = 0.0
GDC_TRANSITION = 'sequential'
SEED = 42


# ---- HMM utilities ---------------------------------------------------------

def sample_sequence(rng: np.random.Generator, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample one sequence of length T from the HMM. Returns (states, emissions)."""
    n_states = JOINT.shape[0]
    state = int(rng.choice(n_states, p=INIT_DIST))
    states = np.empty(T, dtype=int)
    emits = np.empty(T, dtype=int)
    for t in range(T):
        flat = JOINT[state].flatten()
        flat = flat / flat.sum()
        idx = int(rng.choice(len(flat), p=flat))
        next_state, emit = idx // N_EMIT, idx % N_EMIT
        states[t] = next_state
        emits[t] = emit
        state = next_state
    return states, emits


def hmm_log_predictive(observations: np.ndarray) -> np.ndarray:
    """Returns log P(o_t | o_0..o_{t-1}) at each step under the true HMM."""
    T = len(observations)
    log_p = np.zeros(T)
    pi = INIT_DIST.copy()
    for t in range(T):
        # Predictive over emissions: sum_{i,j} pi(i) * joint[i,j,l]
        p_ot = np.einsum('i,ijl->l', pi, JOINT)
        log_p[t] = np.log(p_ot[observations[t]])
        # Filter update
        new_pi = np.einsum('i,ij->j', pi, JOINT[:, :, observations[t]])
        Z = new_pi.sum()
        pi = new_pi / Z if Z > 0 else INIT_DIST.copy()
    return log_p


def hmm_forecast_distribution(prefix: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Posterior predictive distribution over the next n_steps emissions under
    the true HMM, given a prefix of observations. Returns shape (n_steps, N_EMIT).
    """
    pi = INIT_DIST.copy()
    for o in prefix:
        new_pi = np.einsum('i,ij->j', pi, JOINT[:, :, int(o)])
        Z = new_pi.sum()
        pi = new_pi / Z if Z > 0 else INIT_DIST.copy()
    transition_marg = JOINT.sum(axis=2)  # P(state_{t+1} | state_t)
    out = np.zeros((n_steps, N_EMIT))
    for k in range(n_steps):
        out[k] = np.einsum('i,ijl->l', pi, JOINT)
        pi = pi @ transition_marg
    return out


# ---- GDC predictive --------------------------------------------------------

def gdc_log_predictive(gdc: GenerativeDenseChain, observations: np.ndarray) -> np.ndarray:
    """Returns log P(o_t | o_0..o_{t-1}) at each step under the GDC.

    Uses the GDC's own noisy emission model:
      P(obs=l | s) = (1 - beta) * 1[s.value == l] + beta/V
    so the predictive over emissions is
      P(obs=l | pi) = (1 - beta) * sum_{s: s.value=l} pi(s) + beta/V * sum(pi)
    which is a proper distribution with a beta/V floor — preventing the
    pathological log-prob explosions of an arbitrary epsilon floor when
    the training set is small.
    """
    T = len(observations)
    log_p = np.zeros(T)
    pi = gdc._get_initial_distribution()
    states_flat = gdc.states.flatten().astype(int)
    beta = gdc.beta
    V = N_EMIT
    # Effective smoothing constant. With beta=0 the predictive is exactly
    # the marginal of pi over states-by-emission, which can be zero;
    # apply a tiny floor to keep log-prob finite in that limiting case.
    floor = max(beta / V, 1e-6)

    for t in range(T):
        if t > 0:
            pi = gdc._transition(pi)
        total_mass = float(pi.sum())
        # bincount aggregates pi by emission value in a single pass.
        raw_p = np.bincount(states_flat, weights=pi, minlength=V)[:V]
        if total_mass > 0:
            p_ot = (1 - beta) * raw_p + beta / V * total_mass
        else:
            p_ot = np.ones(V) / V
        p_ot = np.clip(p_ot, floor, None)
        p_ot = p_ot / p_ot.sum()
        log_p[t] = np.log(p_ot[observations[t]])

        # Bayes update with the actual emission, matching forward_pass.
        if beta == 0:
            mask = (states_flat == observations[t])
            new_pi = np.zeros_like(pi)
            new_pi[mask] = pi[mask]
            Z = new_pi.sum()
            pi = new_pi / Z if Z > 0 else np.ones_like(pi) / len(pi)
        else:
            mask = (states_flat == observations[t])
            beta_over_V = beta / V
            new_pi = pi * beta_over_V
            new_pi[mask] += pi[mask] * (1 - beta)
            Z = new_pi.sum()
            pi = new_pi / Z if Z > 0 else np.ones_like(pi) / len(pi)

    return log_p


def gdc_forecast_distribution(gdc: GenerativeDenseChain,
                              prefix: np.ndarray, n_steps: int) -> np.ndarray:
    """
    After conditioning on `prefix`, return the GDC's predictive distribution
    over the next `n_steps` emissions (no observations after the prefix).
    Returns array shape (n_steps, N_EMIT).
    """
    pi = gdc.forward_pass(prefix.reshape(-1, 1))
    states_flat = gdc.states.flatten().astype(int)
    out = np.zeros((n_steps, N_EMIT))
    for k in range(n_steps):
        pi = gdc._transition(pi)
        out[k] = np.bincount(states_flat, weights=pi, minlength=N_EMIT)[:N_EMIT]
        s = out[k].sum()
        if s > 0:
            out[k] /= s
    return out


# ---- Diagram drawing -------------------------------------------------------

def draw_hmm_diagram(ax) -> None:
    ax.set_xlim(-0.10, 1.10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')

    # Triangle layout — pulled inward so self-loop labels stay inside the panel.
    pos = np.array([
        [0.50, 0.76],   # S1 top
        [0.22, 0.22],   # S2 bottom-left
        [0.78, 0.22],   # S3 bottom-right
    ])
    radius = 0.085
    r_loop = 0.05

    # ---- Cross arrows (drawn first so circles cover the endpoints) ----
    # Use the SAME rad sign for every cross arrow. Because arc3 computes the
    # control point as midpoint + rad * perp_ccw(chord_dir) * |chord|, and the
    # chord direction flips for opposite-direction arrows, opposing arrows
    # end up curving on opposite sides of the chord line in absolute coords.
    rad_const = 0.22
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            from_pos = pos[i]
            to_pos = pos[j]
            direction = to_pos - from_pos
            length = float(np.linalg.norm(direction))
            unit = direction / length
            start = from_pos + unit * radius
            end = to_pos - unit * radius
            arrow = FancyArrowPatch(
                start, end,
                connectionstyle=f'arc3,rad={rad_const}',
                arrowstyle='-|>,head_width=3.5,head_length=5',
                color=COLORS['muted'], lw=0.7, zorder=1,
                shrinkA=0, shrinkB=0,
            )
            ax.add_patch(arrow)

            # Position each label at t=0.65 along the chord (closer to the
            # destination state) plus a small perpendicular offset on the
            # curve-bulge side. This way opposing arrows' labels separate
            # both along the chord and across it. All three emissions are
            # listed; .2g rounding hides the tiny numerical drift introduced
            # by row normalization.
            em = JOINT[i, j]
            label = '\n'.join(f'{c}:{v:.2g}' for c, v in zip(EMIT_LABELS, em))
            t_label = 0.65
            chord_pt = (1 - t_label) * start + t_label * end
            perp_ccw = np.array([-direction[1], direction[0]]) / length
            offset = perp_ccw * (rad_const * length * 0.6)
            label_pos = chord_pt + offset
            ax.text(label_pos[0], label_pos[1], label,
                    ha='center', va='center', fontsize=6.5,
                    color=COLORS['text'], zorder=4,
                    linespacing=1.05)

    # ---- Self loops (Arc patches + arrowhead) ----
    centroid = pos.mean(axis=0)
    for i in range(3):
        outward = pos[i] - centroid
        outward = outward / np.linalg.norm(outward)
        loop_center = pos[i] + outward * (radius + r_loop * 0.7)

        # Gap of the loop faces back toward the state circle.
        gap_angle = float(np.degrees(np.arctan2(-outward[1], -outward[0])))
        arc = Arc(
            loop_center, 2 * r_loop, 2 * r_loop, angle=0,
            theta1=gap_angle + 30, theta2=gap_angle + 330,
            color=COLORS['blue_dark'], linewidth=0.9, zorder=2,
        )
        ax.add_patch(arc)

        # Arrowhead at the end of the (CCW) arc.
        end_theta = np.radians(gap_angle + 330)
        end_pos = loop_center + r_loop * np.array(
            [np.cos(end_theta), np.sin(end_theta)]
        )
        tangent = np.array([-np.sin(end_theta), np.cos(end_theta)])
        head = FancyArrowPatch(
            end_pos - 0.018 * tangent, end_pos,
            arrowstyle='-|>,head_width=3.5,head_length=5',
            color=COLORS['blue_dark'], lw=0.9, zorder=2,
            shrinkA=0, shrinkB=0,
        )
        ax.add_patch(head)

        # Label position: above S1, directly under S2/S3 (centered on each
        # state's x-coordinate so they read as belonging to that state).
        em = JOINT[i, i]
        label = f'A:{em[0]:.2f}  B:{em[1]:.2f}  C:{em[2]:.2f}'
        if i == 0:
            label_pos = np.array([pos[i][0], pos[i][1] + radius + 2 * r_loop + 0.04])
        else:
            label_pos = np.array([pos[i][0], pos[i][1] - radius - 0.07])
        ax.text(label_pos[0], label_pos[1], label,
                ha='center', va='center', fontsize=7,
                color=COLORS['text'], zorder=4)

    # ---- Nodes ----
    for i in range(3):
        circle = Circle(pos[i], radius,
                        facecolor=COLORS['blue_light'],
                        edgecolor=COLORS['blue_dark'],
                        linewidth=0.9, zorder=3)
        ax.add_patch(circle)
        ax.text(pos[i][0], pos[i][1], STATE_LABELS[i],
                ha='center', va='center', fontsize=10,
                color=COLORS['text'], zorder=5)


# ---- Main ------------------------------------------------------------------

def main() -> None:
    apply_style()

    # Run the experiment N_RUNS times. Each run draws 40 fresh HMM sequences
    # and grows the GDC's training set one sequence at a time, recording the
    # GDC's and the true HMM's per-step predictive perplexity at each
    # training-set size.
    ppl_gdc_all = np.empty((N_RUNS, N_SEQ))
    ppl_true_all = np.empty((N_RUNS, N_SEQ))
    last_training: list[np.ndarray] = []

    for run in range(N_RUNS):
        rng = np.random.default_rng(SEED + run)
        sequences = [sample_sequence(rng, SEQ_LEN)[1] for _ in range(N_SEQ)]

        training: list[np.ndarray] = []
        for i, seq in enumerate(sequences):
            log_p_true = hmm_log_predictive(seq)
            ppl_true_all[run, i] = float(np.exp(-log_p_true.mean()))

            if not training:
                log_p_gdc = np.full(SEQ_LEN, np.log(1.0 / N_EMIT))
            else:
                seqs_2d = [s.reshape(-1, 1) for s in training]
                gdc = GenerativeDenseChain(
                    seqs_2d,
                    alpha=GDC_ALPHA,
                    beta=GDC_BETA,
                    transition_type=GDC_TRANSITION,
                    initial_dist='uniform',
                    terminal_behavior='absorb',
                )
                log_p_gdc = gdc_log_predictive(gdc, seq)
            ppl_gdc_all[run, i] = float(np.exp(-log_p_gdc.mean()))
            training.append(seq)
        last_training = training

    ratios = ppl_gdc_all / ppl_true_all  # shape (N_RUNS, N_SEQ)
    mean_ratio = ratios.mean(axis=0)
    se_ratio = ratios.std(axis=0, ddof=1) / np.sqrt(N_RUNS)

    # Forecast subplot: use the GDC trained on the final run's full training
    # set, with a fresh test sample so the prefix is independent of training.
    rng_test = np.random.default_rng(SEED + 10_000)
    _, test_emits = sample_sequence(rng_test, 10)
    prefix = test_emits[:5]

    seqs_2d = [s.reshape(-1, 1) for s in last_training]
    gdc_final = GenerativeDenseChain(
        seqs_2d,
        alpha=GDC_ALPHA,
        beta=GDC_BETA,
        transition_type=GDC_TRANSITION,
        initial_dist='uniform',
        terminal_behavior='absorb',
    )
    forecast_gdc = gdc_forecast_distribution(gdc_final, prefix, n_steps=5)
    forecast_hmm = hmm_forecast_distribution(prefix, n_steps=5)

    # ---- Figure -----------------------------------------------------------
    fig = plt.figure(figsize=(7.4, 4.6))
    gs = gridspec.GridSpec(
        nrows=3, ncols=2,
        width_ratios=[1.0, 1.15],
        height_ratios=[1.5, 1, 1],
        hspace=0.50, wspace=0.22,
        figure=fig,
        left=0.03, right=0.97, top=0.94, bottom=0.10,
    )
    ax_hmm = fig.add_subplot(gs[:, 0])
    ax_ratio = fig.add_subplot(gs[0, 1])
    ax_fc_gdc = fig.add_subplot(gs[1, 1])
    ax_fc_hmm = fig.add_subplot(gs[2, 1], sharex=ax_fc_gdc, sharey=ax_fc_gdc)

    draw_hmm_diagram(ax_hmm)

    # ---- Mean perplexity ratio plot ----
    x = np.arange(N_SEQ)
    ax_ratio.fill_between(
        x, mean_ratio - se_ratio, mean_ratio + se_ratio,
        color=COLORS['blue_light'], alpha=0.55, linewidth=0,
        zorder=2,
    )
    ax_ratio.plot(
        x, mean_ratio,
        color=COLORS['blue'], linewidth=1.1,
        marker='o', markersize=3.0,
        markerfacecolor=COLORS['blue'],
        markeredgecolor=COLORS['blue_dark'],
        markeredgewidth=0.4,
        zorder=3,
    )
    ax_ratio.axhline(1.0, color=COLORS['muted'],
                     linestyle=(0, (1, 2)), linewidth=0.8, zorder=1)
    ax_ratio.set_xlabel('# training sequences seen', labelpad=1.5)
    ax_ratio.set_ylabel('GDC ppl / true ppl')
    ax_ratio.set_title(f'Mean perplexity ratio  (mean over {N_RUNS} runs, '
                       r'$\pm$1 SE)', pad=4)
    ax_ratio.set_xlim(5.0, N_SEQ - 0.5)
    # Set ylim from the visible portion only — otherwise the early-training
    # spikes (which we are deliberately cropping out via xlim) still drive
    # matplotlib's automatic y-scaling.
    vis_lo = (mean_ratio - se_ratio)[5:]
    vis_hi = (mean_ratio + se_ratio)[5:]
    y_pad = 0.05 * (vis_hi.max() - vis_lo.min())
    ax_ratio.set_ylim(min(1.0, vis_lo.min() - y_pad), vis_hi.max() + y_pad)
    for s in ax_ratio.spines.values():
        s.set_color(COLORS['text'])
        s.set_linewidth(0.7)
    ax_ratio.tick_params(direction='in', length=2.5, width=0.7)

    # ---- Forecast heatmaps (GDC + true HMM, on the same prefix) ----
    cmap = LinearSegmentedColormap.from_list(
        'amber_seq', [COLORS['bg'], COLORS['amber']]
    )

    def _draw_forecast(ax, forecast_dists, title):
        data = np.zeros((N_EMIT, 10))
        for i, obs in enumerate(prefix):
            data[int(obs), i] = 1.0
        for k, dist in enumerate(forecast_dists):
            data[:, 5 + k] = dist
        ax.imshow(
            data, aspect='auto',
            cmap=cmap, vmin=0, vmax=1, origin='upper',
            interpolation='nearest',
        )
        ax.axvline(4.5, color=COLORS['muted'],
                   linestyle=(0, (1, 2)), linewidth=0.9, zorder=2)
        ax.set_xticks(np.arange(10))
        ax.set_xticklabels([str(i + 1) for i in range(10)])
        ax.set_yticks(np.arange(N_EMIT))
        ax.set_yticklabels(EMIT_LABELS)
        ax.set_title(title, pad=4)
        for s in ax.spines.values():
            s.set_color(COLORS['text'])
            s.set_linewidth(0.7)
        ax.tick_params(direction='in', length=2.5, width=0.7)
        for r in range(N_EMIT):
            for c in range(10):
                v = data[r, c]
                if v < 0.005:
                    continue
                txt_color = COLORS['text'] if v < 0.55 else COLORS['bg']
                ax.text(c, r, f'{v:.2f}'.lstrip('0') if v < 1 else '1',
                        ha='center', va='center', fontsize=7,
                        color=txt_color)

    _draw_forecast(ax_fc_gdc, forecast_gdc,
                   'Prefix + GDC posterior predictive (5 + 5 steps)')
    _draw_forecast(ax_fc_hmm, forecast_hmm,
                   'Prefix + true HMM posterior predictive (5 + 5 steps)')
    ax_fc_gdc.tick_params(labelbottom=False)
    ax_fc_gdc.set_xlabel('')
    ax_fc_hmm.set_xlabel('time step', labelpad=1.5)

    out_pdf = os.path.join(HERE, 'hmm_perplexity.pdf')
    out_png = os.path.join(HERE, 'hmm_perplexity.png')
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=220)
    print(f'Saved {out_pdf}')
    print(f'Saved {out_png}')


if __name__ == '__main__':
    main()
