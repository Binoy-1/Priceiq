"""Training Studio — AI control center for retraining the Q-Learning agent."""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

from utils.helpers import hero, section, status_pill, insight
from utils.charts import line, epsilon_curve
from utils.loaders import MODELS, load_coefficients

# Original RL engine — untouched.
from agent import QLearningAgent, AgentConfig
from environment import DynamicPricingEnv, EnvConfig
from data_processing import fit_loglog


def _resolve_coefficients() -> tuple[float, float, str]:
    """Get (a, b) from uploaded dataset or fall back to disk."""
    df = st.session_state.get("active_dataset")
    roles = st.session_state.get("active_roles")
    if df is not None and roles is not None:
        price_col = roles.get("price")
        qty_col = roles.get("quantity")
        if price_col and qty_col and price_col in df.columns and qty_col in df.columns:
            p = df[price_col].astype(float).values
            q = df[qty_col].astype(float).values
            mask = (p > 0) & (q > 0) & np.isfinite(p) & np.isfinite(q)
            if int(mask.sum()) >= 5:
                a, b, _ = fit_loglog(p[mask], q[mask])
                src = st.session_state.get("dataset_name", "uploaded dataset")
                return float(a), float(abs(b)), f"from {src} ({int(mask.sum())} rows)"
    coef = load_coefficients()
    if coef is not None:
        return float(coef[0]), float(abs(coef[1])), "from saved coefficients.npy"
    return 5.0, 1.5, "default (5.0, 1.5)"


def render() -> None:
    if "trained_agent" in st.session_state and not st.session_state.get("restart_training"):
        agent = st.session_state["trained_agent"]
        rewards = st.session_state["trained_rewards"]
        st.success(f"Loaded previous training run — {len(rewards)} episodes. Click Start to retrain.")

    a_val, b_val, coef_src = _resolve_coefficients()
    badge = f"TRAIN · {coef_src}"
    st.markdown(hero(
        "Training Studio",
        "Re-train the Q-Learning agent with new hyperparameters and persist the "
        "updated policy to disk. Monitor convergence in real time.",
        badge=badge,
    ), unsafe_allow_html=True)

    # Hyperparameter panel
    st.markdown('<div class="pi-card"><h3>Hyperparameters</h3>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    alpha    = c1.slider("α  Learning rate", 0.01, 1.0, 0.1, 0.01)
    gamma    = c2.slider("γ  Discount", 0.5, 0.999, 0.99, 0.001)
    eps0     = c3.slider("ε  Start", 0.05, 1.0, 0.3, 0.05)
    episodes = c4.slider("Episodes", 100, 5000, 1000, 100)
    start = st.button("▸  Start training run", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    # Preview epsilon curve
    st.markdown(section("Exploration Schedule (preview)"), unsafe_allow_html=True)
    st.plotly_chart(epsilon_curve(eps0, 0.01, 0.995, episodes), width='stretch')

    if not start:
        st.markdown(insight(
            "Adjust α, γ, ε to control learning aggressiveness, future-discounting, "
            "and exploration. Then hit <strong>Start training run</strong> to execute "
            "with live diagnostics.",
            label="Operator Tip"), unsafe_allow_html=True)
        return

    # Snapshot old Q-table for before/after comparison
    old_q = None
    try:
        old_q = np.load(MODELS / "q_table.npy")
    except Exception:
        old_q = None

    cfg_a = AgentConfig(alpha=alpha, gamma=gamma, epsilon_start=eps0)
    cfg_e = EnvConfig(a=a_val, b=b_val)
    agent = QLearningAgent(cfg_a)
    env = DynamicPricingEnv(cfg_e)

    st.markdown(section("Live Training", "real-time metrics"), unsafe_allow_html=True)
    progress = st.progress(0.0, text="Initializing…")
    m_col = st.columns(4)
    m_ep   = m_col[0].empty(); m_eps = m_col[1].empty()
    m_rwd  = m_col[2].empty(); m_avg = m_col[3].empty()
    chart_slot = st.empty()
    status_slot = st.empty()

    rewards: list[float] = []
    update_every = max(1, episodes // 80)

    for ep in range(episodes):
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        total_r = 0.0; done = False
        while not done:
            action = agent.choose_action(int(state))
            step_out = env.step(action)
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_out
            agent.update(int(state), int(action), float(reward),
                        int(next_state), bool(done))
            state = next_state; total_r += float(reward)
        rewards.append(total_r)

        if ep % update_every == 0 or ep == episodes - 1:
            progress.progress((ep + 1) / episodes,
                              text=f"Training · episode {ep+1}/{episodes}")
            m_ep.metric("Episode", f"{ep+1}/{episodes}")
            m_eps.metric("ε", f"{agent.epsilon:.3f}")
            m_rwd.metric("Last reward", f"{total_r:.3f}")
            m_avg.metric("Mean (100)", f"{np.mean(rewards[-100:]):.3f}")
            window = max(10, len(rewards) // 20)
            smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
            chart_slot.plotly_chart(
                line(range(len(smooth)), smooth,
                      "Rolling Mean Reward", "Reward", color="#8b5cf6"),
                width='stretch')
            status_slot.markdown(status_pill("Training in progress", "warn"),
                                  unsafe_allow_html=True)

    # Auto-save to disk
    out = MODELS / "q_table.npy"
    np.save(out, agent.q_table)
    st.cache_data.clear()

    st.session_state["trained_agent"] = agent
    st.session_state["trained_rewards"] = rewards

    final_mean = float(np.mean(rewards[-100:]))
    status_slot.markdown(status_pill(
        f"Run complete · final ε = {agent.epsilon:.3f} · mean reward (last 100) = {final_mean:.3f}"
        " · auto-saved to q_table.npy",
        "ok"), unsafe_allow_html=True)

    # Before / after comparison
    st.markdown(section("Before vs After", "Policy change from previous Q-table"),
                unsafe_allow_html=True)
    if old_q is not None and old_q.shape == agent.q_table.shape:
        col_a, col_b = st.columns(2)
        labels = ["LOWER", "HOLD", "RAISE"]
        states = [f"S{i}" for i in range(old_q.shape[0])]
        old_best = np.argmax(old_q, axis=1)
        new_best = np.argmax(agent.q_table, axis=1)
        changed = sum(1 for o, n in zip(old_best, new_best) if o != n)
        col_a.dataframe(
            pd.DataFrame({"State": states, "Old action": [labels[a] for a in old_best],
                          "New action": [labels[a] for a in new_best],
                          "Changed": ["✓" if o != n else "" for o, n in zip(old_best, new_best)]}),
            width='stretch', hide_index=True, height=380)
        col_b.markdown(insight(
            f"<strong>{changed}</strong> of {len(states)} states changed policy. "
            "Review Analytics page for detailed Q-value surface.",
            label="Policy Shift"), unsafe_allow_html=True)
    else:
        st.info("No previous Q-table snapshot available for comparison.")
    st.markdown(insight(
        f"Auto-saved to <code>{out}</code>. Switch to Simulator or Dashboard "
        "to see updated policy in action.", label="Saved"), unsafe_allow_html=True)
