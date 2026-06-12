"""Training Studio — AI control center for retraining the Q-Learning agent."""
from __future__ import annotations
import numpy as np
import streamlit as st

from utils.helpers import hero, section, status_pill, insight
from utils.charts import line, epsilon_curve
from utils.loaders import MODELS

# Original RL engine — untouched.
from agent import QLearningAgent, AgentConfig
from environment import DynamicPricingEnv, EnvConfig


def render() -> None:
    st.markdown(hero(
        "Training Studio",
        "Re-train the Q-Learning agent with new hyperparameters and persist the "
        "updated policy to disk. Monitor convergence in real time.",
        badge="TRAIN · CONTROL CENTER",
    ), unsafe_allow_html=True)

    # Hyperparameter panel
    st.markdown('<div class="pi-card"><h3>Hyperparameters</h3>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    alpha    = c1.slider("α  Learning rate", 0.01, 1.0, 0.1, 0.01)
    gamma    = c2.slider("γ  Discount", 0.5, 0.999, 0.99, 0.001)
    eps0     = c3.slider("ε  Start", 0.05, 1.0, 0.3, 0.05)
    episodes = c4.slider("Episodes", 100, 5000, 1000, 100)
    c5, c6 = st.columns([1, 3])
    save  = c5.checkbox("Persist Q-table", value=False)
    start = c6.button("▸  Start training run", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    # Preview epsilon curve
    st.markdown(section("Exploration Schedule (preview)"), unsafe_allow_html=True)
    st.plotly_chart(epsilon_curve(eps0, 0.01, 0.995, episodes), use_container_width=True)

    if not start:
        st.markdown(insight(
            "Adjust α, γ, ε to control learning aggressiveness, future-discounting, "
            "and exploration. Then hit <strong>Start training run</strong> to execute "
            "with live diagnostics.",
            label="Operator Tip"), unsafe_allow_html=True)
        return

    cfg_a = AgentConfig(alpha=alpha, gamma=gamma, epsilon_start=eps0)
    cfg_e = EnvConfig()
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
                use_container_width=True)
            status_slot.markdown(status_pill("Training in progress", "warn"),
                                 unsafe_allow_html=True)

    final_mean = float(np.mean(rewards[-100:]))
    status_slot.markdown(status_pill(
        f"Run complete · final ε = {agent.epsilon:.3f} · mean reward (last 100) = {final_mean:.3f}",
        "ok"), unsafe_allow_html=True)

    if save:
        out = MODELS / "q_table.npy"
        np.save(out, agent.q_table)
        st.cache_data.clear()
        st.markdown(insight(
            f"Persisted updated Q-table → <code>{out}</code>. Reload other pages to "
            "see the new policy take effect.", label="Saved"), unsafe_allow_html=True)
