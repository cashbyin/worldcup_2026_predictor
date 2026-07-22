import streamlit as st
import plotly.graph_objects as go

from src.streamlit_utils import load_predictor, _get_model_version

predictor = load_predictor(_get_model_version())

st.title("⚽ Match Predictor")

col1, col2 = st.columns(2)

teams = sorted(predictor.current_elo.keys())

with col1:
    team_a = st.selectbox("Team A", teams, key="team_a")

with col2:
    team_b = st.selectbox("Team B", teams, key="team_b")

if st.button("Predict Match", use_container_width=True):
    result = predictor.predict_match_proba(team_a, team_b)

    st.markdown("---")
    st.subheader("🎯 Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Win Probability",
            value=f"{result['team_a_win']*100:.1f}%",
            delta=None
        )

    with col2:
        st.metric(
            label="Draw Probability",
            value=f"{result['draw']*100:.1f}%",
            delta=None
        )

    with col3:
        st.metric(
            label="Loss Probability",
            value=f"{result['team_b_win']*100:.1f}%",
            delta=None
        )

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Outcome Probabilities")

        fig = go.Figure(data=[
            go.Bar(
                y=[team_a, "Draw", team_b],
                x=[result['team_a_win']*100, result['draw']*100, result['team_b_win']*100],
                orientation='h',
                marker=dict(
                    color=['#2E86AB', '#A23B72', '#F18F01'],
                ),
                text=[f"{result['team_a_win']*100:.1f}%", f"{result['draw']*100:.1f}%", f"{result['team_b_win']*100:.1f}%"],
                textposition='outside',
            )
        ])

        fig.update_layout(
            xaxis_title="Probability (%)",
            yaxis_title="Outcome",
            height=300,
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig.update_xaxes(range=[0, 100])

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Summary")

        probabilities = {
            f"🟢 {team_a} Win": f"{result['team_a_win']*100:.1f}%",
            "🔵 Draw": f"{result['draw']*100:.1f}%",
            f"🔴 {team_b} Win": f"{result['team_b_win']*100:.1f}%",
        }

        for outcome, prob in probabilities.items():
            st.write(f"**{outcome}**: {prob}")

    st.markdown("---")

    st.subheader("📊 Model Confidence")

    max_prob = max(result['team_a_win'], result['draw'], result['team_b_win'])
    confidence = max_prob * 100

    st.progress(confidence / 100, text=f"Confidence: {confidence:.1f}%")