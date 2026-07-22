import streamlit as st
import pandas as pd
import plotly.express as px

from src.streamlit_utils import load_predictor, _get_model_version

predictor = load_predictor(_get_model_version())

st.title("🏆 Tournament Simulator")

if "sim_results" not in st.session_state:
    st.session_state.sim_results = None

if st.button("Simulate Tournament"):

    with st.spinner("Running simulations..."):

        predictor.load_2026_teams()

        for group in predictor.groups:
            predictor.simulate_group(group)

        qualifiers = predictor.get_group_qualifiers()

        mc = predictor.run_many_simulations(
            qualifiers,
            n_simulations=2000
        )

    st.session_state.sim_results = mc

if st.session_state.sim_results is not None:

    mc = st.session_state.sim_results
    champion = mc["probabilities"][0]

    st.success(
        f'Champion: {champion["team"]} — {champion["probability"]}% probability'
    )

    st.subheader("Top Contenders")

    df = pd.DataFrame(mc["probabilities"]).head(6)
    df.columns = ["Team", "Wins", "Probability (%)"]

    fig = px.pie(
        df,
        names="Team",
        values="Probability (%)",
        hole=0.3,
    )
    fig.update_traces(textposition="inside", textinfo="label+percent")
    fig.update_layout(showlegend=True)

    st.plotly_chart(fig, use_container_width=True)