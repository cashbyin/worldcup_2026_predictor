import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.streamlit_utils import load_predictor, _get_model_version

predictor = load_predictor(_get_model_version())

st.title("🤖 Model Insights")

st.markdown("---")

# Model Performance Metrics
st.subheader("📊 Model Performance Metrics")

if predictor.evaluation_results:
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Accuracy",
            f"{predictor.evaluation_results['accuracy']:.4f}"
        )

    with col2:
        st.metric(
            "Balanced Accuracy",
            f"{predictor.evaluation_results['balanced_accuracy']:.4f}"
        )

    with col3:
        st.metric(
            "Precision (Macro)",
            f"{predictor.evaluation_results['precision_macro']:.4f}"
        )

    with col4:
        st.metric(
            "Recall (Macro)",
            f"{predictor.evaluation_results['recall_macro']:.4f}"
        )

    with col5:
        st.metric(
            "F1 Score (Macro)",
            f"{predictor.evaluation_results['f1_macro']:.4f}"
        )

    st.markdown("---")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = predictor.evaluation_results["confusion_matrix"]

    # Convert to numpy array if it's a list (from loaded JSON)
    if isinstance(cm, list):
        cm = np.array(cm)

    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Home Win", "Draw", "Away Win"],
        y=["Home Win", "Draw", "Away Win"],
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 14},
    ))

    fig_cm.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        width=500
    )

    st.plotly_chart(fig_cm, use_container_width=False)
else:
    st.warning("Model evaluation results not available")

st.markdown("---")

# Feature Importance
st.subheader("⭐ Feature Importance")

if predictor.feature_importance_df is not None and not predictor.feature_importance_df.empty:

    # Display top features
    st.write("**Top 15 Most Important Features**")

    top_features = predictor.feature_importance_df.head(15)

    fig = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance Scores",
        labels={"importance": "Importance Score", "feature": "Feature"},
        color="importance",
        color_continuous_scale="Viridis"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=500,
        showlegend=False
    )

    fig.update_xaxes(title_text="Importance Score")
    fig.update_yaxes(title_text="")

    st.plotly_chart(fig, use_container_width=True)

    # Show all features in a table
    st.write("**All Features with Importance Scores**")

    display_df = predictor.feature_importance_df.copy()
    display_df["importance"] = display_df["importance"].round(6)
    display_df = display_df.reset_index(drop=True)
    display_df.index = display_df.index + 1

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=False
    )
else:
    st.warning("Feature importance data not available")

st.markdown("---")

# Current Elo Ratings
st.subheader("⚡ Current Elo Ratings")

elo = pd.DataFrame(
    predictor.current_elo.items(),
    columns=["Team", "Elo"]
)

elo = elo.sort_values("Elo", ascending=False)

st.dataframe(
    elo,
    use_container_width=True
)