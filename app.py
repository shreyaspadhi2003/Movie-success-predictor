# --- app.py ---

import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from utils.preprocess import engineer_features
import warnings

# UI Setup
st.set_page_config(layout="wide")
st.title("🎬 Movie Rating Predictor")

plt.switch_backend('Agg')

try:
    name_lookups = joblib.load('models/name_lookups.pkl')
    stats = joblib.load('models/dataset_stats.pkl')
    top_genres = joblib.load('models/top_genres.pkl')

    model_type = st.sidebar.selectbox("Model", ["Random Forest", "XGBoost"])

    if model_type == "Random Forest":
        reg_model = joblib.load('models/rf_reg.pkl')
        explainer = joblib.load('models/rf_reg_explainer.pkl')
    else:
        reg_model = joblib.load('models/xgb_reg.pkl')
        explainer = joblib.load('models/xgb_reg_explainer.pkl')

except FileNotFoundError as e:
    st.error(f" Missing files! Please run train.py first.\nError: {str(e)}")
    st.stop()

with st.form("movie_input"):
    col1, col2 = st.columns(2)

    with col1:
        runtime = st.slider(
            "Runtime (minutes)",
            min_value=stats['min_runtime'],
            max_value=stats['max_runtime'],
            value=stats['common_runtime'],
            help=f"Dataset range: {stats['min_runtime']}-{stats['max_runtime']} mins"
        )

        release_year = st.slider(
            "Release Year",
            min_value=stats['min_year'],
            max_value=stats['max_year'],
            value=2020,
            help=f"Dataset range: {stats['min_year']}-{stats['max_year']}"
        )

    with col2:
        actors = st.multiselect(
            "Lead Actors",
            options=name_lookups['actors'],
            default=[],
            placeholder="Type actor names..."
        )

        directors = st.multiselect(
            "Directors",
            options=name_lookups['directors'],
            default=[],
            placeholder="Type director names..."
        )

    genres = st.multiselect(
        "Genres",
        top_genres,
        default=[]
    )

    if st.form_submit_button("Predict"):
        if not actors or not directors:
            st.warning("Please select at least one actor and one director")
            st.stop()

        if len(genres) == 0:
            st.warning("Please select at least one genre")
            st.stop()

        def get_avg(names, lookup):
            return sum(lookup.get(name, 6.5) for name in names)/max(1, len(names))

        actor_avg = get_avg(actors, name_lookups['actor_avg'])
        director_avg = get_avg(directors, name_lookups['director_avg'])

        input_features = {
            'runtime': runtime,
            'release_year': release_year,
            'actor_avg': actor_avg,
            'director_avg': director_avg,
            **{f'genre_{g}': 0 for g in top_genres}
        }

        X_base = pd.DataFrame([input_features], columns=reg_model.feature_names_in_)
        base_pred = reg_model.predict(X_base)[0]

        genre_impacts = {}
        for g in genres:
            test_features = input_features.copy()
            test_features[f'genre_{g}'] = 1
            X_test = pd.DataFrame([test_features], columns=reg_model.feature_names_in_)
            genre_pred = reg_model.predict(X_test)[0]
            genre_impacts[g] = genre_pred - base_pred

        for g in genres:
            input_features[f'genre_{g}'] = 1
        X_pred = pd.DataFrame([input_features], columns=reg_model.feature_names_in_)
        rating = reg_model.predict(X_pred)[0]

        st.success(f"## Predicted IMDb Rating: {rating:.1f}/10")

        st.write("### Performance Averages")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Actors**")
            for actor in actors:
                avg = name_lookups['actor_avg'].get(actor, 6.5)
                st.write(f"- {actor}: {avg:.1f} (Δ{avg-6.5:+.1f} vs default)")
        with col2:
            st.write("**Directors**")
            for director in directors:
                avg = name_lookups['director_avg'].get(director, 6.5)
                st.write(f"- {director}: {avg:.1f} (Δ{avg-6.5:+.1f} vs default)")

        st.write("### Genre Impact Analysis")
        if len(genres) > 0:
            for g in genres:
                st.write(f"- **{g}**: {'+' if genre_impacts[g] >=0 else ''}{genre_impacts[g]:.2f} points")
        else:
            st.write("No genres selected")

        st.write("### Feature Influence")
        try:
            shap_values = explainer.shap_values(X_pred)
            plt.figure()
            fig, ax = plt.subplots()
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                X_pred.iloc[0,:],
                matplotlib=True,
                show=False
            )
            st.pyplot(bbox_inches='tight')
            plt.close()

            plt.figure()
            fig_summary, ax_summary = plt.subplots()
            shap.summary_plot(shap_values, X_pred, plot_type="bar", show=False)
            st.pyplot(fig_summary)
            plt.close()

        except Exception as e:
            st.warning(f"Visualization error: {str(e)}")

with st.expander("Dataset Information"):
    st.write(f"- Actors: {len(name_lookups['actors'])}")
    st.write(f"- Directors: {len(name_lookups['directors'])}")
    st.write(f"- Year range: {stats['min_year']}-{stats['max_year']}")
    st.write(f"- Runtime range: {stats['min_runtime']}-{stats['max_runtime']} mins")