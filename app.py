Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
# app.py

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(layout="wide")
st.title("🌾 Explainable AI in Agriculture: Wheat Yield Forecasting")

# Upload Dataset
uploaded_file = st.file_uploader("📤 Upload your Wheat Dataset (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10))

    # Simulate Year if missing
    if 'Year' not in df.columns:
        df['Year'] = np.random.randint(2000, 2022, size=len(df))

    # Select key features
    features = ['Yield of CT', 'P', 'Tmax', 'Tmin', 'E', 'Tave', 'Years since NT started (yrs)']
    target = 'Yield change (%)'

    if all(f in df.columns for f in features) and target in df.columns:

        df = df.sort_values('Year')
        X = df[features]
        y = df[target]

...         train_size = int(0.8 * len(X))
...         X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
...         y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
... 
...         model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5)
...         model.fit(X_train, y_train)
... 
...         y_pred = model.predict(X_test)
... 
...         st.subheader("📈 Model Performance")
...         st.markdown(f"**R² Score:** `{r2_score(y_test, y_pred):.3f}`")
...         st.markdown(f"**RMSE:** `{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}`")
... 
...         # SHAP explanation
...         st.subheader("📌 Feature Importance with SHAP")
...         with st.spinner("Explaining predictions..."):
...             explainer = shap.Explainer(model, X_train)
...             shap_values = explainer(X_test)
... 
...             fig, ax = plt.subplots(figsize=(10, 6))
...             shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
...             st.pyplot(fig)
... 
...         # Optional Forecast Output
...         if st.button("💾 Export Future Predictions"):
...             results_df = X_test.copy()
...             results_df['Predicted Yield Change (%)'] = y_pred
...             st.download_button(
...                 label="Download CSV",
...                 data=results_df.to_csv(index=False),
...                 file_name="wheat_yield_forecast.csv",
...                 mime="text/csv"
...             )
...     else:
...         st.error("🚫 Dataset is missing required columns.")
... else:
...     st.info("⬆️ Please upload a valid Excel file to begin.")
