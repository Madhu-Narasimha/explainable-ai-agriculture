
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import shap
import warnings
warnings.filterwarnings("ignore")

# Title
st.title("Explainable AI in Agriculture - Wheat Yield Prediction")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("Wheat_Filtered_Cleaned.xlsx")
    # Convert 'Year' to numeric and clean
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])
    df = df.sort_values('Year')
    return df

df = load_data()

# Show dataset preview
if st.checkbox("Show Raw Data"):
    st.write(df.head())

# Feature and target selection
features = ['Yield of CT', 'P', 'Tmax', 'Years since NT started (yrs)']
target = 'Relative yield change'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and evaluation
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}

r2_scores = {}

st.subheader("Model Performance Comparison")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores[name] = r2
    st.write(f"{name} R² Score: {r2:.3f}")

# Bar chart of model performance
st.subheader("Model Comparison (R² Score)")
fig, ax = plt.subplots()
sns.barplot(x=list(r2_scores.keys()), y=list(r2_scores.values()), ax=ax)
plt.ylabel("R² Score")
plt.xticks(rotation=45)
st.pyplot(fig)

# SHAP Explanation for Gradient Boosting
st.subheader("SHAP Explanation - Gradient Boosting")
explainer = shap.Explainer(models["Gradient Boosting"], X_train)
shap_values = explainer(X_test)

st.write("SHAP Summary Plot:")
fig2 = plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig2)

# Show feature importance
st.subheader("Feature Importances (Gradient Boosting)")
importances = models["Gradient Boosting"].feature_importances_
importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
st.write(importance_df.sort_values(by="Importance", ascending=False))
