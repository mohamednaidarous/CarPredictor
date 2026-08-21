# 🚗 Car Predictor

An interactive **Streamlit** app that predicts the market value of a used car in **CAD**, trained on real Canadian used-car listings using a Gradient Boosting Regressor.

Enter a vehicle's details — make, year, mileage, body type, drivetrain, transmission, fuel type, engine size, and province — and get an instant price estimate along with a typical price range, plus a full set of interactive charts for exploring the underlying data and model.

## Features

- **Price prediction** — configure a vehicle in the sidebar and click **Predict** to get an estimated market value with a typical range (±8%).
- **Price Distribution** — box plots of price by make, body type, fuel type, drivetrain, transmission, or province, plus a mileage-vs-price scatter plot.
- **Feature Importance** — see which vehicle attributes most influence the predicted price, and how median price trends by model year.
- **Model Performance** — actual vs. predicted scatter plot, residual distribution, and R² / MAE / RMSE metrics on a hold-out test set.
- **Dataset Explorer** — filter listings by make and body type, browse the data in a table, and download the filtered results as CSV.

## How it works

The app loads and cleans a CSV of used-car listings (`used_car_canada_clean.csv`), filtering out unrealistic prices, mileages, and years, and normalizing categorical fields. It then trains a `GradientBoostingRegressor` (scikit-learn) on the cleaned data using the following features:

- Make
- Year
- Mileage
- Body type
- Drivetrain
- Transmission
- Fuel type
- Engine size
- Engine block
- Province/state

Categorical columns are label-encoded before training. Data loading and model training are cached with Streamlit's `@st.cache_data` / `@st.cache_resource` so the app starts quickly after the first run.

## Getting started

### Prerequisites

- Python 3.9+

### Installation

```bash
git clone https://github.com/mohamednaidarous/CarPredictor.git
cd CarPredictor
pip install streamlit pandas numpy plotly scikit-learn
```

### Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Data

The repo includes several CSV datasets of used-car listings:

- `used_car_canada_clean.csv` — the cleaned dataset used by the app for training and prediction.
- `ca-dealers-used.csv` — raw dealer listings data.
- `honda_toyota_ca.csv` — a Honda/Toyota-specific subset.

## Tech stack

- [Streamlit](https://streamlit.io/) — web app framework
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — data loading and processing
- [scikit-learn](https://scikit-learn.org/) — Gradient Boosting Regressor model
- [Plotly](https://plotly.com/python/) — interactive charts

## License

No license specified yet — consider adding one (e.g. MIT) if you plan to share or accept contributions.
