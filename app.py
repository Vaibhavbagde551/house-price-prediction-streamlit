import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Prediction",layout='wide')

model = joblib.load("house_price_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

df = pd.read_csv("train.csv")

df = df.drop(columns=['Id','SalePrice'])

st.title("House Price Prediction")
st.write("Enter the house details below to predict the selling price")

st.divider()

col1,col2 = st.columns(2)

columns = df.columns.tolist()


feature_groups = {
    "Basic Information": [
        "MSSubClass", "MSZoning", "Neighborhood",
        "HouseStyle", "OverallQual", "OverallCond"
    ],

    "Construction": [
        "YearBuilt", "YearRemodAdd",
        "RoofStyle", "RoofMatl",
        "Exterior1st", "Exterior2nd"
    ],

    "Area Details": [
        "LotArea", "LotFrontage",
        "GrLivArea", "TotalBsmtSF",
        "1stFlrSF", "2ndFlrSF"
    ],

    "Basement": [
        "BsmtQual", "BsmtCond",
        "BsmtExposure",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF"
    ],

    "Garage": [
        "GarageType",
        "GarageFinish",
        "GarageCars",
        "GarageArea",
        "GarageYrBlt"
    ]
}

feature_labels = {
    "MSSubClass": "Building Class",
    "MSZoning": "Zoning Classification",
    "LotFrontage": "Lot Frontage (ft)",
    "LotArea": "Lot Area (sq ft)",
    "Street": "Street Type",
    "Alley": "Alley Access",
    "LotShape": "Lot Shape",
    "LandContour": "Land Contour",
    "Utilities": "Utilities",
    "LotConfig": "Lot Configuration",
    "LandSlope": "Land Slope",
    "Neighborhood": "Neighborhood",
    "Condition1": "Primary Condition",
    "Condition2": "Secondary Condition",
    "BldgType": "Building Type",
    "HouseStyle": "House Style",
    "OverallQual": "Overall House Quality (1-10)",
    "OverallCond": "Overall House Condition (1-10)",
    "YearBuilt": "Year Built",
    "YearRemodAdd": "Year Remodeled",
    "RoofStyle": "Roof Style",
    "RoofMatl": "Roof Material",
    "Exterior1st": "Exterior Material",
    "Exterior2nd": "Exterior Material (Second)",
    "MasVnrType": "Masonry Veneer Type",
    "MasVnrArea": "Masonry Veneer Area (sq ft)",
    "ExterQual": "Exterior Quality",
    "ExterCond": "Exterior Condition",
    "Foundation": "Foundation Type",
    "BsmtQual": "Basement Quality",
    "BsmtCond": "Basement Condition",
    "BsmtExposure": "Basement Exposure",
    "BsmtFinType1": "Basement Finish Type 1",
    "BsmtFinSF1": "Finished Basement Area 1 (sq ft)",
    "BsmtFinType2": "Basement Finish Type 2",
    "BsmtFinSF2": "Finished Basement Area 2 (sq ft)",
    "BsmtUnfSF": "Unfinished Basement Area (sq ft)",
    "TotalBsmtSF": "Total Basement Area (sq ft)",
    "Heating": "Heating Type",
    "HeatingQC": "Heating Quality",
    "CentralAir": "Central Air Conditioning",
    "Electrical": "Electrical System",
    "1stFlrSF": "First Floor Area (sq ft)",
    "2ndFlrSF": "Second Floor Area (sq ft)",
    "LowQualFinSF": "Low Quality Finished Area (sq ft)",
    "GrLivArea": "Above Ground Living Area (sq ft)",
    "BsmtFullBath": "Basement Full Bathrooms",
    "BsmtHalfBath": "Basement Half Bathrooms",
    "FullBath": "Full Bathrooms",
    "HalfBath": "Half Bathrooms",
    "BedroomAbvGr": "Bedrooms Above Ground",
    "KitchenAbvGr": "Kitchens Above Ground",
    "KitchenQual": "Kitchen Quality",
    "TotRmsAbvGrd": "Total Rooms Above Ground",
    "Functional": "Home Functionality",
    "Fireplaces": "Number of Fireplaces",
    "FireplaceQu": "Fireplace Quality",
    "GarageType": "Garage Type",
    "GarageYrBlt": "Garage Year Built",
    "GarageFinish": "Garage Finish",
    "GarageCars": "Garage Capacity (Cars)",
    "GarageArea": "Garage Area (sq ft)",
    "GarageQual": "Garage Quality",
    "GarageCond": "Garage Condition",
    "PavedDrive": "Paved Driveway",
    "WoodDeckSF": "Wood Deck Area (sq ft)",
    "OpenPorchSF": "Open Porch Area (sq ft)",
    "EnclosedPorch": "Enclosed Porch Area (sq ft)",
    "3SsnPorch": "Three Season Porch Area (sq ft)",
    "ScreenPorch": "Screen Porch Area (sq ft)",
    "PoolArea": "Swimming Pool Area (sq ft)",
    "PoolQC": "Pool Quality",
    "Fence": "Fence Quality",
    "MiscFeature": "Miscellaneous Feature",
    "MiscVal": "Miscellaneous Value ($)",
    "MoSold": "Month Sold",
    "YrSold": "Year Sold",
    "SaleType": "Sale Type",
    "SaleCondition": "Sale Condition"
}

user_input = {}

for section, features in feature_groups.items():

    st.subheader(section)

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):

        if feature not in df.columns:
            continue

        current_col = col1 if i % 2 == 0 else col2

        with current_col:
            label = feature_labels.get(feature, feature)
            
            if df[feature].dtype == "object":
            
                user_input[feature] = st.selectbox(
                    label,
                    sorted(df[feature].dropna().unique())
                )

            else:
                numerical_col = pd.to_numeric(df[feature],errors='coerce')
                user_input[feature] = st.number_input(
                    label,
                    value=float(numerical_col.median())
                )

remaining_features = [
    col for col in df.columns
    if col not in sum(feature_groups.values(), [])
]

st.subheader("Other Features")

col1, col2 = st.columns(2)

for i, feature in enumerate(remaining_features):

    current_col = col1 if i % 2 == 0 else col2

    with current_col:
        label = feature_labels.get(feature, feature)

        if df[feature].dtype == "object":

            user_input[feature] = st.selectbox(
                label,
                sorted(df[feature].dropna().unique())
            )

        else:
            numerical_col = pd.to_numeric(df[feature],errors='coerce')
            user_input[feature] = st.number_input(
                label,
                value=float(numerical_col.median())
                )

input_df = pd.DataFrame([user_input])

if st.button("Predict House Price"):

    processed_data = preprocessor.transform(input_df)
    prediction = model.predict(processed_data)

    st.success(
        f"Estimated house Price: ${prediction[0]:,.2f}"
    )

st.sidebar.title("House Price Prediction")

st.sidebar.markdown(
    """
Welcome to the **House Price Prediction App**.

This application predicts the selling price of a house using a **Random Forest Regression** model.
"""
)

st.sidebar.markdown("---")

st.sidebar.subheader("Model Information")

st.sidebar.write("**Algorithm:** Random Forest Regressor")
st.sidebar.write("**Input Features:** 77")
st.sidebar.write("**Target Variable:** SalePrice")

st.sidebar.markdown("---")

st.sidebar.subheader("Model Performance")

st.sidebar.metric(
    label="Train R² Score",
    value="0.979"
)

st.sidebar.metric(
    label="Test R² Score",
    value="0.886"
)

st.sidebar.markdown("---")

st.sidebar.subheader("Instructions")

st.sidebar.info(
    """
1. Enter the house details.
2. Click **Predict House Price**.
3. View the estimated selling price.
"""
)

st.sidebar.markdown("---")

st.sidebar.subheader("Developer")

st.sidebar.write("**Vaibhav Bagde**")















