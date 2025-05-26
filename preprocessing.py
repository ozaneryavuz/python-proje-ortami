import logging
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def preprocess_data_advanced(data: pd.DataFrame):
    """Apply imputation, scaling and encoding to the dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw input data.

    Returns
    -------
    tuple
        Transformed array and the fitted preprocessor.
    """
    # Identify numeric and categorical columns
    numerical_features = data.select_dtypes(include=[np.number]).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    # Pipeline for numeric data
    numerical_pipeline = Pipeline(
        steps=[
            ('imputer', IterativeImputer(random_state=42)),
            ('scaler', StandardScaler())
        ]
    )

    # Pipeline for categorical data
    categorical_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )

    # Fit and transform the data
    transformed = preprocessor.fit_transform(data)
    return transformed, preprocessor


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """Create additional statistical features from input odds."""
    logging.info("Özellik mühendisliği gerçekleştiriliyor")
    new_features = pd.DataFrame()

    # Basic statistics using all columns except the target
    new_features['mean_odds'] = data.iloc[:, :-1].mean(axis=1)
    new_features['std_odds'] = data.iloc[:, :-1].std(axis=1)
    new_features['var_odds'] = data.iloc[:, :-1].var(axis=1)
    new_features['range_odds'] = data.iloc[:, :-1].max(axis=1) - data.iloc[:, :-1].min(axis=1)
    new_features['sum_odds'] = data.iloc[:, :-1].sum(axis=1)
    new_features['prod_odds'] = data.iloc[:, :-1].prod(axis=1)

    # Return dataset with engineered features appended
    return pd.concat([data, new_features], axis=1)

