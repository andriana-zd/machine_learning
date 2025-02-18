import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, Any

def preprocess_data(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Виконує попередню обробку сирих даних, включаючи розбиття на train/val/test,
    обробку категоріальних змінних, масштабування числових ознак і заповнення пропусків.

    Args:
        raw_df (pd.DataFrame): Вхідний датафрейм з сирими даними.

    Returns:
        Dict[str, Any]: Оброблені X_train, y_train, X_val, y_val, X_test, y_test.
    """
    drop_cols = ['Surname', 'CustomerId']
    raw_df = raw_df.drop(columns=[col for col in drop_cols if col in raw_df.columns], errors='ignore')

    # Extract 'id' column separately
    ids = raw_df[['id']]

    # Define input features and target variable
    target_col = 'Exited'
    input_cols = [col for col in raw_df.columns if col not in ['id', target_col]]

    # Split into train and test
    train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)

    # Separate inputs (X) and targets (y)
    X_train, y_train = train_df[input_cols], train_df[target_col]
    X_test, y_test = test_df[input_cols], test_df[target_col]

    # Save 'id' separately
    train_ids, test_ids = train_df[['id']], test_df[['id']]

    # Identify numerical and categorical columns
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # Impute missing values for numeric data
    imputer = SimpleImputer(strategy='mean')
    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

    # Scale numeric features
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # One-hot encoding for categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
    X_test_encoded = encoder.transform(X_test[categorical_cols])

    # Get encoded column names
    encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()

    # Create final DataFrames including 'id'
    X_train_final = pd.DataFrame(np.hstack((X_train[numeric_cols], X_train_encoded)),
                                 columns=numeric_cols + encoded_cols, index=X_train.index)
    X_train_final.insert(0, 'id', train_ids.values)

    X_test_final = pd.DataFrame(np.hstack((X_test[numeric_cols], X_test_encoded)),
                                columns=numeric_cols + encoded_cols, index=X_test.index)
    X_test_final.insert(0, 'id', test_ids.values)

    # Split train into train/validation
    X_train_final, X_val_final, y_train, y_val, train_ids, val_ids = train_test_split(
        X_train_final, y_train, train_ids, test_size=0.2, random_state=42)

    return {
        'train_X': X_train_final,
        'train_y': y_train,
        'val_X': X_val_final,
        'val_y': y_val,
        'test_X': X_test_final,
        'test_y': y_test,
        'train_ids': train_ids,
        'val_ids': val_ids,
        'test_ids': test_ids,
        'scaler': scaler,
        'encoder': encoder,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }
def preprocess_new_data(new_df: pd.DataFrame, scaler, encoder, numeric_cols, categorical_cols) -> pd.DataFrame:
    """
    Попередня обробка нових даних, використовуючи навчені scaler та encoder.

    Args:
        new_df (pd.DataFrame): Нові сирі дані.
        scaler: Навчений StandardScaler.
        encoder: Навчений OneHotEncoder.
        numeric_cols (list): Числові ознаки.
        categorical_cols (list): Категоріальні ознаки.

    Returns:
        pd.DataFrame: Оброблений DataFrame.
    """
    drop_cols = ['CustomerId', 'Surname']
    new_df = new_df.drop(columns=[col for col in drop_cols if col in new_df.columns], errors='ignore')

    # Extract 'id' separately
    new_ids = new_df[['id']] if 'id' in new_df.columns else None

    # Ensure only valid numeric columns are used
    numeric_cols = [col for col in numeric_cols if col in new_df.columns]

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    new_df[numeric_cols] = imputer.fit_transform(new_df[numeric_cols])

    # Scale numeric features
    new_df[numeric_cols] = scaler.transform(new_df[numeric_cols])

    # One-hot encode categorical features
    new_encoded = encoder.transform(new_df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()

    # Create final DataFrame including 'id'
    new_df_final = pd.DataFrame(np.hstack((new_df[numeric_cols], new_encoded)),
                                columns=numeric_cols + encoded_cols, index=new_df.index)

    if new_ids is not None:
        new_df_final.insert(0, 'id', new_ids.values)

    return new_df_final
