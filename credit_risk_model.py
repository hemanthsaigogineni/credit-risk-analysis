import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Credit Risk Analysis using LSTM Neural Network
# Author: Hemanth Sai Gogineni
# Role: Data Scientist @ Mizuho Bank
# ============================================================

def load_and_preprocess_data(filepath: str) -> tuple:
    """
    Load and preprocess financial dataset for credit risk modeling.
    Handles missing values, encoding, and feature engineering.
    """
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Fill missing numerical values with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Feature engineering: debt-to-income ratio, credit utilization
    if 'loan_amount' in df.columns and 'income' in df.columns:
        df['debt_to_income'] = df['loan_amount'] / (df['income'] + 1)
    if 'credit_used' in df.columns and 'credit_limit' in df.columns:
        df['credit_utilization'] = df['credit_used'] / (df['credit_limit'] + 1)

    return df


def build_lstm_model(input_shape: tuple, num_classes: int = 2) -> Model:
    """
    Build LSTM model for credit risk prediction.
    Architecture: LSTM -> BatchNorm -> Dense -> Dropout -> Output
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary: default or not
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


def train_model(X_train, y_train, X_val, y_val, epochs: int = 50, batch_size: int = 64):
    """
    Train LSTM model with early stopping and model checkpointing.
    """
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max'),
        ModelCheckpoint('best_credit_risk_model.h5', monitor='val_auc', save_best_only=True, mode='max')
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance: AUC-ROC, classification report, confusion matrix.
    """
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    print("\n===== Model Evaluation =====")
    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    plt.title('Confusion Matrix - Credit Risk Model')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    return y_pred_prob


def plot_training_history(history):
    """
    Plot training and validation loss/AUC curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history.history['auc'], label='Train AUC')
    axes[1].plot(history.history['val_auc'], label='Val AUC')
    axes[1].set_title('Model AUC-ROC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int = 10) -> tuple:
    """
    Reshape tabular data into sequences for LSTM input.
    Shape: (samples, time_steps, features)
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


if __name__ == '__main__':
    # Example usage with a synthetic dataset
    np.random.seed(42)
    n_samples = 5000

    # Simulate credit risk features
    data = {
        'age': np.random.randint(22, 65, n_samples),
        'income': np.random.normal(60000, 20000, n_samples),
        'loan_amount': np.random.normal(15000, 8000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'employment_years': np.random.randint(0, 30, n_samples),
        'debt_existing': np.random.normal(5000, 3000, n_samples),
        'num_late_payments': np.random.randint(0, 10, n_samples),
        'default': np.random.randint(0, 2, n_samples)
    }
    df = pd.DataFrame(data)
    df['debt_to_income'] = df['loan_amount'] / (df['income'] + 1)

    # Features and target
    feature_cols = ['age', 'income', 'loan_amount', 'credit_score',
                    'employment_years', 'debt_existing', 'num_late_payments', 'debt_to_income']
    X = df[feature_cols].values
    y = df['default'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create LSTM sequences
    TIME_STEPS = 10
    X_seq, y_seq = create_sequences(X_scaled, y, TIME_STEPS)
    print(f"Sequence shape: {X_seq.shape}")

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val, epochs=30)

    # Evaluate
    evaluate_model(model, X_test, y_test)
    plot_training_history(history)

    print("\nCredit Risk Model training complete. Model saved as best_credit_risk_model.h5")
