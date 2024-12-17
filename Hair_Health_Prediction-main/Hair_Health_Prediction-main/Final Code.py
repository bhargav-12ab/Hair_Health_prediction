# Step 01: Install necessary libraries
!pip install numpy pandas scipy scikit-learn tensorflow matplotlib seaborn jupyter keras-tuner imbalanced-learn pingouin plotly

# Step 2: Import necessary libraries
import pandas as pd
import numpy as np
import random
import os
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pingouin as pg
import plotly.express as px
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Conv1D, MaxPooling1D, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.regularizers import l2


# Step 3: Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Ensure reproducibility across sessions
set_seed()


# Step 4: Load the dataset
data = pd.read_csv("Predict Hair Fall.csv")
data.columns = data.columns.str.strip()


# Step 5: Data Preprocessing
data.fillna(method='ffill', inplace=True)  # Fill missing values (modify as needed)
data.drop_duplicates(inplace=True)


# Step 6: Display dataset information
print(data.info())
print(data.describe())


# EDA Analysis
# Step 7: Display unique values for categorical features
categorical_features = data.select_dtypes(include=['object', 'category']).columns.to_list()
print(f'Total categorical features: {len(categorical_features)}')
for feature in categorical_features:
    print(f'{feature}: {data[feature].unique()}\n')

# Step 8: Plot numerical feature distribution
numerical_features = ['Age']  # Add more numerical features if needed

for numerical_feature in numerical_features:
    plt.figure(figsize=(5, 3.2))
    sns.kdeplot(data[numerical_feature], fill=True, color='green')
    sns.histplot(data[numerical_feature], stat='density', fill=False, color='green')
    plt.title(f"Distribution of {numerical_feature}", color='black')
    plt.show()

    # QQ plot
    plt.figure(figsize=(3, 3.2))
    sm.qqplot(data[numerical_feature], line='q', lw=2.1)
    plt.title(f'QQ Plot of {numerical_feature}', color='black')
    plt.show()

    # Skewness and Kurtosis
    print(f'Skewness: {data[numerical_feature].skew()}')
    print(f'Kurtosis: {data[numerical_feature].kurt()}')

    # Normality test
    print(pg.normality(data[numerical_feature]))


# Step 9: Pie charts for categorical features
for feature in categorical_features:
    feature_counts = data[feature].value_counts().to_frame().reset_index()
    feature_counts.columns = [feature, 'count']
    fig = px.pie(feature_counts,
                 values='count',
                 names=feature,
                 title=f'Distribution of {feature}')
    fig.show()


# Step 10: Count plots for categorical features
for feature in categorical_features:
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax = sns.countplot(data=data, y=feature, palette='dark')
    for bars in ax.containers:
        ax.bar_label(bars, fontsize=9, fontweight='bold', color='black')
    plt.title(f'Count of {feature}', fontsize=14)
    plt.show()


# Step 11: KDE plots for 'Age' by 'Hair Loss'
if 'Hair Loss' in data.columns:  # Ensure the target variable exists
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.kdeplot(data=data, x='Age', hue='Hair Loss', fill=True, ax=ax)
    sns.rugplot(data=data, x='Age', hue='Hair Loss', ax=ax)
    plt.title('KDE of Age by Hair Loss', fontsize=14)
    plt.show()

# Step 12: Grouped statistics
if 'Hair Loss' in data.columns:
    print(data.groupby('Hair Loss')['Age'].describe().style.bar(color='gold', subset=['mean', '50%']))
    print(pg.normality(data, dv='Age', group='Hair Loss'))
    print(pg.homoscedasticity(data, dv='Age', group='Hair Loss'))


# Step 13: Correlation matrix as an alternative to PPS
# Convert categorical columns to numeric or drop them
data_numeric = data.select_dtypes(include=[np.number])

# Calculate the correlation matrix
corr_matrix = data_numeric.corr()

# Plot the correlation matrix
plt.figure(figsize=(20, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, linewidths=1.1, square=True)
plt.title("Correlation Matrix", fontsize=12, fontweight='bold', color='black')
plt.show()


# Step 14: FacetGrid plots
for feature in categorical_features:
    g = sns.FacetGrid(data, col="Hair Loss", row=feature)
    g.map_dataframe(sns.kdeplot, x="Age", fill=True)
    plt.subplots_adjust(top=0.8)
    g.fig.suptitle(f'KDE of Age by {feature}', fontsize=16)
    plt.show()


# Step 15: Split the data into features and target
X = data.drop(columns=['Hair Loss'])
y = data['Hair Loss']


# Step 16: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 17: Identify numerical and categorical columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Step 18: Create and apply preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_columns)
    ])

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)


# Step 19: Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_preprocessed, y_train)

# Step 20: Convert to numpy arrays and reshape for RNN and CNN
X_train_array = X_train_preprocessed
X_test_array = X_test_preprocessed
y_train_array = y_train.values
y_test_array = y_test.values

X_train_rnn = X_train_array.reshape((X_train_array.shape[0], X_train_array.shape[1], 1))
X_test_rnn = X_test_array.reshape((X_test_array.shape[0], X_test_array.shape[1], 1))


# Step 21: Define custom focal loss for class imbalance
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl = - alpha_t * K.pow((1 - p_t), gamma) * K.log(p_t)
        return K.mean(fl)
    return focal_loss_fixed

# Step 22: Create CNN + RNN hybrid model
def create_cnn_rnn_model(input_shape):
    model = Sequential()

    # CNN part
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # RNN part (GRU)
    model.add(GRU(64, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(GRU(32, return_sequences=False))

    # Fully connected layers
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.0003), loss=focal_loss(), metrics=[f2_score])

    return model

# Step 23: Define F2 score metric
def f2_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred_binary = tf.where(y_pred > 0.5, 1.0, 0.0)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred_binary, tf.float32))
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred_binary, tf.float32))
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred_binary), tf.float32))

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f2 = 3 * precision * recall / (precision + recall + K.epsilon())
    return f2


# Step 24: Use class weights for class imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_array), y=y_train_array)
class_weight = dict(enumerate(class_weights))


# Step 25: Define learning rate scheduler and early stopping
early_stopping = EarlyStopping(monitor='val_f2_score', mode='max', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_f2_score', factor=0.5, patience=5, min_lr=1e-6, mode='max')


# Step 26: Create CNN + RNN model
cnn_rnn_model = create_cnn_rnn_model((X_train_rnn.shape[1], 1))


# Step 27: Train the model with the balanced dataset
cnn_rnn_history = cnn_rnn_model.fit(X_train_rnn, y_train_balanced, epochs=150, batch_size=64, validation_split=0.2,
                                    callbacks=[early_stopping, reduce_lr], class_weight=class_weight)


# Step 28: Evaluate the model
cnn_rnn_pred = cnn_rnn_model.predict(X_test_rnn)
cnn_rnn_pred_binary = (cnn_rnn_pred > 0.1).astype(int)

cnn_rnn_f2_test = f2_score(y_test_array, cnn_rnn_pred_binary)
cnn_rnn_f2_train = f2_score(y_train_array, (cnn_rnn_model.predict(X_train_rnn) > 0.1).astype(int))

print("F2 Test Score:", K.eval(cnn_rnn_f2_test))
print("F2 Train Score:", K.eval(cnn_rnn_f2_train))


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Step 29: Find the optimal threshold
thresholds = np.arange(0.0, 1.0, 0.1)
f2_scores = []

def f2_score(precision, recall):
    """Calculate the F2 score."""
    return (5 * precision * recall) / (4 * precision + recall) if (4 * precision + recall) != 0 else 0

for threshold in thresholds:
    preds_binary = (cnn_rnn_pred > threshold).astype(int)
    # Get precision and recall
    precision = precision_score(y_test_array, preds_binary)
    recall = recall_score(y_test_array, preds_binary)
    f2 = f2_score(precision, recall)
    f2_scores.append(f2)  # Append F2 score

optimal_threshold = thresholds[np.argmax(f2_scores)]
print(f"Optimal Threshold: {optimal_threshold}")

# Apply optimal threshold
cnn_rnn_pred_binary = (cnn_rnn_pred > optimal_threshold).astype(int)

# Print updated classification report
print("\nClassification Report with Optimal Threshold:")
print(classification_report(y_test_array, cnn_rnn_pred_binary))

# Step 30: Classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test_array, cnn_rnn_pred_binary))

# Try simple neural network architecture
nn_model = Sequential()
nn_model.add(Dense(64, activation='relu', input_shape=(X_train_array.shape[1],)))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(1, activation='sigmoid'))

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

nn_model.fit(X_train_array, y_train_array, epochs=10, batch_size=32, validation_data=(X_val, y_val))

nn_pred = nn_model.predict(X_val)
nn_pred_binary = (nn_pred > 0.5).astype(int)

print("\nNeural Network Classification Report:")
print(classification_report(y_val, nn_pred_binary))

# Neural Network Confusion Matrix
nn_conf_matrix = confusion_matrix(y_val, nn_pred_binary)
plt.figure(figsize=(5, 4))
sns.heatmap(nn_conf_matrix, annot=True, fmt='d', cmap='Oranges')
plt.title("Neural Network Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Optional: Print the confusion matrices as raw numbers
print("\nNeural Network Confusion Matrix:\n", nn_conf_matrix)


# Step 32: Plot training history
# Plot training and testing history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cnn_rnn_history.history['loss'], label='Train Loss')
plt.plot(cnn_rnn_history.history['val_loss'], label='Validation Loss')
plt.title('Best Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cnn_rnn_history.history['f2_score'], label='Train F2')
plt.plot(cnn_rnn_history.history['val_f2_score'], label='Validation F2')
plt.title('Best Model F2 Score')
plt.xlabel('Epoch')
plt.ylabel('F2 Score')
plt.legend()

plt.show()

