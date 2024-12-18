from flask import Flask, render_template, request, flash, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_EXTENSIONS'] = ['.csv', '.xlsx', '.tsv']
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
df = None
model = None

def encode_categorical_columns(df):
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    return df

@app.route("/", methods=["GET", "POST"])
def main():
    global df, model
    accuracy, precision, recall, report, plot_url = None, None, None, None, None
    confusion_matrix_url, auc_roc_url = None, None
    columns = []
    data_head = None
    missing_values_report = None
    summary_statistics = None
    eda_plots = []
    statistical_inferences = None

    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            session['filename'] = file.filename
            file_extension = file.filename.split('.')[-1].lower()

            if f".{file_extension}" not in app.config['UPLOAD_EXTENSIONS']:
                flash("Invalid file type. Please upload a CSV, Excel, or TSV file.")
                return render_template("index.html", columns=columns, filename="No file chosen", data_head=data_head)

            try:
                if file_extension == "csv":
                    df = pd.read_csv(file)
                elif file_extension == "xlsx":
                    df = pd.read_excel(file, engine='openpyxl')
                elif file_extension == "tsv":
                    df = pd.read_csv(file, sep='\t')

                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], "uploaded_data.csv"), index=False)

                columns = df.columns.tolist()
                df.replace('?', np.nan, inplace=True)
                session['columns'] = columns

                data_head = df.head().to_html(classes='data', header="true", index="false")

            except Exception as e:
                flash(f"Error reading file: {e}")
                return render_template("index.html", columns=columns, filename="No file chosen", data_head=data_head)

        # Handle Exploratory Data Analysis
        elif request.form.get("action") == "Show Summary Statistics" and df is not None:
            selected_columns = request.form.getlist("selected_columns")
            if selected_columns:
                summary_statistics = df[selected_columns].describe().to_html(classes='data', header="true", index="false")

        elif request.form.get("action") == "Show Distribution Plots" and df is not None:
            selected_columns = request.form.getlist("selected_columns")
            if selected_columns:
                for col in selected_columns:
                    plt.figure(figsize=(8, 6))
                    sns.histplot(df[col], kde=True)
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    plot_url = base64.b64encode(buf.getvalue()).decode("utf8")
                    buf.close()
                    plt.close()
                    eda_plots.append((col, plot_url))

        elif request.form.get("action") == "Handle Missing Values" and df is not None:
            fill_method = request.form.get("fill_method")
            numeric_data = df.select_dtypes(include=['number'])

            try:
                if fill_method == "Mean":
                    imputer = SimpleImputer(strategy='mean')
                    df[numeric_data.columns] = imputer.fit_transform(numeric_data)
                    missing_values_report = "Missing values handled using Mean."
                elif fill_method == "Median":
                    imputer = SimpleImputer(strategy='median')
                    df[numeric_data.columns] = imputer.fit_transform(numeric_data)
                    missing_values_report = "Missing values handled using Median."
                elif fill_method == "Mode":
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[numeric_data.columns] = imputer.fit_transform(numeric_data)
                    missing_values_report = "Missing values handled using Mode."
                elif fill_method == "Drop":
                    df.dropna(inplace=True)
                    missing_values_report = "Rows with missing values dropped."

            except Exception as e:
                flash(f"Error handling missing values: {e}")

        elif request.form.get("action") == "Normalize Data" and df is not None:
            numeric_data = df.select_dtypes(include=['number'])
            try:
                scaler = MinMaxScaler()
                df[numeric_data.columns] = scaler.fit_transform(numeric_data)
                missing_values_report = "Data normalization applied."
            except Exception as e:
                flash(f"Error normalizing data: {e}")

        elif request.form.get("action") == "Generate Plot" and df is not None:
            x_axis = request.form.get("x_axis")
            y_axis = request.form.get("y_axis")
            plot_type = request.form.get("plot_type")

            if x_axis and y_axis and plot_type:
                plt.figure(figsize=(8, 6))
                try:
                    if plot_type == "scatter":
                        sns.scatterplot(data=df, x=x_axis, y=y_axis)
                    elif plot_type == "line":
                        sns.lineplot(data=df, x=x_axis, y=y_axis)
                    elif plot_type == "bar":
                        sns.barplot(data=df, x=x_axis, y=y_axis)
                    elif plot_type == "hist":
                        sns.histplot(data=df[x_axis], kde=True)

                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    plot_url = base64.b64encode(buf.getvalue()).decode("utf8")
                    buf.close()
                    plt.close()

                except Exception as e:
                    flash(f"Error generating plot: {e}")

        elif request.form.get("action") == "Train Model" and df is not None:
            target_column = request.form.get("target")
            learning_rate = float(request.form.get("learning_rate", 0.1))
            optimization_method = request.form.get("optimization_method")

            try:
                X = df.drop(target_column, axis=1)
                y = df[target_column]
                X = encode_categorical_columns(X)

                # Splitting dataset
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Handling class imbalance
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

                # Training the LightGBM model
                model = lgb.LGBMClassifier(learning_rate=learning_rate)
                param_grid = {
                    'num_leaves': [20, 30, 40],
                    'n_estimators': [50, 100, 150]
                }

                if optimization_method == "grid_search":
                    search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=2)
                    search.fit(X_resampled, y_resampled)
                    model = search.best_estimator_
                elif optimization_method == "random_search":
                    search = RandomizedSearchCV(model, param_grid, cv=3, n_iter=5, scoring='accuracy', random_state=42, verbose=2)
                    search.fit(X_resampled, y_resampled)
                    model = search.best_estimator_
                else:
                    model.fit(X_resampled, y_resampled)

                y_pred = model.predict(X_test)

                # Model evaluation
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                report = classification_report(y_test, y_pred)

                # Confusion Matrix Plot
                conf_matrix = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                confusion_matrix_url = base64.b64encode(buf.getvalue()).decode("utf8")
                buf.close()
                plt.close()

                # AUC-ROC Curve
                y_prob = model.predict_proba(X_test)
                if y_test.nunique() == 2:  # For binary classification
                    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("AUC-ROC Curve")
                    plt.legend(loc="lower right")
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    auc_roc_url = base64.b64encode(buf.getvalue()).decode("utf8")
                    buf.close()
                    plt.close()

            except Exception as e:
                flash(f"Error during model training: {e}")

        # Generate Statistical Inferences based on results
        if accuracy or precision or recall:
            statistical_inferences = f"""
            The trained model achieved an accuracy of {accuracy:.2f}, 
            a precision of {precision:.2f}, and a recall of {recall:.2f}.
            This suggests the model performs well on this dataset, 
            balancing between identifying true positives and avoiding false positives.
            """

        if summary_statistics:
            statistical_inferences = (statistical_inferences or "") + """
            The summary statistics indicate the central tendency and dispersion of the selected columns.
            Consider features with high variability for further analysis or normalization.
            """

        if eda_plots:
            statistical_inferences = (statistical_inferences or "") + """
            Distribution plots highlight the data's spread and potential outliers.
            Check for skewness or multimodal distributions that may require transformations.
            """            

    columns = session.get('columns', [])
    filename = session.get('filename', 'No file chosen')

    return render_template("index.html", columns=columns, filename=filename, accuracy=accuracy, precision=precision, recall=recall, report=report, plot_url=plot_url, data_head=data_head, missing_values_report=missing_values_report, summary_statistics=summary_statistics, eda_plots=eda_plots, confusion_matrix_url=confusion_matrix_url, auc_roc_url=auc_roc_url, statistical_inferences=statistical_inferences)

if __name__ == "__main__":
    app.run(debug=True)
