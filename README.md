

# Laptop Price Prediction Web App


https://github.com/user-attachments/assets/4e009830-a215-42bb-b753-de69126121df


## Overview

This project is a web application designed to predict laptop prices based on various features such as brand, processor type, RAM, storage, and more. The application uses machine learning models to provide predictions and offers a user-friendly interface for interacting with the model.

## Project Structure

The project is organized into several components:

1. **Data Collection**: The dataset used for training the model.
2. **Data Preprocessing**: Steps to clean and prepare the data for modeling.
3. **Model Training**: Training and evaluating machine learning models.
4. **Web Application**: Implementation of the web interface using Flask.
5. **Deployment**: Instructions for running the application locally or deploying it.

## Data Collection

### Dataset

The dataset used for this project contains information about laptop specifications and their prices. It includes features such as:

- Brand
- Processor Type
- RAM
- Storage Type and Size
- Display Size
- Operating System
- Weight

The dataset is sourced from [insert source if applicable].

## Data Preprocessing

### 1. Loading the Data

The data is loaded into a DataFrame using pandas:

```python
import pandas as pd

data = pd.read_csv('path_to_your_dataset.csv')
```

### 2. Handling Missing Values

Missing values are handled by:

- Dropping rows with missing target values (prices).
- Imputing missing values in categorical features with the mode.

```python
data = data.dropna(subset=['Price'])
data['RAM'] = data['RAM'].fillna(data['RAM'].mode()[0])
```

### 3. Feature Engineering

- **Extracting Numeric Values**: Convert features like 'RAM' and 'Storage' to numeric values.

```python
data['RAM'] = data['RAM'].str.extract('(\d+)').astype(int)
data['Storage'] = data['Storage'].str.extract('(\d+)').astype(int)
```

- **Encoding Categorical Features**: Use one-hot encoding for categorical features like 'Brand' and 'Processor Type'.

```python
data = pd.get_dummies(data, columns=['Brand', 'Processor Type'])
```

### 4. Splitting the Data

The data is split into training and testing sets:

```python
from sklearn.model_selection import train_test_split

X = data.drop('Price', axis=1)
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Model Training

### 1. Choosing the Model

For this project, we used a Random Forest Regressor due to its effectiveness in handling tabular data with various features.

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
```

### 2. Training the Model

```python
model.fit(X_train, y_train)
```

### 3. Evaluating the Model

Evaluate the model's performance using metrics like Mean Absolute Error (MAE) and R-squared.

```python
from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')
```

## Web Application

### 1. Setting Up Flask

Flask is used to create a web interface for the application.

```python
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')
```

### 2. Creating Routes

- **Home Route**: Renders the main page.

```python
@app.route('/')
def home():
    return render_template('index.html')
```

- **Prediction Route**: Handles form submissions and returns predictions.

```python
@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])
    return render_template('result.html', prediction=prediction[0])
```

### 3. Templates

- **index.html**: Form for user input.

```html
<!DOCTYPE html>
<html>
<head>
    <title>Laptop Price Prediction</title>
</head>
<body>
    <h1>Enter Laptop Details</h1>
    <form action="/predict" method="post">
        <!-- Form Fields for Features -->
        <input type="submit" value="Predict">
    </form>
</body>
</html>
```

- **result.html**: Displays the prediction result.

```html
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
</head>
<body>
    <h1>Predicted Price: {{ prediction }}</h1>
</body>
</html>
```

## Deployment

### Running Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Flask application:

```bash
python app.py
```

### Deployment on Heroku

1. Create a `Procfile`:

```
web: python app.py
```

2. Follow Heroku deployment steps:

- Create a Heroku account.
- Install Heroku CLI.
- Run `heroku create`.
- Deploy the app using `git push heroku master`.

## Contributing

Feel free to contribute to the project by submitting issues or pull requests.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Insert any acknowledgments or credits here]

---

Let me know if you need any more details or adjustments!
