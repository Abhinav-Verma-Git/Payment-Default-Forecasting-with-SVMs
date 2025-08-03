
<!-- README.md for Credit Card Default Prediction (SVM Project) -->
<h1 align="center">💳 Payment-Default-Forecasting-with-SVMs</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg">
  <img src="https://img.shields.io/badge/Libraries-pandas%2C%20numpy%2C%20scikit--learn%2C%20seaborn%2C%20matplotlib-brightgreen">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg">
</p>

<hr>

<h2>📖 Project Overview</h2>
<p>
  This project implements a <b>Support Vector Machine (SVM)</b> approach for predicting credit card default risk, using the "Default of Credit Card Clients" dataset. The workflow covers data cleaning, feature engineering, visualization, dimensionality reduction, SVM training/optimization, and model evaluation.
</p>

<hr>

<h2>🧰 Libraries &amp; Tools</h2>
<ul>
  <li><b>pandas</b>: Data handling and manipulation</li>
  <li><b>numpy</b>: Numerical computations</li>
  <li><b>seaborn &amp; matplotlib</b>: Data visualization</li>
  <li><b>scikit-learn</b>: Feature engineering, modeling, evaluation</li>
  <li><b>SVM</b>: Classification algorithm</li>
  <li><b>PCA</b>: For dimensionality reduction and visualization</li>
</ul>

<hr>

<h2>🚀 Workflow</h2>
<ol>
  <li>Load and clean the dataset from a remote source.</li>
  <li>Validate and filter invalid values; downsample for class balance.</li>
  <li>Visualize class distribution and relationships with seaborn/matplotlib.</li>
  <li>Feature encoding (OneHotEncoder) and scaling (standardization).</li>
  <li>Split into training and testing sets.</li>
  <li>Train <b>SVM</b> and optimize using <b>GridSearchCV</b>.</li>
  <li>Assess model accuracy and display confusion matrices.</li>
  <li>Use <b>PCA</b> to visualize decision boundary in lower dimensions.</li>
</ol>

<hr>

<h2>✨ Key Features</h2>
<ul>
  <li><b>Data Cleaning:</b> Handles missing/invalid values and ensures valid input.</li>
  <li><b>Imbalance Handling:</b> Downsamples classes for equal representation.</li>
  <li><b>Visualizations:</b> Class distributions, confusion matrices, PCA plots.</li>
  <li><b>SVM Optimization:</b> Hyperparameter tuning with cross-validation.</li>
  <li><b>PCA Visuals:</b> Plots variance explained and decision boundaries.</li>
</ul>

<hr>

<h2>📁 Dataset</h2>
<p>
  <b>Source:</b> <a href="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/eWtLeiKCjP9dCyP9AecgPA/default%20of%20credit%20card%20clients.xls">Default of Credit Card Clients Dataset</a><br>
  Main columns: <i>ID, LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0–PAY_6, BILL_AMT1–BILL_AMT6, PAY_AMT1–PAY_AMT6, default payment next month</i>
</p>

<hr>

<h2>🚦 How to Run</h2>
<ol>
  <li><b>Clone this repository</b> and ensure Python 3.8+ is installed.</li>
  <li><b>Install dependencies:</b>
    <pre><code>pip install pandas numpy scikit-learn seaborn matplotlib xlrd</code></pre>
  </li>
  <li><b>Run the script</b> in your favorite environment (e.g., Jupyter Notebook or IDE).</li>
</ol>

<h2>📊 Highlights</h2>
<ul>
  <li>Class distribution plots with <b>seaborn</b></li>
  <li>Confusion matrices before/after tuning</li>
  <li>PCA scree plot (variance explained)</li>
  <li>Decision boundary visualized for the first two PC components</li>
</ul>

<hr>

<h2>🧑💻 Contributing</h2>
<p>Pull requests and suggestions are very welcome! Please open an issue for discussion.</p>

<hr>

<h2>📚 License</h2>
<p>This project is licensed under the MIT License.</p>

<hr>

<p align="center"><b>Happy coding and exploration! 🚀</b></p>
