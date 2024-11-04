# Development-of-an-algorithm-for-detecting-malicious-links-on-the-Instagram
 
Description

This project is a Python system for classifying URLs as safe or malicious using machine learning models. The system is trained to check Instagram links and detect potentially phishing or malicious URLs. The basic rule of verification is if the URL does not contain `instagram.com `, then it is considered potentially malicious, but additional training helps to identify more complex threats. The code will be much more improved in the future.

 Project structure

- Link classification: Two models, a Random forest and a Neural Network, are used to classify URLs.
- Data preprocessing: URL characteristics are extracted and analyzed to create a training sample.
- Real-time implementation: You can check any link and immediately get a warning about its security.

 Installation

 Dependencies

Before you start, make sure that you have the necessary dependencies installed. You can install them using the command:

```bash
pip install -r requirements.txt
```

File `requirements.txt ` must contain:

```text
numpy
pandas
scikit-learn
imblearn
tensorflow
joblib
```

 Launch

1. Clone the repository:

   ```bash
   git clone https://github.com/username/instagram-url-safety-checker.git
   ```

2. Go to the project folder:

   ```bash
   cd instagram-url-safety-checker
   ```

3. Make sure you have the `instagram_links.csv` file with the training data. The file should contain two columns:
- `url': URL for analysis.
   - `label': A label where `1` indicates a malicious link and `0` indicates a secure one.

 Using

 The Main Script

Run `main.py ` to train models and verify provided links:

```bash
python main.py
```

 Conclusion

The script outputs the following data:

- Evaluation of models: Accuracy, completeness, F1-measure, ROC-AUC.
- URL verification results: Shows for each link the probability of a threat and the result of model verification (safe or malicious).

 Real-time URL verification

You can add your verification URLs to the script. In the section `if __name__ == "__main__": From the `urls_to_check` list, you can add the links you need.

```python
urls_to_check = [
    'https://instagram.com/username',
    'http://fake-instagram.com/login'
]
```

The prediction results for each link will be displayed in the console.

 The logic of threat detection

- Any URL that does not contain `instagram.com `, is automatically marked as potentially malicious.
- For URLs with `instagram.com ` models are used to assess the likelihood of a threat.

 Saving the model

Data preprocessing models and objects are saved for reuse:
- `rf_model.pkl`: Random Forest model
- `nn_model.keras': Neural network
- `scaler.pkl` and `vectorizer.pkl`: data preprocessing objects

 Features

- Class imbalance handling: `RandomOverSampler` is used to increase the sample of a smaller class.
- Ease of use: The code is modular, adaptable and supports a variety of data.
- Real-time support: Instant detection of potentially dangerous URLs.

 Notes

- If you plan to deploy the system in production, it is recommended to add additional checks and protections when processing user URLs.
- Make sure you are using the latest version of tensorflow to avoid outdated warnings.

 Contacts

If you have any questions, please contact Aisultan Aitmaganbet (mail: ais.aitmaganbet@gmail.com ).
