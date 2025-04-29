# 📩 SMS Spam Classifier

This project is a machine learning-based SMS Spam Classifier that detects whether a given SMS message is **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques and a Logistic Regression model. It demonstrates the end-to-end ML pipeline including data preprocessing, training, testing, and real-time message prediction.

---

## 🚀 Features

- ✅ Clean text using regex and stopwords
- ✅ TF-IDF feature extraction
- ✅ Binary classification using Logistic Regression
- ✅ Performance evaluation with accuracy, confusion matrix, and classification report
- ✅ CLI-based custom message testing
- ✅ Lightweight and easy to run on local machine
- ✅ Modular and readable Python code

---

## 📁 Dataset

The project uses the **SMS Spam Collection Dataset**, which is a public set of SMS labeled messages (ham or spam) collected for text classification.

- Dataset link: [SMS Spam Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Format: CSV with columns for message content and label
- Labels: `ham` for legitimate messages and `spam` for unwanted or promotional messages

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sms-spam-classifier.git
cd sms-spam-classifier
```

### 2. Create and Activate Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Stopwords

```python
import nltk
nltk.download('stopwords')
```

---

## 📄 Code Usage

Run the spam classifier script:

```bash
python app.py
```
if want run model enter on bash:
 python model.py

You will be prompted to enter a message:

```bash
Enter a message to classify: Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005!
The message is: Spam
```

---

## 🧠 How It Works

1. **Text Preprocessing**
   - Lowercasing
   - Removing numbers, punctuation, and stopwords

2. **TF-IDF Vectorization**
   - Converts text into numerical vectors that represent word importance

3. **Model Training**
   - A `Logistic Regression` classifier learns to distinguish between spam and non-spam

4. **Prediction**
   - The model predicts spam (1) or not spam (0) based on input

---

## 🧪 Model Performance

| Metric       | Score    |
|--------------|----------|
| Accuracy     | ~98%     |
| Precision    | High     |
| Recall       | High     |
| F1-Score     | Balanced |

Model performs well in both identifying spam and avoiding false positives.

---

## 📌 File Structure

```
├── spam.csv                # Input dataset
├── spam_classifier.py      # Main Python script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 💬 FAQs

**Q: Can I use this with a different dataset?**  
Yes, but ensure the new dataset has a similar structure with 'label' and 'message' columns.

**Q: Can this model run on mobile/web?**  
Yes. The trained model can be exported and integrated into a Flask app, web service, or mobile app.

**Q: How do I improve accuracy?**  
You can try advanced models like RandomForest, XGBoost, or fine-tune using GridSearchCV.

---

## 🧑‍💻 Contributing

Contributions are welcome!

1. Fork this repo
2. Create your feature branch: `git checkout -b feature/feature-name`
3. Commit your changes: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature/feature-name`
5. Create a pull request

---

## 🧾 License

This project is licensed under the MIT License.

---

## 🙋‍♀️ Acknowledgments

- UCI Machine Learning Repository
- Kaggle Datasets
- scikit-learn & NLTK libraries
