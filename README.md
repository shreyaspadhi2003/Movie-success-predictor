# Movie Success Predictor

A machine learning-powered application that predicts the IMDb rating of a movie and classifies it as a “Hit” or “Flop” using metadata such as actors, directors, genres, runtime, and release year. Built with support for both regression and classification models, integrated SHAP explainability, and a user-friendly Streamlit frontend.

---

## Project Structure

movie-success-predictor/ 
├── data/

│ ├── titles.csv

│ └── credits.csv

├── utils/ 

│ ├── preprocess.py 

│ └── evaluate.py 

├── models/ 

│ ├── random_forest.py

│ └── xgboost.py 

├── app.py 

├── train.py 

└── README.md

---

##  Features

- Predicts IMDb score using regression
- Classifies a movie as *Hit* or *Flop*
- Multi-model support: Random Forest, XGBoost
- Handles partial or full metadata input
- SHAP explainability for feature influence
- Streamlit UI for interactive predictions
- Auto-saves confusion matrix and performance report

---

##  Setup Instructions

1. **Clone the Repository**

git clone https://github.com/shreyaspadhi2003/Movie-success-predictor.git

cd movie-success-predictor

2. Install Dependencies

bash
Copy
pip install -r requirements.txt
Required packages include: pandas, numpy, scikit-learn, xgboost, shap, streamlit, matplotlib, joblib

3. Add Data Files

Place your titles.csv and credits.csv inside the data/ folder.

Train the Models
Run the training script to generate all models and evaluation outputs:

python train.py

This will:

Train regression and classification models

Save models and SHAP explainers in models/

Output confusion_matrix.png and performance_metrics.txt

Run the Streamlit App
Launch the UI to start making predictions:

streamlit run app.py

You can input:

. One or more actors

. One or more directors

. Multiple genres

. Runtime and release year

Then view:

. IMDb score prediction

. Hit/Flop classification

. SHAP-based feature influence plots

. Actor/director performance averages

Evaluation Outputs:
After training, the following files will be generated in /models:

confusion_matrix.png: Visual confusion matrix of classification performance

performance_metrics.txt: Detailed classification and regression metrics

Pickled model files for both XGBoost and Random Forest

Future Improvements:
Add trailer/video-based sentiment features

Incorporate release season and production budget

Deploy app with Docker or on cloud (e.g., Streamlit Cloud / Heroku)

Implement user authentication and saved prediction history

Contact
For questions or collaborations, reach out at:
shreyaspadhi2003@gmail.com
https://www.linkedin.com/in/shreyas-padhi-01100b221/

License
This project is open-source under the MIT License.

Copy

---

Just copy and paste this into a `README.md` file in your project directory. Let me know if you need further assistance!
