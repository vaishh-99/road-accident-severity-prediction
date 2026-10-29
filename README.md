
# AI-Based Road Accident Severity Prediction

## Project Objective
This project uses machine learning to predict an accident severity
category from historical road accident records.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Joblib

## Dataset
Download ETP_4_New_Data_Accidents.csv from:
https://zenodo.org/records/16946653

Place the file inside the data folder.

## How to Run

1. Install Python 3.10 or later.
2. Open the project folder in VS Code.
3. Open a terminal in the project folder.
4. Create a virtual environment:

   python -m venv .venv

5. Activate the environment on Windows CMD:

   .venv\Scripts\activate

6. Install the libraries:

   python -m pip install -r requirements.txt

7. Train and evaluate the model:

   python train_model.py

8. Start the web application:

   streamlit run app.py

9. Open the local URL shown in the terminal.

## Output Files
- models/accident_severity_model.joblib
- models/metadata.json
- outputs/dataset_summary.json
- outputs/evaluation_metrics.json
- outputs/severity_distribution.png
- outputs/confusion_matrix.png

## Limitation
This model predicts severity for recorded accidents.
It does not directly predict the likelihood of an accident occurring
at a location and time where no accident has been recorded.
