# WaterPotability

## Demo app

Launch the app [![Open In Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/georgiosdolias/waterpotability/main/WaterPotApp.py)

## App info

This repository creates a Data Web App  about safe water for human consumption by using Streamlit Python Package. More specifically, it allows its users to change the values of the nine predictor variables:
1. pH
2. Hardness
3. Solids
4. Chloramines
5. Sulfate
6. Conductivity
7. Organic_carbon
8. Trihalomethanes
9. Turbidity

trains a Random Forest Classifier model and, finally, observe the prediction of the trained model.

## Dataset

The imported csv file contains water quality metrics for 3276 different water bodies.

## Reproducing the App

1. First, we create a virtual Python environment called my_venv
```
  python3 -m venv my_venv
```
2. Then, we activate the virtual environment
```
source path_to_your_virtual_environment/bin/activate
```
3. After getting to the virtual environment's file, install prerequisite packages
```
wget https://raw.githubusercontent.com/GeorgiosDolias/WaterPotability/main/requirements.txt
```
and
```
pip install -r requirements.txt
```
4. Dowload and unzip contents from Github repo

Dowload and unzip contents from https://github.com/GeorgiosDolias/WaterPotability/archive/main.zip

5. Launch the app
```
streamlit run WaterPotApp.py
```


## Requirements

| Package | Version |
--- | ---
| streamlit | 0.87.0 |
| pandas |  1.1.3 |
| sci-kit learn | 0.23.2 |
| numpy |  1.19.1 |

## Useful Resources

1.  [Youtube tutorial from Chanin Nantasenamat (Data Professor) ](https://www.youtube.com/watch?v=8M20LyCZDOY )
2.  [More info about the used dataset on Kaggle](https://www.kaggle.com/adityakadiwal/water-potability ) 
