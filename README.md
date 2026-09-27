# SuperKart Sales Prediction Project

This project builds a machine learning pipeline to predict the sales total for SuperKart products using retail/store features.

## Project structure

- `data/` - raw and processed datasets
- `notebooks/` - exploratory notebooks
- `src/` - training and preparation scripts
- `app/` - Streamlit deployment app
- `.github/workflows/` - GitHub Actions CI pipeline
- `Dockerfile` - container definition for deployment

## Workflow

1. Explore and clean the dataset
2. Split into train/test sets
3. Train a regression model
4. Evaluate with RMSE, MAE and R2
5. Save the model
6. Create a Streamlit app for prediction

## Run locally

```bash
python src/prepare_data.py
python src/train_model.py
streamlit run app/app.py
```
