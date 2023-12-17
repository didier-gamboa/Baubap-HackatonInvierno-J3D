import pandas as pd
import pickle
from sklearn.metrics import brier_score_loss

def evaluate_model(model_path, data_path, output_path):
    # Carga el modelo entrenado
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # Carga el conjunto de datos de evaluación
    eval_data = pd.read_csv(data_path)
    
    # Guarda las variables predictoras
    X_eval = eval_data.copy()
    if 'Target' in X_eval.columns:
        X_eval = X_eval.drop('Target', axis=1)
    
    # Realiza predicciones en el conjunto de evaluación
    predictions_eval = model.predict_proba(X_eval)[:, 1]
     
    # Agrega las predicciones al conjunto de datos en la columna 'Target'
    eval_data['Target'] = predictions_eval
    
    # Guarda los datos con las predicciones en un archivo CSV
    eval_data.to_csv(output_path, index=False)

def main():
    model_path = 'reglog_model.pkl'
    data_path = '../../Data/Processed/selected_features_dataset_evaluate.csv'
    output_path = '../../Data/Processed/selected_features_dataset_predicted.csv'
    
    evaluate_model(model_path, data_path, output_path)

if __name__ == "__main__":
    main()
