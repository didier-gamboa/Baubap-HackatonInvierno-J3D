import pandas as pd
import pickle
from sklearn.metrics import brier_score_loss

def evaluate_model(model_path, data_path, output_path):
    # Carga el modelo entrenado
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # Carga el conjunto de datos de evaluación
    eval_data = pd.read_csv(data_path)
    
    # Separa la variable objetivo y las variables predictoras
    X_eval = eval_data.drop("Target", axis=1)
    y_eval = eval_data["Target"]
    
    # Realiza predicciones en el conjunto de evaluación
    predictions_eval = model.predict_proba(X_eval)[:, 1]
    
    # Calcula el Brier Score en el conjunto de evaluación
    brier_score_eval = brier_score_loss(y_eval, predictions_eval)
    print(f"Brier Score en conjunto de evaluación: {brier_score_eval}")
    
    # Guarda las probabilidades predichas en un archivo CSV
    eval_data['Predicted_Probability'] = predictions_eval
    eval_data.to_csv(output_path, index=False)

def main():
    model_path = 'reglog_model.pkl'
    data_path = '../../Data/Processed/selected_features_dataset_evaluate.csv'
    output_path = '../../Data/Processed/selected_features_dataset_predicted.csv'
    
    evaluate_model(model_path, data_path, output_path)

if __name__ == "__main__":
    main()
