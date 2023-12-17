import pandas as pd
import pickle
import optuna
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

# Define la función objetivo para Optuna
def objective(trial, X_train, y_train, X_eval, y_eval):
    # Define los hiperparámetros a optimizar
    C = trial.suggest_loguniform('C', 0.1, 10)
    
    # Entrena un modelo de regresión logística con los hiperparámetros sugeridos
    model = LogisticRegression(C=C, max_iter=1000)  # Aumentar el número de iteraciones
    model.fit(X_train, y_train)
    
    # Realiza predicciones en el conjunto de evaluación
    predictions_eval = model.predict_proba(X_eval)[:, 1]
    
    # Calcula el Brier Score como métrica de rendimiento en evaluación
    brier_score_eval = brier_score_loss(y_eval, predictions_eval)
    
    return brier_score_eval

# Función para cargar los datos y ejecutar la optimización
def run_optimization():
    # Lee el dataset desde la nueva ubicación
    data = pd.read_csv("../../Data/Processed/selected_features_dataset_train.csv")
    
    # Separa la variable objetivo y las variables predictoras
    X = data.drop("Target", axis=1)
    y = data["Target"]
    
    # Divide los datos en conjunto de entrenamiento y evaluación (80/20)
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define la función de optimización y crea un estudio Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_eval, y_eval), n_trials=100)
    
    # Obtiene los mejores hiperparámetros
    best_params = study.best_params
    
    # Entrena un modelo con los mejores hiperparámetros
    best_model = LogisticRegression(C=best_params['C'], max_iter=1000)  # Aumentar el número de iteraciones
    best_model.fit(X_train, y_train)
    
    # Calcula el Brier Score en el conjunto de evaluación con el mejor modelo
    predictions_eval = best_model.predict_proba(X_eval)[:, 1]
    brier_score_eval = brier_score_loss(y_eval, predictions_eval)
    
    # Imprime el Brier Score en el conjunto de evaluación antes de guardar el modelo
    print(f"Brier Score en conjunto de evaluación: {brier_score_eval}")
    
    # Guarda el modelo en formato pickle
    with open('reglog_model.pkl', 'wb') as file:
        pickle.dump(best_model, file)

# Función main
if __name__ == "__main__":
    run_optimization()
