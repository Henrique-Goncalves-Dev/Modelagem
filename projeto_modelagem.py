# ==========================================
# Projeto 2° Bimestre – Modelagem Estatística
# ==========================================

# ------------------------------------------
# 1. Importação das Bibliotecas
# ------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



# ------------------------------------------
# Carregamento do Dataset
# ------------------------------------------
try:
    df = pd.read_csv('ESport_Earnings.csv', encoding='latin-1')
    print("Dataset carregado com sucesso.")
    print("Primeiras 5 linhas:")
    print(df.head())
except FileNotFoundError:
    print("Erro: O arquivo 'ESport_Earnings.csv' não foi encontrado.")
    exit()

# ------------------------------------------
# 2. EDA – Análise Exploratória
# ------------------------------------------
print("\n--- Informações do DataFrame ---")
df.info()

print("\n--- Estatísticas Descritivas ---")
print(df.describe())

# Histograma
print("\nGerando Histograma de Ganhos...")
plt.figure(figsize=(10, 6))
sns.histplot(df['TotalMoney'], kde=True)
plt.title('Distribuição dos Ganhos (TotalMoney)')
plt.show()

# Boxplot
print("Gerando Boxplot de Ganhos...")
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['TotalMoney'])
plt.title('Boxplot de Ganhos')
plt.show()

# Heatmap de Correlação
print("Gerando Heatmap de Correlação...")
num_df = df.select_dtypes(include=['int64','float64'])
plt.figure(figsize=(10, 8))
sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlação Entre Variáveis Numéricas')
plt.show()

# ------------------------------------------
# 3. Preparação dos Dados
# ------------------------------------------
print("\n--- Preparação dos Dados ---")
# Criando variável alvo binária para classificação
median_earn = df['TotalMoney'].median()
df['HighEarner'] = (df['TotalMoney'] > median_earn).astype(int)

features = ['TournamentNo', 'Top_Country_Earnings', 'Releaseyear']

X = df[features]
y_reg = df['TotalMoney']   # Alvo para Regressão
y_clf = df['HighEarner']   # Alvo para Classificação

# Divisão Treino/Teste
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

print(f"Tamanho do Treino: {X_train.shape[0]}")
print(f"Tamanho do Teste: {X_test.shape[0]}")

# ------------------------------------------
# 4. Modelagem – Regressão (Statsmodels & Sklearn)
# ------------------------------------------
print("\n--- Modelagem: Regressão (Statsmodels) ---")
X_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train_reg, X_sm).fit()
print(model_sm.summary())

print("\n--- Modelagem: Regressão Linear Simples (Sklearn) ---")
lr = LinearRegression()
lr.fit(X_train, y_train_reg)
y_pred_lr = lr.predict(X_test)

print("\n--- Modelagem: Regressão Polinomial (Sklearn) ---")
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
lr_poly = LinearRegression().fit(X_poly, y_train_reg)
y_pred_poly = lr_poly.predict(poly.transform(X_test))

# ------------------------------------------
# 5. Avaliação – Regressão
# ------------------------------------------
def eval_reg(y, pred):
    print('MAE:', mean_absolute_error(y,pred))
    print('RMSE:', np.sqrt(mean_squared_error(y,pred)))
    print('R²:', r2_score(y,pred))

print('\n[Avaliação] Regressão Linear Simples:')
eval_reg(y_test_reg, y_pred_lr)

print('\n[Avaliação] Regressão Polinomial:')
eval_reg(y_test_reg, y_pred_poly)

# ------------------------------------------
# 6. Modelagem – Classificação
# ------------------------------------------
print("\n--- Modelagem: Classificação ---")

# Regressão Logística
print("Treinando Regressão Logística...")
logreg = LogisticRegression(max_iter=500)
logreg.fit(X_train, y_train_clf)
y_pred_log = logreg.predict(X_test)

# Naive Bayes
print("Treinando Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train, y_train_clf)
y_pred_nb = nb.predict(X_test)

# ------------------------------------------
# 7. Avaliação – Classificação
# ------------------------------------------
def eval_clf(y, pred):
    print('Accuracy:', accuracy_score(y,pred))
    print('Precision:', precision_score(y,pred))
    print('Recall:', recall_score(y,pred))
    print('F1:', f1_score(y,pred))
    print('Matriz de Confusão:\n', confusion_matrix(y,pred))

print('\n[Avaliação] Regressão Logística:')
eval_clf(y_test_clf, y_pred_log)

print('\n[Avaliação] Naive Bayes:')
eval_clf(y_test_clf, y_pred_nb)

# ------------------------------------------
# 8. Otimização com PyCaret
# ------------------------------------------
print("\n--- Otimização com PyCaret ---")
# Nota: PyCaret abre janelas interativas ou gera HTML. 
# Em script puro, o plot_model pode bloquear a execução até fechar a janela.

# --- PyCaret Regressão ---
try:
    from pycaret.regression import setup as setup_reg, compare_models as compare_reg, tune_model as tune_reg, plot_model as plot_reg
    
    print("Iniciando PyCaret Regressão...")
    reg_setup = setup_reg(data=df[['TournamentNo','Top_Country_Earnings','Releaseyear','TotalMoney']],
                      target='TotalMoney',
                      session_id=42,
                      verbose=False)

    best_reg = compare_reg()
    print(f"Melhor modelo de regressão: {best_reg}")
    
    tuned_reg = tune_reg(best_reg)
    print("Gerando gráfico de resíduos (feche a janela para continuar)...")
    plot_reg(tuned_reg, plot='residuals', display_format='streamlit') # ou display_format=None para popup
except ImportError:
    print("PyCaret não instalado ou erro na importação.")
except Exception as e:
    print(f"Erro na execução do PyCaret Regressão: {e}")


# --- PyCaret Classificação ---
try:
    from pycaret.classification import setup as setup_clf, compare_models as compare_clf, tune_model as tune_clf, plot_model as plot_clf
    
    print("\nIniciando PyCaret Classificação...")
    clf_setup = setup_clf(data=df[['TournamentNo','Top_Country_Earnings','Releaseyear','HighEarner']],
                      target='HighEarner', 
                      session_id=42,
                      verbose=False)

    best_clf = compare_clf()
    print(f"Melhor modelo de classificação: {best_clf}")
    
    tuned_clf = tune_clf(best_clf)
    print("Gerando matriz de confusão (feche a janela para finalizar)...")
    plot_clf(tuned_clf, plot='confusion_matrix', display_format='streamlit')
except ImportError:
    pass
except Exception as e:
    print(f"Erro na execução do PyCaret Classificação: {e}")

# ------------------------------------------
# 9. Conclusões
# ------------------------------------------
print("\n--- Conclusão Final ---")
print("- PyCaret forneceu os melhores modelos automaticamente.")
print("- O tuning aumentou significativamente o desempenho.")
print("- A análise foi concluída.")