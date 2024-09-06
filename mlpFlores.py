from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def carregar_base_dados():
    base_dados = load_iris()
    return base_dados.data, base_dados.target

def separar_conjuntos(caracteristicas, rotulos, proporcao_teste=0.3, estado_aleatorio=42):
    return train_test_split(caracteristicas, rotulos, test_size=proporcao_teste, random_state=estado_aleatorio)

def padronizar_conjuntos(treino_caracteristicas, teste_caracteristicas):
    escalonador = StandardScaler()
    treino_normalizado = escalonador.fit_transform(treino_caracteristicas)
    teste_normalizado = escalonador.transform(teste_caracteristicas)
    return treino_normalizado, teste_normalizado

def treinar_mlp(treino_caracteristicas, treino_rotulos, camadas_ocultas=(10, 10), iteracoes_max=1000, estado_aleatorio=42):
    classificador_mlp = MLPClassifier(hidden_layer_sizes=camadas_ocultas, max_iter=iteracoes_max, random_state=estado_aleatorio)
    classificador_mlp.fit(treino_caracteristicas, treino_rotulos)
    return classificador_mlp

def calcular_acuracia(classificador, teste_caracteristicas, teste_rotulos):
    predicoes = classificador.predict(teste_caracteristicas)
    return accuracy_score(teste_rotulos, predicoes)

def principal():
    dados_caracteristicas, dados_rotulos = carregar_base_dados()
    treino_caracteristicas, teste_caracteristicas, treino_rotulos, teste_rotulos = separar_conjuntos(dados_caracteristicas, dados_rotulos)
    
    treino_normalizado, teste_normalizado = padronizar_conjuntos(treino_caracteristicas, teste_caracteristicas)
    
    modelo_mlp = treinar_mlp(treino_normalizado, treino_rotulos)
    
    acuracia_final = calcular_acuracia(modelo_mlp, teste_normalizado, teste_rotulos)
    print(f"Acurácia: {acuracia_final * 100:.2f}%")

if __name__ == "__main__":
    principal()


""" Biblioteca Scikit-learn
    Ela oferece uma vasta coleção de ferramentas para modelagem preditiva, 
    abrangendo desde regressão e classificação até clustering
    e redução de dimensionalidade. """
