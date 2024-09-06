import matplotlib.pyplot as plt 
import networkx as nx 
import heapq


class RomeniaPonderado:
    def __init__(self):
        
        self.G = nx.Graph()

        
        edges = [
            ("Arad", "Zerind", 75),
            ("Arad", "Sibiu", 140),
            ("Arad", "Timisoara", 118),
            ("Zerind", "Oradea", 71),
            ("Oradea", "Sibiu", 151),
            ("Timisoara", "Lugoj", 111),
            ("Lugoj", "Mehadia", 70),
            ("Mehadia", "Drobeta", 75),
            ("Drobeta", "Craiova", 120),
            ("Sibiu", "Fagaras", 99),
            ("Sibiu", "Rimnicu Vilcea", 80),
            ("Rimnicu Vilcea", "Pitesti", 97),
            ("Rimnicu Vilcea", "Craiova", 146),
            ("Craiova", "Pitesti", 138),
            ("Fagaras", "Bucharest", 211),
            ("Pitesti", "Bucharest", 101),
            ("Giurgiu", "Bucharest", 90),
            ("Bucharest", "Urziceni", 85),
            ("Urziceni", "Hirsova", 98),
            ("Urziceni", "Vaslui", 142),
            ("Vaslui", "Iasi", 92),
            ("Iasi", "Neamt", 87),
            ("Hirsova", "Eforie", 86)
        ]

        
        for edge in edges:
            self.G.add_edge(edge[0], edge[1], weight=edge[2])

    def imprimir(self):
        
        for u, v, weight in self.G.edges(data=True):
            print(f"De {u} para {v} com distância {weight['weight']} km")

    def plotar(self):
        
        pos = nx.spring_layout(self.G)  
        nx.draw(self.G, pos, with_labels=True, node_color='lightblue', edge_color='#909090', node_size=500, font_size=8,
                font_weight='bold')

       
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)

        plt.title("Mapa de Cidades da Romênia com Distâncias")
        plt.show()
def busca_melhor(self, inicio, objetivo):
    fronteira =[]
    heapq.heappush(fronteira, (self.heuristica[inicio], inicio))

    visitados = set()

    passados = {inicio: None}

    while fronteira:
            
            _, atual = heapq.heappop(fronteira)

            if atual == objetivo:
                
                caminho = []
                while atual is not None:
                    caminho.append(atual)
                    atual = passados[atual]
                caminho.reverse()
                return caminho

            visitados.add(atual)

            for vizinho in self.G.neighbors(atual):
                if vizinho not in visitados:
                    heapq.heappush(fronteira, (self.heuristicas[vizinho], vizinho))
                    if vizinho not in passados:
                        passados[vizinho] = atual

            return None  


romenia = RomeniaPonderado()


caminho = romenia.busca_melhor_escolha("Arad", "Bucharest")
print("Caminho encontrado:", caminho)

