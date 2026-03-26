import numpy as np
#Exemple de données d'entrée ( nbr de compte et nbr d'enfants)
input_data = np.array ([2,3])   

# exemple de poinds pour chaque noeud de la couche cachée de la couche de sortie


weights = {
    'node_0' : np.array ([1,1]), #5 poid allant dans le noeud_0 (pour inpu_data[0] et  input_data[1]
    'node_1' : np.array ([-1,1]),    # poid allant dans le noeud_1 (pour inpu_data[0] et  input_data[1]
    'output' : np.array ([2,-1])   # poid allant de la couche cachée vers le noeud de sortie 
}
# --- ÉTAPE 1 : Calcul de la couche cachée ---
# On multiplie l'entrée par les poids de chaque nœud
node_0_input = (input_data * weights['node_0']).sum() # (2*1 + 3*1) = 5
node_1_input = (input_data * weights['node_1']).sum() # (2*-1 + 3*1) = 1

# On regroupe les résultats dans un tableau
hidden_layer_outputs = np.array([node_0_input, node_1_input])

# --- ÉTAPE 2 : Calcul de la sortie finale ---
# On multiplie les résultats de la couche cachée par les poids de sortie
output = (hidden_layer_outputs * weights['output']).sum() # (5*2 + 1*-1) = 9

print(f"Sortie finale : {output}")
