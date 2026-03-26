import numpy as np

#calculer la fonction non lineaire sigmoid
def sigmoid (x):
    output = 1 / (1 + np.exp(-x))
    return output

    #convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative (output):
    return output * (1 - output)

#dataset d'entree

x = np.array([[0,1],
              [0,1],     
              [1,0],
              [1,0]])

#dataset de sortie
y = np.array([[0, 0 , 1 ,1]]) .T #transpose pour faire correspondre les dimensions de y avec celles de la sortie du réseau   


# une seul la nerone de sortie soit 0 soit 1 et entres x ou y et chaqucun soit 1 ou 0


#iNITIALISATION DES POIDS PAS DE 0 ? DEVALEURS ALEATOIRE MAIS PROCHE DE 0
np.random.seed(1) #pour que les résultats soient reproductibles

#iNITIALISATION DES POIDS
synapse_0 = 2 * np.random.random((2, 1)) - 1 #poids entre la couche d'entrée et la couche cachée (2 neurones d'entrée et 3 neurones cachés)
print(f"Poids apres l'initalisation  : \n {synapse_0}\n")


for iter in rage (10000):
    # propagation vers l'avant
    layer_0 = x #couche d'entrée
    layer_1 = sigmoid(np.dot(layer_0, synapse_0)) #couche cachée

    # fonction de cout (erreur)
    layer_1_error = layer_1 - y #erreur de la couche cachée par rapport à la sortie attendue


    # --- ÉTAPE 3 : CALCUL DE LA PENTE DE LA FONCTION SIGMOID ---
    layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
    synapse_0_derivated = np.dot(layer_0.T, layer_1_delta) # gradient : layer_1_deltat

    # --- ÉTAPE 4 : MISE A JOUR DES POIDS ---
    synapse_0 -=  synapse_0_derivated # mise à jour des poids en soustrayant le gradient
