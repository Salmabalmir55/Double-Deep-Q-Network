import numpy as np 
import matplotlib.pyplot as plt

nombre_neurones = range(4,33, 4)
liste_erreur=[]

# Calculer la fonction non lineaire sigmoid 
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# Calculer la derivee 
def sigmoid_output_to_derivative(output):
    return output*(1-output)

# dataset d entree
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

for nb in nombre_neurones:
    print(" Entrainement avec la valeur de nombre de neurones de:"+ str(nb))
    # Reproductibilité des résultats
    np.random.seed(1)

    # initialiser les poids 
    synapse_0 = 2*np.random.random((3,nb)) - 1
    synapse_1 = 2*np.random.random((nb,1)) - 1

    for iter in range(60000):

    # propagation avant
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0,synapse_0))
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        layer_2_error = layer_2 - y

        if(iter % 10000) == 0:
            print ("Erreur apres "+ str(iter)+ "iterations"+ str(np.mean(np.abs(layer_2_error ))))

        # calcule de la pente 
        layer_2_delta= layer_2_error * sigmoid_output_to_derivative(layer_2)
        layer_1_error = layer_2_delta.dot(synapse_1.T)
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        # Ajuster les poids 

        synapse_1 -= 10 * (layer_1.T.dot(layer_2_delta)) 
        synapse_0 -= 10 * (layer_0.T.dot(layer_1_delta)) 

    erreur_final = np.mean(np.abs(layer_2_error ))
    print("erreur final avec "+str(nb)+" "+ str(erreur_final))

    liste_erreur.append(erreur_final)

#tracer un courbe 
plt.figure( figsize=(8,5))
plt.plot(nombre_neurones,liste_erreur, marker='o',color='b',label='Fonction de cout')
plt.xlabel("Nombre de neurones dan la couche cachee")
plt.xlabel("la valeur absolue de l erreur trouvee")
plt.title(" Fonction de cout selon le nombre de neurones")
plt.legend()
plt.grid()
plt.show()




      


