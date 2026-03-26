import numpy as np

alphas = [0.001,0.01,0.1,1,10,100,1000]
hiddenSize = 32
# calculer la fonction non lineaire sigmoid
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# calculer la derivee de la fonction sigmoid 
def sigmoid_output_to_derivative(output):
    return output*(1-output)
    
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])

for alpha in alphas:
    print ("\nTraining With Alpha:" + str(alpha))
    np.random.seed(1)

     # initialiser les poids
    synapse_0 = 2*np.random.random((3,hiddenSize)) - 1
    synapse_1 = 2*np.random.random((hiddenSize,1)) - 1

    for j in range(60000):

        # Propagation avant
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0,synapse_0))
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        # Fonction de cout
        layer_2_error = layer_2 - y

        if (j% 10000) == 0:
            print ("Error after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))))

        # trouver est dans quelle direction la valeur voulue?
        layer_2_delta = layer_2_error*sigmoid_output_to_derivative(layer_2)

         # Combien la valeur de la couche 1 contribue a l'erreur dans la couche 2  (selon les poinds)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

       # trouver est dans quelle direction la valeur voulue?
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
        synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))
