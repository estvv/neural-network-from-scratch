import math, random

def relu(x: float) -> float:
    """
        Utilisation:
            - ReLU est largement utilisée dans les réseaux de neurones pour introduire de la non-linéarité tout en étant simple à calculer.

        Input:
            x: float

        Formule de la fonction d'activation ReLU :
            ReLU(x) = max(0, x)

        Output:
            Le résultat de la fonction d'activation ReLU appliquée à x
                - ReLU(x) = max(0, x)
    """
    return max(0, x)

def sigmoid(x: float) -> float:
    """
        Utilisation:
            - La fonction Sigmoid est utilisée pour convertir une valeur en une probabilité entre 0 et 1, ce qui est particulièrement utile pour les tâches de classification binaire.
            - Elle est souvent utilisée dans la couche de sortie des réseaux de neurones pour des problèmes de classification binaire.

        Input:
            x: float

        Formule de la fonction d'activation Sigmoid :
            Sigmoid(x) = 1 / (1 + exp(-x))

        Output:
            Le résultat de la fonction d'activation Sigmoid appliquée à x
                - Sigmoid(x) = 1 / (1 + exp(-x))
    """
    return 1 / (1 + math.exp(-x))

def forward(x: list[float], W1: list[list[float]], b1: list[float], W2: list[float], b2: float) -> tuple[list[float], float]:
    """
        Input:
            x: list[float]        : un point d'entraînement (2 features)
            W1: list[list[float]] : poids de la couche cachée (3x2)
            b1: list[float]       : biais de la couche cachée (3,)
            W2: list[float]       : poids de la couche de sortie (3,)
            b2: float             : biais de la couche de sortie

        Formules :
            Forward pass:
                - ReLU(W1 * x + b1)  # activations cachées
                - Sigmoid(W2 * h + b2)  # probabilité finale

        Output:
            h     : list[float]  — les 3 activations cachées
            y_hat : float        — probabilité finale (0..1)
    """

    # Un neurone caché est une somme pondérée des entrées plus un biais, passée par une fonction d'activation

    # Pour chaque neurone caché j : z1[j] = Σ W1[j][i]*x[i] + b1[j]
    z1 = [sum(W1[j][i] * x[i] for i in range(len(x))) + b1[j] for j in range(len(b1))]

    # h[j] = ReLU(z1[j])
    h = [relu(z) for z in z1]

    # Neurone de sortie : z2 = Σ W2[j]*h[j] + b2
    z2 = sum(W2[j] * h[j] for j in range(len(W2))) + b2

    # y_hat = sigmoid(z2) — probabilité entre 0 et 1
    y_hat = sigmoid(z2)

    return (h, y_hat)

def loss(y_hats: list[float], ys: list[int]) -> float:
    """
        Input:
            y_hats: list[float] : les probabilités prédites par le modèle pour chaque point d'entraînement
            ys: list[int]       : les étiquettes réelles (0 ou 1) pour chaque point d'entraînement

        Formules :
            Binary Cross-Entropy Loss :
                - L = - (1/N) * Σ [y_i * log(y_hat_i) + (1 - y_i) * log(1 - y_hat_i)]

        Output:
            L: float: la valeur de la perte calculée à partir des prédictions et des étiquettes réelles
    """

    # Calcul de la perte binaire
    N = len(y_hats)
    loss = 0.0

    # Calcul de la somme des pertes pour chaque point d'entraînement
    for y, y_hat in zip(ys, y_hats):
        loss += y * math.log(y_hat) + (1 - y) * math.log(1 - y_hat)

    # Retourne la perte moyenne négative
    return -(1/N) * loss

def gradients(xs: list[list[float]], ys: list[int], W1: list[list[float]], b1: list[float], W2: list[float], b2: float) -> tuple[list[list[float]], list[float], list[float], float]:
    """
        Utilisation:
            - Cette fonction calcule les gradients des poids et des biais du réseau de neurones à partir des données d'entraînement et des paramètres actuels du modèle.
            - Ces gradients sont essentiels pour la mise à jour des poids lors de l'entraînement du modèle.

        Input:
            xs: list[list[float]] : les points d'entraînement (N x 2)
            ys: list[int] : les étiquettes réelles (0 ou 1) pour chaque point d'entraînement
            W1: list[list[float]] : poids de la couche cachée (3x2)
            b1: list[float] : biais de la couche cachée (3,)
            W2: list[float] : poids de la couche de sortie (3,)
            b2: float : biais de la couche de sortie

        Formule du calcul des gradients (backpropagation) :
            1. Calcul du forward pass pour obtenir les activations cachées et les prédictions
            2. Calcul de la perte
            3. Calcul des gradients pour W2, b2, W1, b1

        Output:
            dW1: list[list[float]] : gradients pour les poids de la couche cachée (3x2)
            db1: list[float] : gradients pour les biais de la couche cachée (3,)
            dW2: list[float] : gradients pour les poids de la couche de sortie (3,)
            db2: float : gradient pour le biais de la couche de sortie
    """
    N = len(xs)

    n_in     = len(xs[0])   # nb features par point
    n_hidden = len(b1)      # nb neurones cachés

    # Accumulateurs — même forme que les poids, initialisés à zéro
    dW1 = [[0.0] * n_in for _ in range(n_hidden)]
    db1 = [0.0] * n_hidden
    dW2 = [0.0] * n_hidden
    db2 = 0.0

    for x, y in zip(xs, ys):

        # ── FORWARD (on a besoin de z1 pour la dérivée de ReLU) ──────────
        z1    = [sum(W1[j][i] * x[i] for i in range(n_in)) + b1[j] for j in range(n_hidden)]
        h     = [relu(z) for z in z1]
        z2    = sum(W2[j] * h[j] for j in range(n_hidden)) + b2
        y_hat = sigmoid(z2)

        # ── BACKWARD ─────────────────────────────────────────────────────

        # Étape 1 — erreur en sortie
        # propriété de sigmoid + BCE : la dérivée vaut simplement y_hat - y
        delta_out = y_hat - y

        # Étape 2 — gradients de W2 et b2
        # chaque poids W2[j] a contribué delta_out * h[j] à l'erreur
        for j in range(n_hidden):
            dW2[j] += delta_out * h[j]
        db2 += delta_out

        # Étape 3 — propager l'erreur vers la couche cachée
        # chaque neurone caché j a contribué delta_out * W2[j]
        delta_h = [delta_out * W2[j] for j in range(n_hidden)]

        # Étape 4 — passer par la dérivée de ReLU
        # ReLU bloque les neurones à 0 : dérivée = 1 si z1[j] > 0, sinon 0
        delta_z1 = [delta_h[j] * (1.0 if z1[j] > 0 else 0.0) for j in range(n_hidden)]

        # Étape 5 — gradients de W1 et b1
        # W1[j][i] a contribué delta_z1[j] * x[i] à l'erreur
        for j in range(n_hidden):
            for i in range(n_in):
                dW1[j][i] += delta_z1[j] * x[i]
            db1[j] += delta_z1[j]

    # Moyenne sur N points
    dW1 = [[dW1[j][i] / N for i in range(n_in)] for j in range(n_hidden)]
    db1 = [db1[j] / N for j in range(n_hidden)]
    dW2 = [dW2[j] / N for j in range(n_hidden)]
    db2 = db2 / N

    return dW1, db1, dW2, db2

def train(xs: list[list[float]], ys: list[int], lr: float, epochs: int) -> list[tuple]:
    """
        Input:
            xs: list[list[float]] : les points d'entraînement (N x 2)
            ys: list[int] : les étiquettes réelles (0 ou 1) pour chaque point d'entraînement
            lr: float : taux d'apprentissage
            epochs: int : nombre d'itérations d'entraînement

        Formule de l'entraînement :
            1. Initialisation des poids W1, b1, W2, b2
            2. Pour chaque epoch :
                a. Calcul des gradients dW1, db1, dW2, db2
                b. Mise à jour des poids et biais

        Output:
            history: list[tuple] : une liste contenant les paramètres du modèle et la perte à chaque epoch pour analyse ultérieure
    """
    W1 = [[random.uniform(-1, 1) for _ in range(len(xs[0]))] for _ in range(3)]
    b1 = [random.uniform(-1, 1) for _ in range(3)]
    W2 = [random.uniform(-1, 1) for _ in range(3)]
    b2 = random.uniform(-1, 1)

    history = []

    for epoch in range(epochs):
        dW1, db1, dW2, db2 = gradients(xs, ys, W1, b1, W2, b2)

        for j in range(len(W1)):
            for i in range(len(W1[0])):
                W1[j][i] -= lr * dW1[j][i]

            b1[j] -= lr * db1[j]

        for j in range(len(W2)):
            W2[j] -= lr * dW2[j]

        b2 -= lr * db2

        y_hats = [forward(x, W1, b1, W2, b2)[1] for x in xs]
        current_loss = loss(y_hats, ys)
        history.append((W1, b1, W2, b2, current_loss))
    return history
