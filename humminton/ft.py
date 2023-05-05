import re
import numpy as np


# your code here


with open('input.txt', 'r', encoding='utf-8',errors='ignore') as f:
    text = f.read() 

# %%
text = text.replace('\n', '').lower().replace('\t', '')

# %%
text  = re.sub('[^a-z]+', '', text.lower())

# %%
text = re.sub('\s+', ' ', text).strip()
# Define a function to encode the string into a list of 0's and 1's
def encode_binary_list(s):
    vowels = "aeiou"
    encoded = []
    for letter in s:
        if letter.lower() in vowels:
            encoded.append(0)
        else:
            encoded.append(1)
    return encoded

# Call the function on the input string and print the result
text = (encode_binary_list(text))
print(text)

vocab_set = list(set(text))
vocab_set

class MyMultinomialHMM:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.startprob_ = None
        self.transmat_ = None
        self.emissionprob_ = None

    def set_startprob(self, startprob):
        self.startprob_ = startprob

    def set_transmat(self, transmat):
        self.transmat_ = transmat

    def set_emissionprob(self, emissionprob):
        self.emissionprob_ = emissionprob

    def generate(self, length):
        if self.startprob_ is None or self.transmat_ is None or self.emissionprob_ is None:
            raise ValueError("Model parameters not set")

        hidden_states = []
        obs = []
        state = np.random.choice(self.n_components, p=self.startprob_)
        for i in range(length):
            hidden_states.append(state)
            obs.append(np.random.choice(2, p=self.emissionprob_[state]))
            state = np.random.choice(self.n_components, p=self.transmat_[state])
        return np.array(obs), np.array(hidden_states)

    def viterbi(self, X):
        if self.startprob_ is None or self.transmat_ is None or self.emissionprob_ is None:
            raise ValueError("Model parameters not set")

        T = len(X)
        delta = np.zeros((T, self.n_components))
        psi = np.zeros((T, self.n_components), dtype=int)

        # Initialization
        delta[0] = self.startprob_ * self.emissionprob_[:, X[0]]

        # Recursion
        for t in range(1, T):
            for j in range(self.n_components):
                temp = delta[t-1] * self.transmat_[:, j] * self.emissionprob_[j, X[t]]
                psi[t, j] = np.argmax(temp)
                delta[t, j] = temp[psi[t, j]]

        # Termination
        q = np.zeros(T, dtype=int)
        q[T-1] = np.argmax(delta[T-1])

        # Backtracking
        for t in range(T-2, -1, -1):
            q[t] = psi[t+1, q[t+1]]

        return q

    def fit(self, X):
        # Calculate initial state probabilities
        self.startprob_ = np.array([0.5, 0.5])
        print("start probabilities:", self.startprob_)

        # Calculate transition probabilities
        self.transmat_ = np.array([[0.7, 0.3], [0.3, 0.7]])
        print("transition matrix:")
        print(self.transmat_)

        # Calculate emission probabilities
        counts = np.zeros((self.n_components, 2))
        for i in range(len(X)):
            state = X[i]
            counts[state, i%2] += 1
        self.emissionprob_ = counts / counts.sum(axis=1)[:, np.newaxis]
        print("observation matrix:")
        print(self.emissionprob_)

         # Apply Baum-Reestimation formula
        for iter in range(10):
            alpha = self.forward(X)
            beta = self.backward(X)
            gamma = alpha * beta / np.sum(alpha * beta, axis=1, keepdims=True)
            xi = np.zeros((len(X)-1, self.n_components, self.n_components))
            for t in range(len(X)-1):
                for i in range(self.n_components):
                    for j in range(self.n_components):
                        xi[t, i, j] = alpha[t, i] * self.transmat_[i, j] * self.emissionprob_[j, X[t+1]] * beta[t+1, j]
                xi[t] /= xi[t].sum()

    # Re-estimate parameters
    def forward(self, X):
        alpha = np.zeros((len(X), self.n_components))
         # Initialization
        alpha[0] = self.startprob_ * self.emissionprob_[:, X[0]]

        # Induction
        for t in range(1, len(X)):
            alpha[t] = np.dot(alpha[t-1], self.transmat_) * self.emissionprob_[:, X[t]]

        return alpha

    def backward(self, X):
        beta = np.zeros((len(X), self.n_components))

        # Initialization
        beta[-1] = 1.0

        # Induction
        for t in range(len(X)-2, -1, -1):
            beta[t] = np.dot(self.transmat_, self.emissionprob_[:, X[t+1]] * beta[t+1])

        return beta

    def predict(self, X):
        q = self.viterbi(X)
        return q

    def score(self, X):
        alpha = self.forward(X)
        return np.log(np.sum(alpha[-1]))

    def decode(self, X):
        delta = self.forward(X)
        psi = np.zeros((len(X), self.n_components), dtype=int)

        # Initialization
        psi[0] = 0

        # Recursion
        for t in range(1, len(X)):
            psi[t] = np.argmax(np.dot(psi[t-1], self.transmat_))  # Using argmax instead of max

        # Termination
        q = np.zeros(len(X), dtype=int)
        q[-1] = np.argmax(delta[-1])

        # Backtracking
        for t in range(len(X)-2, -1, -1):
            q[t] = psi[t+1, q[t+1]]

        return q

model = MyMultinomialHMM(n_components=2)
model.fit(text)
