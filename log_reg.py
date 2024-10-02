import tensorflow as tf
import numpy as np


class Logistic_Regression(tf.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.dtype = None
        self.weights = None
        self.best_weights = None
        self.bias = None
        self.best_bias = None

        self.losses = []

        self.iterations = 1000
        self.best_loss = None

    def forward_pass(self, X):
        return tf.matmul(X, self.weights) + self.bias

    def predict_proba(self, X):
        return tf.sigmoid(self.forward_pass(X))

    def predict(self, X):
        return tf.cast(self.predict_proba(X) > 0.5, self.dtype)

    def train(
        self,
        X,
        y,
        learning_rate: float = 0.001,
        iterations: int = 1000,
        patience: int = 0,
        tolerance: float = 1e-5,
    ):
        early_stopping = False
        patience_counter = 0
        self.dtype = X.dtype
        if patience > 0:
            early_stopping = True

        self.weights = tf.Variable(
            tf.random.uniform(
                shape=[X.shape[-1], 1],
                seed=42,
                minval=-0.01,
                maxval=0.01,
                dtype=self.dtype,
            ),
            name="weights",
        )
        self.bias = tf.Variable(tf.zeros(shape=[1], dtype=self.dtype), name="bias")

        self.best_weights = tf.Variable(
            tf.random.uniform(
                shape=[X.shape[-1], 1],
                seed=42,
                minval=-0.01,
                maxval=0.01,
                dtype=self.dtype,
            ),
            name="best_weights",
        )

        self.best_bias = tf.Variable(
            tf.zeros(shape=[1], dtype=self.dtype), name="best_bias"
        )

        self.best_loss = tf.Variable(
            tf.float32.max if self.dtype == np.float32 else tf.float64.max,
            name="best_loss",
        )

        self.optimizer = tf.optimizers.SGD(learning_rate, learning_rate)

        for iteration in range(iterations):
            with tf.GradientTape() as tape:
                h = self.forward_pass(X)
                loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=h, labels=y)
                )

            gradients = tape.gradient(loss, [self.weights, self.bias])
            self.optimizer.apply_gradients(zip(gradients, [self.weights, self.bias]))
            self.losses.append(loss)

            if loss < self.best_loss:
                self.best_loss.assign(loss)
                patience_counter = 0
                self.best_weights.assign(self.weights)
                self.best_bias.assign(self.bias)
            else:
                patience_counter += 1
            if len(self.losses) > 1 and abs(self.losses[-2] - loss) < tolerance:
                print(f"Model converged on iteration {iteration}.")
                break

            if early_stopping and patience_counter >= patience:
                print(
                    f"Early Stopping on iteration {iteration} due to no change in loss for {patience} iterations"
                )
                self.weights.assign(self.best_weights)
                self.bias.assign(self.best_bias)
                break
            print(f"Iteration: {iteration}/{iterations}\tLoss: {loss}")
