import warnings
import flwr as fl
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score

import utils

if __name__ == "__main__":
    file_paths = ['../nsl-kdd/KDDTrain+.txt', '../nsl-kdd/KDDTest+.txt']
    
    (X_train, y_train), (X_test, y_test) = utils.preprocessing(
        file_path_train=file_paths[0], file_path_test=file_paths[1])
    
    partition_id = np.random.choice(10)
    (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

    n_features = X_train.shape[1]
    model = utils.LR(utils.BasicLR(n_features))

    # Define Flower client
    class Client(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_params(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)

            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for x, y in zip(X_train, y_train):
                    out = model.forward(x)
                    out = torch.tensor(out)
                    model.backward(x, out, y)
                model.update_parameters()
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_params(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            total_loss = []
            for x, y in zip(X_test, y_test):
                out = model(x)
                out = torch.tensor(out).double()
                y = y.double()
                loss = out - y
                total_loss.append(loss.item())
            loss = np.mean(total_loss)
            accuracy = model.plain_accuracy(X_test, y_test)
            loss = float(loss)
            accuracy = float(accuracy)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client(server_address="localhost:8081", client=Client())
