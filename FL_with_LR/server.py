import flwr as fl
import numpy as np
import utils
import torch
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    file_paths = ['../nsl-kdd/KDDTrain+.txt', '../nsl-kdd/KDDTest+.txt']

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils.preprocessing(file_path_train=file_paths[0], file_path_test=file_paths[1])

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        # print('parameters to server : ', parameters)
        utils.set_model_params(model, parameters)
        # loss = log_loss(y_test, model.predict_proba(X_test))
        total_loss = []
        for x, y in zip(X_test, y_test):
            out = model(x)
            out = torch.tensor(out).double()
            y = y.double()
            loss = out - y
            total_loss.append(loss.item())
        loss = np.mean(total_loss)
        # accuracy = model.score(X_test, y_test)
        accuracy = model.plain_accuracy(X_test, y_test)
        loss = float(loss)
        accuracy = float(accuracy)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = utils.LR(utils.BasicLR(48))     # 不確定要怎麼取得features的數量
    # utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="127.0.0.1:8081",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5),
    )
