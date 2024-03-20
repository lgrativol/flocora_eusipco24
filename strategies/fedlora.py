from typing import Callable, Dict, List, Optional, Tuple
from logging import WARNING
import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from flwr.common import (
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from strategies import FedAvg
from flwr.server.strategy.aggregate import aggregate
from flwr.common.logger import log

from utils.utils import get_random_guess_perf

class FedLora(FedAvg):
    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        device=None,
        drop_random : bool = False,
        fedbn : bool = False
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.device = device
        self.server_parameters = initial_parameters
        self.clients_params: Dict[NDArrays] = {}
        self.drop_random = drop_random
        self.method_string = "FedLora"

    def __repr__(self) -> str:
        rep = f"FedLora(accept_failures={self.accept_failures})"
        return rep


    def aggregate_fit(self,server_round,results,failures,):
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if(self.drop_random):
            weights_results = []
            random_guess = get_random_guess_perf("cifar10") #TODO : Redo
            # Convert results
            for _,fit_res in results:
                _,res = self.evaluate_fn(-1,parameters_to_ndarrays(fit_res.parameters),{})
                if res["accuracy"] > random_guess:
                    weights_results.append([parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples])
        else:
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) # lora weights
                for _, fit_res in results
            ]

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
