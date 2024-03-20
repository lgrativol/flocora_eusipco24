import flwr as fl
from utils.utils import FlInfo, Info
import torch.multiprocessing as mp
mp.set_start_method('spawn',force=True)
import copy
from utils.mp_utils import mp_fit

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, info: Info, fl_info: FlInfo):
        self.fl_info = fl_info
        self.info = info
        # try : 
        #     if(info.lr_sched is not None):
        #         self.lr_sched = info.lr_sched
        #     else:
        #         self.lr_sched = None
        # except:
        #     self.lr_sched = None

    def get_properties(self, config):
        """Returns a client's set of properties."""
        return {"cid": self.fl_info.cid}

    def get_parameters(self, config):
        return None

    def fit(self, parameters, config):

        # if(self.lr_sched is not None):
        #     config["cl_lr"] = self.lr_sched()

        if self.fl_info.no_thread:
            return_dict = {}
            mp_fit(
                self.info,
                self.fl_info,
                config,
                parameters,
                return_dict,
            )

            params = return_dict["params"]
            size = return_dict["size"]
            del return_dict
        else:
            params, size = self._new_child(mp_fit, config, parameters)

        # if(self.lr_sched is not None):
        #     self.lr_sched.step()
        return params, size, {"cid": self.fl_info.cid}

    def _new_child(self, mp_func, config, parameters):
        manager = mp.Manager()
        return_dict = manager.dict()
        client_process = mp.Process(
            target=mp_func,
            args=(
                self.info,
                self.fl_info,
                config,
                parameters,
                return_dict,
            ),
        )
        client_process.start()
        client_process.join()
        client_process.close()

        params = return_dict["params"]
        size = return_dict["size"]
        del (manager, return_dict, client_process)
        return params, size

    def start_client(self):
        fl.client.start_numpy_client(server_address=self.fl_info.saddr, client=self)

# def client_dry_run():

#     # model = utils.load_efficientnet(classes=10)
#     # trainset, testset = utils.load_partition(0)
#     # trainset = torch.utils.data.Subset(trainset, range(10))
#     # testset = torch.utils.data.Subset(testset, range(10))
#     # client = CifarClient(trainset, testset, device)
#     # client.fit(
#     #     utils.get_model_params(model),
#     #     {"batch_size": 16, "local_epochs": 1},
#     # )

#     # client.evaluate(utils.get_model_params(model), {"val_steps": 32})

#     print("Dry Run Successful")

# if __name__ == "__main__":
#     import argparse
#     import torch
#     # Parse command line argument `partition`
#     parser = argparse.ArgumentParser(description="Flower Client")
#     parser.add_argument(
#         "--dry",
#         action="store_true",
#         help="Do a dry-run to check the client",
#     )
#     parser.add_argument(
#         "--partition",
#         type=int,
#         default=0,
#         help="Specifies the artificial data partition of CIFAR10 to be used. \
#         Picks partition 0 by default",
#     )

#     parser.add_argument(
#         "--cuda",
#         action="store_true",
#         help="Set to true to use GPU. Default: False",
#     )

#     args = parser.parse_args()

#     if args.cuda:
#         device = torch.device(
#             "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
#         )
#     else:
#         torch.device("cpu")

#     if args.dry:
#         client_dry_run(device)
#     else:
#         from pathlib import Path
#         from utils.models import model_selection

#         info = Info(dataset_name=args.dataset,
#                     feature_maps=args.feature_maps,
#                     input_shape = [3,32,32],
#                     num_classes = 10)

#         fl_info = FlInfo(saddr="0.0.0.0:8080",
#                             only_cpu=args.only_cpu,
#                             num_rounds=args.num_rounds,
#                             cid = str(0),
#                             fed_dir = Path('data/cifar-10-batches-py/federated/'),
#                             no_thread = True
#                             server_model = "resnet8")

#         name  = "resnet8"
#         model = model_selection(name)

#         # Start Flower client
#         client = FlowerClient(info,fl_info)
#         client.start_client()
