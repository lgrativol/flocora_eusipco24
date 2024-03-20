from torch.utils.data import DataLoader
from utils.utils import test, save_model,set_params
from log import logger, HFILE
from args import args
from utils.lora import extract_AB_matrix
import torch
from log import logger

if(args.wandb):
    import wandb

class Evaluate():
    def __init__(self,model,test_set,device):
        self.test_loader = DataLoader(test_set, batch_size=256, shuffle=False, pin_memory=True, num_workers=args.nworkers        )
        self.model = model
        self.device = device

    def __call__(self,server_round, parameters,config,to_log={}):
        set_params(self.model,parameters,args.fedbn)
        self.model.to(self.device)
        ans = test(self.model, self.test_loader, self.device)
        loss = ans["test_loss"]
        accuracy = ans["test_acc"]
        logger.info(
            f"Server round {server_round} loss : {loss:.4f} acc : {accuracy:.4f}",
            extra=HFILE,
        )

        if args.num_rounds == server_round or server_round % args.freq_checkpoint == 0:
            save_model(f"checkpoint/{args.file_name}.npy", parameters)
        if(server_round != -1 and args.wandb):
            to_log["acc"] = accuracy
            to_log["loss"] = loss
            wandb.log(to_log)


        return loss, {"accuracy": accuracy}
    

class EvaluateLora():
    def __init__(self,model,lora_config,test_set,device):
        self.test_loader = DataLoader(test_set, batch_size=256, shuffle=False, pin_memory=True, num_workers=args.nworkers)
        # self.model = inject_lora(model,lora_config.alpha,lora_config.r,lora_config.target_modules,lora_config.modules_to_save)
        self.model = model
        self.device = device
        self.past_a_matrix,self.past_b_matrix = extract_AB_matrix(model.state_dict())

    def __call__(self,server_round, parameters,config):
        set_params(self.model,parameters,args.fedbn)
        to_log = {}

        self.model.to(self.device)
        if args.log_a_sim and server_round >=1:
            current_a_list,current_b_list = extract_AB_matrix(self.model.state_dict())
            cos = torch.nn.CosineSimilarity(dim=0)
            simA = [cos(curr,old).mean() for curr,old in zip(current_a_list,self.past_a_matrix)]
            simB = [cos(curr,old).mean() for curr,old in zip(current_b_list,self.past_b_matrix)]
            for i,(sA,sB) in enumerate(zip(simA,simB)):
                logger.info(f"Similarity for layer {i} --> simA: {sA}, simB: {sB}")
                if(args.wandb):
                    # wandb.log({f"layer_{i}" : s})
                    to_log[f"simA_l{i}"]= sA
                    to_log[f"simB_l{i}"]= sB
            # for ii,p in enumerate(current_a_list):
            #     p_rank = torch.linalg.matrix_rank(p)
            #     logger.info(f"--> Layer {ii} : Rank = {p_rank}" )
            self.past_a_matrix = current_a_list
            self.past_b_matrix = current_b_list


        ans = test(self.model, self.test_loader, self.device)
        loss = ans["test_loss"]
        accuracy = ans["test_acc"]
        logger.info(
            f"Server round {server_round} loss : {loss:.4f} acc : {accuracy:.4f}",
            extra=HFILE,
        )

        if args.num_rounds == server_round or server_round % args.freq_checkpoint == 0:
            save_model(f"checkpoint/{args.file_name}.npy", parameters)
        if(server_round != -1 and args.wandb):
            to_log["acc"] = accuracy
            to_log["loss"] = loss
            wandb.log(to_log)

        return loss, {"accuracy": accuracy}
def get_evaluate_fn(model, test_set, device):
    """Return an evaluation function for server-side evaluation."""

    global evaluate
    
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False, pin_memory=True, num_workers=args.nworkers
    )

    def evaluate(server_round, parameters, config):
        set_params(model,parameters,args.fedbn)
        model.to(device)
        ans = test(model, test_loader, device)
        loss = ans["test_loss"]
        accuracy = ans["test_acc"]
        logger.info(
            f"Server round {server_round} loss : {loss:.4f} acc : {accuracy:.4f}",
            extra=HFILE,
        )

        if args.num_rounds == server_round or server_round % args.freq_checkpoint == 0:
            save_model(f"checkpoint/{args.file_name}.npy", parameters)
        if(server_round != -1 and args.wandb):
            wandb.log({"acc": accuracy, "loss": loss})

        return loss, {"accuracy": accuracy}

    return evaluate

def get_model_size(model):
    return [p.shape for p in model.parameters()]
