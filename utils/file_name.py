from args import args

def gen_filename():
    from utils.utils import pile_str

    file_name = ""
    if args.id_exp != "":
        file_name = "exp_" + args.id_exp + "_"

    file_name += args.model
    file_name = pile_str(file_name, args.strategy)
    file_name = pile_str(file_name, args.id_exp)
    file_name = pile_str(file_name, args.dataset)
    file_name = pile_str(file_name, "cle_" + str(args.cl_epochs))
    if args.prune:
        file_name = pile_str(file_name, "prune")
        file_name = pile_str(file_name, str(args.prate))
    if args.fedbn:
        file_name = pile_str(file_name, "fedbn")

    return file_name