import torch

def per_channel_scale_zero(ftensor,bits=8):
    
    qmax = 2**(bits-1) - 1
    qmin = -(2**(bits-1))

    if ftensor.dim() == 4: #conv
        dim_to_apply = (1,2,3) # apply along c_out
    elif ftensor.dim() == 2: # linear
        dim_to_apply = (1) 
    elif ftensor.dim() == 1: # bias and norm NOT SUPPORTED FOR NOW
        dim_to_apply = (0)
    else:
        print(f"Layer of dim: {ftensor.dim()} not supported, currently support for : 4 (conv), 2 (linear) and 1 (bias)")

    fmax = torch.amax(ftensor,dim=dim_to_apply,keepdim=True)
    fmin = torch.amin(ftensor,dim=dim_to_apply,keepdim=True)

    scale = (fmax - fmin)/(qmax - qmin)
    zero  = qmin - (fmin/scale)

    if torch.isnan(zero).any():
        zero = torch.zeros(zero.shape)

    return scale,zero

def quant_per_channel(ftensor,scale,zero,bits=8):
    qmax = 2**(bits-1) - 1
    qmin = -(2**(bits-1))

    if scale.sum() == 0. and zero.sum() ==0.:
        q_tensor = torch.zeros_like(ftensor)
    else:
        q_tensor = torch.clamp(torch.round(ftensor/scale) + zero,min=qmin,max=qmax)

    return q_tensor

def dequant_per_channel(q_tensor,scale,zero):
    dq_tensor = scale*(q_tensor - zero)

    return dq_tensor

def original_msg_size(model):
    model_size = 0 # in bits
    for m in model.parameters():
        if m.requires_grad : #if training 
            model_size+= m.numel()*32
    return model_size/8

def quant_msg_size(model,bits):
    model_size = 0 # in bits
    for _,m in model.named_parameters():
        if m.requires_grad and m.dim()>1: #if training 
            scale,zero = per_channel_scale_zero(m.detach(),bits)
            model_size+= m.numel()*bits + scale.numel()*32 + zero.numel()*32
        else:
            # If we cannot train, we're not sending, so we don't count
            # however, we're not quant. norm/bias tensors, so we
            # need to account for them
            if m.requires_grad:
                model_size+= m.numel()*32
    return model_size/8

def fakequant_trainable_channel(model,bits=8):
    # model_size = 0 # in bits
    for name,m in model.named_parameters():
        if m.requires_grad and m.dim()>1: #if training 
            # print("Quantizing layer : ",name)
            to_quant = m.detach()
            # print("Weights tensor shape : ", to_quant.shape)
            scale,zero = per_channel_scale_zero(to_quant,bits)
            q_tensor = quant_per_channel(to_quant,scale,zero,bits)
            # a bit cheating, we fakequant, but the final model size 
            # takes into account the scales and zero points
            dq_tensor = dequant_per_channel(q_tensor,scale,zero) 
            # quant_err = torch.max(to_quant - dq_tensor)
            # if(torch.isnan(quant_err)):
            #     print("We have a problem")
            # print(f"Max quantization error : {quant_err:.2E} ")
            m.data = dq_tensor
            # model_size+= m.numel()*bits + scale.numel()*32 + zero.numel()*32
        # else:
        #     # If we cannot train, we're not sending, so we don't count
        #     # however, we're not quant. norm/bias tensors, so we
        #     # need to account for them
        #     if m.requires_grad:
        #         model_size+= m.numel()*32

if __name__ == "__main__":
    from models.resnets import *
    from models.vgg import *
    from models.shufflenetv2 import *
    from models.toy_net import Net
    from utils.dcs import LoraInfo
    from utils.lora import *
    
    net_func = vgg16
    model = net_func(64,(3,32,32),10,batchn=False)
    target_modules,modules_to_save,rank_pattern = gen_rank_pattern(model,r=16)

    lora_config = LoraInfo(alpha=256,
                            r=16,
                            target_modules=target_modules,
                            modules_to_save=modules_to_save,
                            lora_type="lora",
                            rank_pattern=rank_pattern)

    # print(model)
    model = inject_low_rank(model,lora_config=lora_config)
    model.print_trainable_parameters()

    ori_msg = original_msg_size(model)
    quant_msg= fakequant_trainable_channel(model,bits=4)
    print(f"Original message size (fp32) : {ori_msg/1024} in KB")
    print(f"After quant message size     : {quant_msg/1024} in KB")
    print(f"Compression of : {ori_msg/quant_msg:.2f} times")