from models.binaryconnect import BC
BNN_CLASS = BC.__name__

class NaiveScheduler():
    def __init__(self,lr_init,lr_min,lr_decay):
        self.lr_init = lr_init
        self.lr = lr_init
        self.lr_min = lr_min
        self.lr_decay = lr_decay
    
    def __call__(self):
        return self.lr

    def step(self):
        if(self.lr > self.lr_min):
            self.lr*=self.lr_decay


class ModelHandler():
    def __init__(self,model,qualifier):
        self.fn_model = model
        self.qualifier = qualifier

    def __call__(self,feature_maps,input_shape,num_classes,batchn):

        if(self.qualifier!=""):
            if(self.qualifier[0] == "w"):
                qnt = self.qualifier[1]
                return self.fn_model(feature_maps,input_shape,num_classes,batchn,wbits=int(qnt))

        model = self.fn_model(feature_maps,input_shape,num_classes,batchn)

        if(self.qualifier == "bnn"):
            model = BC(model)

        return model

def model_selection(model):
    split = model.split("-")
    model_name = split[0]
    if(len(split)<2):
        qualifier = ""
    else:
        qualifier = split[1]

    if(model_name == "resnet8"):
        from models.resnets import resnet8       
        fn_model = resnet8
    elif(model_name == "resnet12"):
        from models.resnet12 import ResNet12
        fn_model = ResNet12
    elif(model_name == "resnet18"):
        from models.resnets import resnet18
        fn_model = resnet18
    elif(model_name == "resnet20"):
        from models.resnets import resnet20
        fn_model = resnet20
    elif(model_name == "resnet32"):
        from models.resnets import resnet32
        fn_model = resnet32
    elif(model_name == "toy"):
        from models.toy_net import Net
        fn_model = Net
    elif(model_name == "vgg9"):
        from models.vgg import vgg9
        fn_model = vgg9
    elif(model_name == "vgg11"):
        from models.vgg import vgg11
        fn_model = vgg11
    elif(model_name == "vgg13"):
        from models.vgg import vgg13
        fn_model = vgg13
    elif(model_name == "vgg16"):
        from models.vgg import vgg16
        fn_model = vgg16
    elif(model_name == "vgg19"):
        from models.vgg import vgg19
        fn_model = vgg19
    elif(model_name == "mobilenetv2"):
        from models.mobilenetv2 import mobilenetv2
        fn_model = mobilenetv2
    elif(model_name == "shufflenetv2"):
        from models.shufflenetv2 import shufflenetv2
        fn_model = shufflenetv2
    elif(model_name == "qresnet8"):
        from models.qresnets import QResNet8       
        fn_model = QResNet8
    elif(model_name == "qresnet12"):
        from models.qresnet12 import QResNet12
        fn_model = QResNet12
    elif(model_name == "qresnet18"):
        from models.qresnets import QResNet18       
        fn_model = QResNet18
    elif(model_name == "qresnet20"):
        from models.qresnets import QResNet20       
        fn_model = QResNet20
    else:
        print("No model, try again")
        exit(-1)

    return ModelHandler(fn_model,qualifier)

def do_model_pool(model,pool_size,kd=False,kd_pool=None):

    server_model = model_selection(model)

    if(kd):
        selection = kd_pool
    else:
        selection = [model]*pool_size
    
    return server_model,selection

if __name__ == "__main__":
    model = "qresnet8-w8"

    net = model_selection(model)

    print(net(32,[3,32,32],10,batchn=False))