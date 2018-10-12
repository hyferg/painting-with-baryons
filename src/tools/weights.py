def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def freeze_model(model):
    FreezeModel(model)


def un_freeze_model(model):
    UnFreezeModel(model)


def FreezeModel(model):
    for param in model.parameters():
        param.requires_grad = False


def UnFreezeModel(model):
    for param in model.parameters():
        param.requires_grad = True
