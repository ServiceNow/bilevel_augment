import torch.optim as optim

def get_optimizer(optimparams, model):
    
    # classifier model optimizer
    if optimparams['optim'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optimparams['lr'], 
            #  betas=(optimparams['beta_1'], 0.999), 
             weight_decay=optimparams['weight_decay'])
    
    elif optimparams['optim'] == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=optimparams['lr'], betas=(optimparams['beta_1'], 0.999), weight_decay=optimparams['weight_decay'])
    
    elif optimparams['optim'] == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=optimparams['lr'])
    
    elif optimparams['optim'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=optimparams['lr'],
         momentum=optimparams['momentum'], weight_decay=optimparams['weight_decay'], \
                               nesterov=optimparams.get('nesterov', False))
    else:
        raise ValueError('opt %s does not exist' % optimparams['optim'])
    return optimizer
