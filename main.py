import argparse
import torch
import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import time
from engine import detector
from utils import *
import torch.utils.data as data
from load_data import data_loader

torch.set_num_threads(4) 

# torch.backends.cudnn.benchmark = True

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
'''Main Hyperparameter'''
parser.add_argument('--dataset', type=str, default='la', help='bay or la')
parser.add_argument('--lane', action='store_true', default=True, help='whether to use lane data')
parser.add_argument('--trend_time', type=int, default=7 * 24, help='the length of trend segment is 7 days')
parser.add_argument('--root_path', type=str, default='./', help='root path: dataset, checkpoint')

'''Model Hyperparameter'''
parser.add_argument('--detector', type=str, default='anodae', 
                    choices=['anodae','anodae1','anodae2', 'anodae3','anodae4','dominant'])

# Anodae
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in LSTM and DCRNN')
parser.add_argument('--hidden1', type=int, default=64)
parser.add_argument('--hidden2', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--alpha', type=float, default= 0.5)
parser.add_argument('--act', type=str, default= 'tanh', choices=['relu', 'tanh'])
parser.add_argument('--emb', type=str, default= 'cnn', choices=['rnn', 'cnn'])

'''Training Hyperparameter'''
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--is_train', type =bool, default=False, help='whether to train original predictor')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--epoch', type=int, default=96, help='Number of training epochs for predictor.')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--patience', type=int, default=7, help='early stop hyperparameter')
parser.add_argument('--print_every_iter', type=int, default=20, help='Number of training epochs for predictor.')

parser.add_argument('--cpu', action='store_true')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cuda_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=20)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
args.device = torch.device(f'cuda:{args.cuda_id}')
print(args.device)

# parameter
opt = vars(args)
# 2017-01-01 - 2017-05-06
if opt['dataset'] == 'bay':
    opt['timestamp'] = 12       # 5min: 12 or 30min: 2
    opt['train_time'] = 95     # days for training 
    opt['eval_time'] = 10     # days for validating
    opt['recent_time'] = 1      # bay: 1 hour, nyc: 2hour
    opt['future_time'] = 1                # bay 
    opt['num_adj'] = 365        # number of nodes in sub graph
    if opt['lane']:
        opt['num_feature'] = 2 * 6      # length of input feature
    else:
        opt['num_feature'] = 2
    opt['time_feature'] = 31        # length of time feature
# 2014-01-01 -- 2017-04-08
elif opt['dataset'] == 'la':
    opt['timestamp'] = 12       # 5min: 12 or 30min: 2
    opt['train_time'] = 70     # days for training 
    opt['eval_time'] = 8     # days for validating
    opt['recent_time'] = 1      # bay: 1 hour, nyc: 2hour
    opt['future_time'] = 1
    opt['num_adj'] = 204       # number of nodes in sub graph
    if opt['lane']:
        opt['num_feature'] = 2 * 6      # length of input feature
    else:
        opt['num_feature'] = 2


opt['base_path'] = opt['root_path'] + opt['dataset'] 
opt['data_path'] = opt['base_path'] + '/data/'

if opt['lane'] == True:
    opt['save_path'] = opt['base_path']  + '/checkpoint/lane/' + opt['detector']
    opt['result_path'] = opt['base_path'] + '/result/lane'
else:
    opt['save_path'] = opt['base_path']  + '/checkpoint/road/' + opt['detector']
    opt['result_path'] = opt['base_path'] + '/result/road' 

opt['train_time'] = opt['train_time'] * opt['timestamp'] * 24 # (days * 24 * 12)
opt['eval_time'] = opt['eval_time'] * opt['timestamp'] * 24 # (days * 24 * 12)

if not os.path.exists(opt['save_path']):
    os.makedirs(opt['save_path'])

if opt['detector'] == 'dominant':
    opt['model_path'] = opt['save_path'] + f'/D_{args.detector}_h_{args.hidden1}_alpha_{args.alpha}.pth' 
elif opt['detector'] == 'anodae':
    opt['model_path'] = opt['save_path'] + f'/D_{args.detector}_h1_{args.hidden1}_h2_{args.hidden2}_alpha_{args.alpha}.pth' 
elif opt['detector'] == 'anodae1':
    opt['model_path'] = opt['save_path'] + f'/D_{args.detector}_emb_{args.emb}_L_{args.num_layers}_h1_{args.hidden1}_h2_{args.hidden2}_alpha_{args.alpha}_act_{args.act}.pth' 
elif opt['detector'] == 'anodae2':
    opt['model_path'] = opt['save_path'] + f'/D_{args.detector}_emb_{args.emb}_L_{args.num_layers}_h1_{args.hidden1}_h2_{args.hidden2}_alpha_{args.alpha}_act_{args.act}.pth' 
elif opt['detector'] == 'anodae3':
    opt['model_path'] = opt['save_path'] + f'/D_{args.detector}_emb_{args.emb}_L_{args.num_layers}_h1_{args.hidden1}_h2_{args.hidden2}_alpha_{args.alpha}_act_{args.act}.pth' 
elif opt['detector'] == 'anodae4':
    opt['model_path'] = opt['save_path'] + f'/D_{args.detector}_emb_{args.emb}_L_{args.num_layers}_h1_{args.hidden1}_h2_{args.hidden2}_alpha_{args.alpha}_act_{args.act}.pth' 

print(opt['model_path'])

def load(opt):
    logging.info('****Loading Data****')

    dataloaders = []
    for set in ['train', 'eval', 'test']:
        loader = data_loader(opt, set)
        if set != 'test':
            dataloaders.append(data.DataLoader(loader, batch_size=opt['batch_size'], shuffle=True))
        else:
            dataloaders.append(data.DataLoader(loader, batch_size=opt['batch_size'], shuffle=False))
    # train_loader = data.DataLoader(data_loader(opt, sets='train'), batch_size=opt['batch_size'], shuffle=True)
    # eval_loader = data.DataLoader(data_loader(opt, sets='eval'), batch_size=opt['batch_size'], shuffle=True)
    # test_loader = data.DataLoader(data_loader(opt, sets='test'), batch_size=opt['batch_size'], shuffle=True)
    # scaler = [loader.min_source1, loader.max_source1, loader.min_source2, loader.max_source2]
    scaler = [loader.min_sources, loader.max_sources]
    dataloaders.append(scaler)
    logging.info('****Data Loaded****')
    
    return dataloaders


def detects(opt, loaders):
    detect = detector(opt)
    train_loader, eval_loader, test_loader, scalar = loaders

    if opt['is_train']:
        logging.info('**** Starting Training the detecter ****')
        loss_eval_min = float('inf')
        wait = 0 
        for e in range(1, opt['epoch']+1):
            t0 = time.time()
            loss_train_e = detect.train(train_loader)
            t1 = time.time()
            logging.info("epoch:%d training time:%f [D loss on trainset: %f]" % (e, t1-t0, loss_train_e))

            loss_eval_e = detect.eval(eval_loader)
            t2 = time.time()
            logging.info("epoch:%d inference time:%f [D loss on evalset: %f]" % (e, t2-t1, loss_eval_e))
            # early stop
            if loss_eval_e < loss_eval_min:
                wait = 0
                logging.info(f'val loss decrease from {loss_eval_min:.4f} to {loss_eval_e:.4f}')
                loss_eval_min = loss_eval_e
                best_e = e
                best_model_wts = detect.model.state_dict()
                torch.save(best_model_wts, opt['model_path'])
            else:
                wait += 1
                if wait == opt['patience']:
                    logging.info(f'Early stop at epoch: {e:03d}')
                    break
        
        logging.info(f'Model achives its peak at epoch {best_e:03d} with val loss {loss_eval_min}')
        logging.info('***** Finishing Training the detecter *****')
        detect.model.load_state_dict(best_model_wts)
    else:
        detect.model.load_state_dict(torch.load(opt['model_path'], map_location=args.device)) # 
        logging.info('**** Import trained model from' + opt['save_path'] + 'D_'+ opt['detector'] + '.pth ****' )


    logging.info('****Test the detecter****')
    t3 = time.time()
    loss_test = detect.eval(test_loader)
    t4 = time.time()
    logging.info(f'Inference Time:{t4-t3:4f}, D[Loss: {loss_test:4f}]')

    return detect.model



if __name__ == "__main__":
    init_logging('logs/logfile.log')
    # load datasets
    loaders = load(opt)
    
    D = detects(opt, loaders)

    # test(opt, loaders, P, R)

  
    
    
    
    






