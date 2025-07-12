from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.postprocessors import get_postprocessor
from openood.utils import setup_logger
import torch
import pdb
import os
import cv2
import random
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
import pandas as pd
# from tsne import bh_sne
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)  

def load_taskres_model(config, net):
    model = net
    model_checkpoint_save_path = os.path.join(config.output_dir, 'model_checkpoint.pth')
    model_checkpoint = torch.load(model_checkpoint_save_path)
    # model_checkpoint['taskres_prompt_learner_state_dict'].pop('text_feature_residuals')
    # model_checkpoint['taskres_prompt_learner_state_dict'].pop('text_feature_scaling')
    model.model.taskres_prompt_learner.load_state_dict(model_checkpoint['taskres_prompt_learner_state_dict'], strict=False)
    # model_checkpoint['taskres_prompt_learner_state_dict']['meta_net.linear1.weight']
    return model.cuda()     

class TrainKRNFTPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)
        setup_seed(self.config.seed)

        # get dataloader
        loader_dict = get_dataloader(self.config)
        ood_loader_dict = get_ood_dataloader(self.config)

        train_loader, val_loader = loader_dict['train'], loader_dict['val']
        test_loader = loader_dict['test']

        # init network
        net = get_network(self.config.network)
        self.model = net

        model_checkpoint_save_path = os.path.join(self.config.output_dir, 'model_checkpoint.pth')
        evaluator = get_evaluator(self.config)
        postprocessor = get_postprocessor(self.config)
        # setup for distance-based methods
        postprocessor.setup(net, loader_dict, ood_loader_dict)
        print('\n', flush=True)
        print(u'\u2500' * 70, flush=True)

        # init recorder
        recorder = get_recorder(self.config)

        if os.path.exists(model_checkpoint_save_path):
            net = load_taskres_model(self.config, net)

        else:
            # init trainer and evaluator
            trainer = get_trainer(net, train_loader, val_loader, self.config)
            # trainer setup
            trainer.setup()
            print('\n' + u'\u2500' * 70, flush=True)

            print('Start training...', flush=True)
            for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
                # train and eval the model
                net, train_metrics = trainer.train_epoch(epoch_idx)


        # # evaluate on test set
        print('Start testing...', flush=True)
        
        # test_metrics = evaluator.eval_acc(net, test_loader, postprocessor)
        # print('\nComplete Evaluation, accuracy {:.2f}'.format(
        #     100.0 * test_metrics['acc']),
        #         flush=True)
            
            # start evaluating ood detection methods   
        #score = self.get_result(net, ood_loader_dict['farood']['places'])
        evaluator.eval_ood(net, loader_dict, ood_loader_dict, postprocessor)
        print('Completed!', flush=True)


