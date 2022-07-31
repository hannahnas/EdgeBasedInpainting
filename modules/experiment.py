import os
import pickle
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from modules.dataset import InpaintDataset
# Models
from depth_models.skip_attention.pl_skipattentionmodel import SkipAttentionModel
from depth_models.baseline.pl_baseline import BaselineModel
# from depth_models.edge_attention.pl_edgeattentionmodel import EdgeAttentionModel

CHECKPOINT_PATH = './experiment_checkpoints'
device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

modelname_to_class = {
    'BaselineModel': BaselineModel,
    'SkipAttentionModel': SkipAttentionModel,
    # 'EdgeAttentionModel': EdgeAttentionModel,
    # 'SkipResNetModel': SkipResNetModel
}

def run_experiment(hyper_params):
    # Reproducability
    pl.seed_everything(42)

    train_set = InpaintDataset(split = 'train')
    print('N datapoints train set:', len(train_set))

    val_set = InpaintDataset(split = 'val')
    print('N datapoints validation set:', len(val_set))

    test_set = InpaintDataset(split = 'test')
    print('N datapoints test set:', len(test_set))

    train_loader = DataLoader(train_set, batch_size=hyper_params['batch size'], shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=4)

    val_loader = DataLoader(val_set, batch_size=hyper_params['batch size'],
                            shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=8,
                             shuffle=False, drop_last=False, num_workers=4)
    
    model_checkpoint = ModelCheckpoint(save_weights_only=True, mode="min", monitor=hyper_params["monitor"])
    early_stopping = EarlyStopping(monitor=hyper_params["monitor"], min_delta=0.001, patience=3)

    model_path = f"{hyper_params['model name']}_batch{hyper_params['batch size']}_lr{hyper_params['lr']}_{hyper_params['experiment id']}"
    path = os.path.join(CHECKPOINT_PATH, model_path)
    trainer = pl.Trainer(default_root_dir=path,
                         val_check_interval=0.25,
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=hyper_params['epochs'],
                         log_every_n_steps=10,
                        #  limit_train_batches=10,
                         callbacks=[model_checkpoint,
                                    LearningRateMonitor("epoch"),\
                                    early_stopping
                                    ])

    model = modelname_to_class[hyper_params['model class']](hyper_params)

    trainer.fit(model, train_loader, val_loader)

    print(model_checkpoint.best_model_path)
    best_model = model_checkpoint.best_model_path

    model = None

    # Test best model on validation and test set
    val_result = trainer.test(
        model, ckpt_path=best_model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(
        model, ckpt_path=best_model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    # To do: save hyperparams in dict
    with open('./results/' + model_path + '.pickle', "wb") as f:
        pickle.dump(result, f)



