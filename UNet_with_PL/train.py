import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from dataloader.crack_dataloader import CrackDataset
from model.Unet_2d import UNet
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def main():
   
    pl.seed_everything(1234)
    
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--image_root', type=str, default="./data/Image/*") #
    parser.add_argument('--mask_root', type=str, default="./data/mask/*") #
    parser.add_argument('--log_dir', type=str, default="./checkpoint")
    parser.add_argument('--gpu_id', type=int, nargs='+', help='GPU ID Lists')
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--name', type=str, default='Trial')
    # add all the available options to the trainer
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    # ------------
    # datasets
    # ------------

    training_dataset = CrackDataset(args.image_root, args.mask_root, "train")
    training_loader = DataLoader(training_dataset, batch_size=args.batch_size)

    val_dataset = CrackDataset(args.image_root, args.mask_root, "val")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    testing_dataset = CrackDataset(args.image_root, args.mask_root, "test")
    testing_loader = DataLoader(testing_dataset, batch_size=args.batch_size)
    # ------------
    # model
    # ------------
    model = UNet()

    # ------------
    # log
    # ------------
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(str(args.log_dir), "logs"),name=args.name)
    
    # print(os.path.join(, "{epoch}-{val_dice:.2f}"))
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=str(args.log_dir),
        filename='Unet_{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        every_n_train_steps = 20,
        mode="max")

    # ------------
    # log
    # ------------

    # ------------
    # training
    # ------------

    trainer = pl.Trainer(
        gpus=args.gpu_id,
        max_epochs=args.max_epoch,
        logger=tb_logger,
        num_sanity_val_steps=1,
        callbacks=[checkpoint_callback],
        default_root_dir = args.log_dir
    )

    trainer.fit(model, training_loader, val_loader)
    
    checkpoint_callback.best_model_path
    # ------------
    # testing
    # ------------

    ## load the best model? did it? 
    result = trainer.test(model, test_dataloaders=testing_loader)
    print(result)


if __name__ == '__main__':
    main()