import torch
from torch import nn, optim
from tqdm import tqdm
from torch.nn import functional as F
from torch.autograd import Variable
from utils.utils import HyperParams, set_logger, RunningAverage
from models.transformers.optim import ScheduledOptimizer
from torch.nn.utils import clip_grad_norm
from utils.utils import make_tgt_mask
import time, math, os, shutil

class Trainer(object):
    """
    Class to handle the training of Encoder-Decoder Architectures 
    """
    def __init__(self, model, optimizer, criterion, num_epochs, train_iter, dev_iter, params):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_iter = train_iter
        self.dev_iter = dev_iter
        self.params = params
        self.epoch = 0
        self.max_num_epochs = num_epochs
        self.best_val_loss = float("inf")
    
    def train_epoch(self):
        """
        Train Encoder-Decoder model for one single epoch
        """
        self.model.train()
        total_loss = 0
        n_word_total = 0

        with tqdm(total=len(self.train_iter)) as t:
            for idx, batch in enumerate(self.train_iter):
                src, src_lengths = batch.src
                trg, trg_lengths = batch.trg
                src_mask = (src != self.params.pad_token).unsqueeze(-2) # [batch_size, 1, src_seq_len]
                trg_mask = make_tgt_mask(trg, self.params.pad_token) # [batch_size, trg_seq_len, trg_seq_len]

                if self.params.cuda:
                    src, trg = src.cuda(), trg.cuda()

                # run the data through the model
                self.optimizer.zero_grad()
                output = self.model(src, trg, src_mask, trg_mask, src_lengths, trg_lengths)

                output = output[:, :-1, :].contiguous().view(-1, self.params.tgt_vocab_size)
                trg = trg[:, 1:].contiguous().view(-1)

                assert output.size(0) == trg.size(0)

                loss = self.criterion(output, trg)
                loss.backward()

                # update the parameters
                if  isinstance(self.optimizer, ScheduledOptimizer):
                    self.optimizer.step_and_update_lr()
                else:
                    self.optimizer.step()

                # update the average loss
                total_loss += loss.item()
                non_pad_mask = trg.ne(self.params.pad_token)
                n_word = non_pad_mask.sum().item()
                n_word_total += n_word

                t.set_postfix(loss='{:05.3f}'.format(loss/n_word))
                t.update()
                torch.cuda.empty_cache()

        loss_per_word = total_loss/n_word_total
        return loss_per_word

    def validate(self):
        """
        Evaluate the loss of the Encoder-Decoder `model` on the dev set
        """
        self.model.eval()
        total_loss = 0
        n_word_total = 0
        with tqdm(total=len(self.dev_iter)) as t:
            with torch.no_grad():
                for idx, batch in enumerate(self.dev_iter):
                    src, src_lengths = batch.src
                    trg, trg_lengths = batch.trg
                    src_mask = (src != self.params.pad_token).unsqueeze(-2)
                    trg_mask = make_tgt_mask(trg, self.params.pad_token) # [batch_size, trg_seq_len, trg_seq_len]

                    if self.params.cuda:
                        src, trg = src.cuda(), trg.cuda()

                    # run the data through the model
                    output = self.model(src, trg, src_mask, trg_mask, src_lengths, trg_lengths)

                    output = output[:, :-1, :].contiguous().view(-1, self.params.tgt_vocab_size)
                    trg = trg[:, 1:].contiguous().view(-1)

                    assert output.size(0) == trg.size(0)

                    # compute the loss 
                    loss = self.criterion(output, trg)
                     
                    total_loss += loss.item()
                    non_pad_mask = trg.ne(self.params.pad_token)
                    n_word = non_pad_mask.sum().item()
                    n_word_total += n_word

                    t.set_postfix(loss='{:05.3f}'.format(loss/n_word))
                    t.update()
        loss_per_word = total_loss/n_word_total
        return loss_per_word

    def train(self):
        print("Starting training for {} epoch(s)".format(self.max_num_epochs - self.epoch))
        for epoch in range(self.max_num_epochs):
            self.epoch = epoch
            print("Epoch {}/{}".format(epoch+1, self.max_num_epochs))

            epoch_start_time = time.time()
            train_loss_avg = self.train_epoch()
            epoch_end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(epoch_start_time, epoch_end_time)
            print(f'Epoch: {epoch+1:02} | Avg Train Loss: {train_loss_avg} | Time: {epoch_mins}m {epoch_secs}s')

            val_loss_avg = self.validate() 
            print(f'Avg Val Loss: {val_loss_avg}')
            print('\n')

            is_best = val_loss_avg < self.best_val_loss

            optim_dict = self.optimizer._optimizer.state_dict() if isinstance(self.optimizer, ScheduledOptimizer) else self.optimizer.state_dict()

            # save checkpoint
            self.save_checkpoint({
                "epoch": epoch+1,
                "state_dict": self.model.state_dict(),
                "optim_dict": optim_dict},
                is_best=is_best,
                checkpoint=self.params.model_dir+"/checkpoints/")

            if is_best:
                print("- Found new lowest loss!")
                self.best_val_loss = val_loss_avg

    def epoch_time(self, start_time, end_time):
        """ Calculate the time to train a `model` on a single epoch """
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    def save_checkpoint(self, state, is_best, checkpoint):
        """
        Save a checkpoint of the model

        Arguments:
            state: dictionary containing information related to the state of the training process
            is_best: boolean value stating whether the current model got the best val loss
            checkpoint: folder where parameters are to be saved
        """
        filepath = os.path.join(checkpoint, "last.pth.tar")
        if not os.path.exists(checkpoint):
            os.mkdir(checkpoint)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint, "best.pth.tar"))

    @classmethod
    def load_checkpoint(cls, model, checkpoint, optimizer=None):
        """
        Loads model parameters (state_dict) from file_path. If optimizer is provided
        loads state_dict of optimizer assuming it is present in checkpoint

        Arguments:
            checkpoint: filename which needs to be loaded
            optimizer: resume optimizer from checkpoint
        """
        if not os.path.exists(checkpoint):
            raise ("File doesn't exist {}".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        if optimizer:
            if isinstance(optimizer, ScheduledOptimizer):
                optimizer._optimizer.load_state_dict(checkpoint["optim_dict"])
            else:
                optimizer.load_state_dict(checkpoint["optim_dict"])
        return model
