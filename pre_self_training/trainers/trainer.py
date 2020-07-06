from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import models
from utils.models import add_net_to_params
from utils.saver import Saver

def norm_01(x):
    x = x.detach().cpu()
    for b in range(x.shape[0]):
        x[b] = (x[b] - x[b].min())/(x[b].max() - x[b].min())
    return x

class Trainer:

    def __init__(self, args):
        # Store args
        self.args = args

    def train(self, datasets):
        # Get args
        args = self.args
        saver = args.saver
        log_every = args.log_metrics_every
        plot_every = args.log_plots_every
        save_every = args.save_every
        # Compute splits names
        splits = list(datasets.keys())

        # Setup model
        module = getattr(models, args.model)
        net = getattr(module, "Model")(vars(args))
        # Check resume
        if args.resume is not None:
            try:
                net.load_state_dict(Saver.load_state_dict(args.resume))
            except:
                # We may have errors because the final layers don't match
                pass
        # Move to device
        net.to(args.device)
        # Add network to params
        add_net_to_params(net, args, 'net')

        # Setup data loader
        loaders = {
            s: DataLoader(datasets[s], batch_size=args.batch_size,
                shuffle=(s == 'train'),
                num_workers=(args.workers if not args.overfit_batch else 0),
                drop_last=True)
            for s in splits
        }

        # Optimizer params
        optim_params = {'lr': args.lr, 'weight_decay': args.weight_decay}
        if args.optim == 'Adam':
            optim_params = {**optim_params, 'betas': (0.9, 0.999)}
        elif args.optim == 'SGD':
            optim_params = {**optim_params, 'momentum': 0.9}
        # Create optimizer
        optim_class = getattr(torch.optim, args.optim)
        optim = optim_class(params=net.parameters(), **optim_params)
        # Configure LR scheduler
        scheduler = None
        if args.reduce_lr_every is not None:
            print("Setting up LR scheduler")
            scheduler = torch.optim.lr_scheduler.StepLR(
                optim, args.reduce_lr_every, args.reduce_lr_factor
            )

        # Initialize output metrics
        result_metrics = {s: {} for s in splits}
        # Process each epoch
        try:
            for epoch in range(args.epochs):
                # Process each split
                for split in splits:
                    # Epoch metrics
                    epoch_metrics = {}
                    # Set network mode
                    if split == 'train':
                        net.train()
                        torch.set_grad_enabled(True)
                    else:
                        net.eval()
                        torch.set_grad_enabled(False)
                    # Process each batch
                    dl = loaders[split]
                    pbar = tqdm(dl, leave=False)
                    for batch_idx, (inputs,labels) in enumerate(pbar):
                        # Compute step
                        step = (epoch * len(dl)) + batch_idx
                        # Set progress bar description
                        pbar_desc = f'{split}, epoch {epoch+1}'
                        if split == 'train':
                            pbar_desc += f', step {step}'
                        pbar.set_description(pbar_desc)
                        # Move to device
                        inputs = inputs.to(args.device)
                        labels = labels.to(args.device)
                        # Forward step
                        outputs = net(inputs)
                        # Check NaN
                        if torch.isnan(outputs).any():
                            raise FloatingPointError('Found NaN values')
                        # Compute loss
                        loss = F.cross_entropy(outputs, labels)
                        # Optimize
                        if split == 'train':
                            optim.zero_grad()
                            loss.backward()
                            optim.step()
                        # Compute accuracy
                        preds = torch.argmax(outputs, dim=1)
                        accuracy = (preds == labels).sum().item()/inputs.shape[0]
                        # Initialize metrics
                        metrics = {'loss': loss.item(),
                                   'accuracy': accuracy,
                        }
                        # Add metrics to epoch results
                        for k, v in metrics.items():
                            epoch_metrics[k] = epoch_metrics[k] + [v] if k in epoch_metrics else [v]
                        # Log metrics
                        if step % log_every == 0:
                            for k, v in metrics.items():
                                saver.dump_metric(v, step, split, k, 'batch')
                        # Plot stuff
                        if step % plot_every == 0:
                            # Log inputs
                            saver.dump_batch_image(norm_01(inputs), step, split, 'inputs')
                    # Epoch end: compute epoch metrics 
                    epoch_loss = sum(epoch_metrics['loss'])/len(epoch_metrics['loss'])
                    epoch_accuracy = sum(epoch_metrics['accuracy'])/len(epoch_metrics['accuracy'])
                    # Print to screen
                    pbar.close()
                    print(f'{split}, {epoch+1}: loss={epoch_loss:.4f}, accuracy={epoch_accuracy:.4f}')
                    # Dump to saver
                    saver.dump_metric(epoch_loss, epoch, split, 'loss', 'epoch')
                    saver.dump_metric(epoch_accuracy, epoch, split, 'accuracy', 'epoch')
                    # Add to output results
                    result_metrics[split]['loss'] = result_metrics[split]['loss'] + [epoch_loss] if 'loss' in result_metrics[split] else [epoch_loss]
                    result_metrics[split]['accuracy'] = result_metrics[split]['accuracy'] + [epoch_accuracy] if 'accuracy' in result_metrics[split] else [epoch_accuracy]
                # Check LR scheduler
                if scheduler is not None:
                    scheduler.step()
                # Save checkpoint
                if (epoch+1) % save_every == 0:
                    saver.save_model(net, args.model, epoch+1)
        except KeyboardInterrupt:
            pass
        except FloatingPointError as err:
            print(f'Error: {err}')
        # Return
        return net, result_metrics
