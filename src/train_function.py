from src.pachage_list import *
from src.utils import *

def train_fn(cfg, fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, inputs in enumerate(train_loader):
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = inputs["target"].to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=cfg.training.apex):
            y_preds, metrics = model(ids=input_ids, mask=attention_mask, targets=labels)
            loss = criterion(y_preds, labels)
        if cfg.training.gradient_accumulation_steps > 1:
            loss = loss / cfg.training.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
        if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if cfg.training.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % cfg.training.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_last_lr()[0]))
        wandb.log({f"[fold{fold}] loss": losses.val,
                    f"[fold{fold}] lr": scheduler.get_last_lr()[0]})
    return losses.avg

@torch.no_grad()
def valid_fn(cfg, valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    labels = []
    start = end = time.time()
    for step, inputs in enumerate(valid_loader):
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        label = inputs["target"].to(device)
        batch_size = label.size(0)
        y_preds, metric = model(ids=input_ids, mask=attention_mask, targets=label)
        loss = criterion(y_preds, label)
        if cfg.training.gradient_accumulation_steps > 1:
            loss = loss / cfg.training.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to('cpu').numpy())
        labels.append(label.to('cpu').numpy())
        end = time.time()
        if step % cfg.training.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Metric: {metric:.4f}'
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader)), metric=metric))
    predictions = np.concatenate(preds)
    return losses.avg, predictions