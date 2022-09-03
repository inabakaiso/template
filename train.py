from src.pachage_list import *
from src.utils import *
from src.train_function import *
from src.preprocess import *
from src.awp import *
from src.calc_score import *
from src.get_optimizer import *
from src.create_folds import *
from dataset.dataset import *
from model.model import *


NUM_JOBS = 12

def train_loop(cfg, df, fold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        wandb.login(key="2d4f1765feba36a6d2dce9f86d9f8f4c5d28daec")
        anony = None
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')


    def class2dict(f):
        return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

    run = wandb.init(project='Final-FeedBack-Prize', 
                     name=cfg.model.model_name,
                     config=class2dict(cfg),
                     group=cfg.model.model_name,
                     job_type="train",
                     anonymous=anony)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    SEP = tokenizer.sep_token

    lengths = []
    tk0 = tqdm(df[cfg.dataset.input_col].fillna("").values, total=len(df))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    max_len = max(lengths) + 3 # cls & sep & sep
    LOGGER.info(f"max_len: {max_len}")

    # DataSet Preparation
    train_df = df[df["kfold"] != fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == fold].reset_index(drop=True)
    valid_labels = valid_df[cfg.dataset.target_cols].values

    train_dataset = FeedbackDataset(cfg, train_df, max_len, tokenizer=tokenizer)
    valid_dataset = FeedbackDataset(cfg, valid_df ,max_len ,tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.dataset.batch_size,
                              shuffle=True,
                              num_workers=cfg.dataset.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg.dataset.batch_size,
                              shuffle=False,
                              num_workers=cfg.dataset.num_workers, pin_memory=True, drop_last=False)

    # Model Preparation
    model = CustomModel(
        cfg=cfg,
        config_path=None,
        pretrained=True
    )
    model.to(device)

    optimizer_parameters = get_optimizer_grouped_parameters(model=model, mode=cfg.optimizer.opt_mode, learning_rate=cfg.optimizer.learning_rate)
    optimizer = AdamW(optimizer_parameters, lr=cfg.training.encoder_lr, eps=cfg.training.eps, betas=(0.9, 0.999))

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler.scheduler_name == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.scheduler.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler.scheduler_name == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.scheduler.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.scheduler.num_cycles
            )
        return scheduler
    
    num_train_steps = int(len(train_df) / cfg.training.batch_size * cfg.training.epochs)
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)
    criterion = nn.SmoothL1Loss(reduction='mean') # RMSELoss(reduction="mean")

    best_score = 100.
    for epoch in range(cfg.training.epochs):
        start_time = time.time()
        avg_loss = train_fn(cfg, fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)
        # eval
        avg_val_loss, predictions = valid_fn(cfg, valid_loader, model, criterion, device)
        
        # scoring
        score, scores = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')
        wandb.log({f"[fold{fold}] epoch": epoch+1, 
                    f"[fold{fold}] avg_train_loss": avg_loss, 
                    f"[fold{fold}] avg_val_loss": avg_val_loss,
                    f"[fold{fold}] score": score})
        
        if best_score > score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        cfg.data.output_dir_path + "/" +f"{cfg.model.model_name.replace('/', '-')}_fold{fold}_best.pth")

    predictions = torch.load(cfg.data.output_dir_path + "/" + f"{cfg.model.model_name.replace('/', '-')}_fold{fold}_best.pth", 
                             map_location=torch.device('cpu'))['predictions']
    
    valid_df[[f"pred_{c}" for c in cfg.dataset.target_cols]] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_df

if __name__ == "__main__":
    with open('config/train_config.yaml', 'r') as yml:
        cfg = DictConfig(yaml.safe_load(yml))
    seed_everything(cfg.cv_strategy.seed)

    os.makedirs(cfg.data.output_dir_path, exist_ok=True)

    df = pd.read_csv(cfg.data.train_data_path)

    ## create folds
    df = create_folds(df, cfg.cv_strategy.num_folds,cfg.cv_strategy.seed, target_cols=cfg.dataset.target_cols, split_type=cfg.cv_strategy.split_type)

    oof_df = pd.DataFrame()
    print("\n")
    LOGGER.info(f"========== model name: {cfg.model.model_name} ==========")
    for fold in range(cfg.cv_strategy.num_split):
        _oof_df = train_loop(cfg, df, fold)
        oof_df = pd.concat([oof_df, _oof_df])
        LOGGER.info(f"========== fold: {fold} result ==========")
        score, scores = get_result(_oof_df)
        LOGGER.info(f'Score: {score:<.4f}  Scores: {scores}')

    oof_df= oof_df.reset_index(drop=False)
    score, scores = get_result(oof_df)
    LOGGER.info(f'Final Score: {score:<.4f}  Scores: {scores}')
    oof_df.to_pickle(cfg.data.output_dir_path + "/oof_df.pkl")
    wandb.finish()