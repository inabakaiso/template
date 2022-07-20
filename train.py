from src.pachage_list import *
from src.utils import *
from dataset.dataset import *
from src.train_function import *
from model.model import *
from typing import Optional
import yaml

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

    run = wandb.init(project='FeedBack-Prize', 
                     name=cfg.model.model_name,
                     config=class2dict(cfg),
                     group=cfg.model.model_name,
                     job_type="train",
                     anonymous=anony)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    SEP = tokenizer.sep_token

    df['text'] = df['discourse_type'] + ' ' + df['discourse_text']
    # DataSet Preparation
    train_df = df[df["kfold"] != fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == fold].reset_index(drop=True)
    valid_labels = valid_df['target'].values

    train_dataset = FeedbackDataset(cfg, train_df, tokenizer=tokenizer)
    valid_dataset = FeedbackDataset(cfg, valid_df ,tokenizer=tokenizer)

    collate_fn = Collate(tokenizer)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.dataset.batch_size,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=cfg.dataset.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg.dataset.batch_size,
                              shuffle=False,
                              collate_fn=collate_fn,
                              num_workers=cfg.dataset.num_workers, pin_memory=True, drop_last=False)

    # Model Preparation
    model = CustomModel(
        cfg=cfg,
        pretrained=True
    )
    model.to(device)

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
            param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_parameters = [
                {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': encoder_lr, 'weight_decay': weight_decay},
                {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': encoder_lr, 'weight_decay': 0.0},
                {'params': [p for n, p in model.named_parameters() if "model" not in n],
                'lr': decoder_lr, 'weight_decay': 0.0}
            ]
            return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=cfg.training.encoder_lr, 
                                                decoder_lr=cfg.training.decoder_lr,
                                                weight_decay=cfg.training.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=cfg.training.encoder_lr, eps=cfg.training.eps, betas=(0.9, 0.999))

    # optimizer = AdamW(model.parameters(), 
    #                   lr=cfg.training.encoder_lr,
    #                   eps=cfg.training.eps, 
    #                   betas=(0.9, 0.999))  

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
        else:
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=cfg.training.t_0, 
                                                             eta_min=cfg.training.min_lr)
        return scheduler
    
    num_train_steps = int(cfg.cv_strategy.num_split / cfg.training.batch_size * cfg.training.epochs)
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)
    criterion = nn.CrossEntropyLoss()

    best_score = 100.
    for epoch in range(cfg.training.epochs):
        start_time = time.time()
        avg_loss = train_fn(cfg, fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)
        # eval
        avg_val_loss, predictions = valid_fn(cfg, valid_loader, model, criterion, device)
        
        # scoring
        score = get_score(valid_labels, predictions)

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
    
    valid_df[['pred_0','pred_1','pred_2']] = predictions

    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_df

if __name__ == "__main__":
    with open('train_config.yaml', 'r') as yml:
        cfg = DictConfig(yaml.safe_load(yml))
    seed_everything(cfg.cv_strategy.seed)

    os.makedirs(cfg.data.output_dir_path, exist_ok=True)

    df = pd.read_csv(cfg.data.train_data_path)
    df['essay_text'] = df['essay_id'].apply(lambda x: get_essay(x, is_train=True))
    df['discourse_text'] = df['discourse_text'].apply(lambda x: resolve_encodings_and_normalize(x))
    df['essay_text'] = df['essay_text'].apply(lambda x: resolve_encodings_and_normalize(x))
    df['essay_text'] = df['essay_text'].str.lower()
    df['discourse_text'] = df['discourse_text'].str.lower()
    df['discourse_type'] = df['discourse_type'].str.lower()
    df['target'] = df['discourse_effectiveness'].map(LABEL_MAPPING)
    df['grp_var'] = df['discourse_type'] + df['discourse_effectiveness']
    ## debug flag
    if cfg.training.debug:
        df = df[:1000]
    
    ## make fold
    # gkf = StratifiedGroupKFold(n_splits=cfg.cv_strategy.num_split)
    # for fold, ( _, val_) in enumerate(gkf.split(df.grp_var,df.essay_id)):
    #     df.loc[val_ , "kfold"] = int(fold)
    # gkf = KFold(n_splits=CFG.n_fold)

    # gkf = GroupKFold(n_splits=CFG.n_fold)
    gkf = StratifiedKFold(n_splits=cfg.cv_strategy.num_split, shuffle=True, random_state=cfg.cv_strategy.seed)
    # gkf = StratifiedGroupKFold(n_splits=CFG.n_fold)
    for fold, (train_id, val_id) in enumerate(gkf.split(X=df, y=df.discourse_effectiveness, groups=df.essay_id)):
        # For all row in val_id list => create kfold column value
        df.loc[val_id , "kfold"] = fold

    df["kfold"] = df["kfold"].astype(int)
    LOGGER.info(df.groupby('kfold')['discourse_effectiveness'].value_counts())

    oof_df = pd.DataFrame()
    for fold in range(cfg.cv_strategy.num_split):
        _oof_df = train_loop(cfg, df, fold)
        oof_df = pd.concat([oof_df, _oof_df])
        LOGGER.info(f"========== fold: {fold} result ==========")
        get_result(_oof_df)
    oof_df= oof_df.reset_index(drop=False)
    get_result(oof_df)
    oof_df.to_pickle(cfg.data.output_dir_path + "/oof_df.pkl")
    wandb.finish()