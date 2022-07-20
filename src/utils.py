from src.pachage_list import *

LABEL_MAPPING = {"Ineffective": 0, "Adequate": 1, "Effective": 2}

def get_logger(filename="output/"+'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = get_logger()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def _prepare_training_data_helper(cfg, tokenizer, df, is_train):
    training_samples = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        idx = row["essay_id"]
        discourse_text = row["discourse_text"]
        discourse_type = row["discourse_type"]

        if is_train:
            filename = os.path.join(cfg.data.input_dir_path, "train", idx + ".txt")
        else:
            filename = os.path.join(cfg.data.input_dir_path, "test", idx + ".txt")

        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()

        encoded_text = tokenizer.encode_plus(
            discourse_type + " " + discourse_text,
            text,
            add_special_tokens=False,
        )
        input_ids = encoded_text["input_ids"]
        sample = {
            "discourse_id": row["discourse_id"],
            "input_ids": input_ids,
            # "discourse_text": discourse_text,
            # "essay_text": text,
            # "mask": encoded_text["attention_mask"],
        }

        if "token_type_ids" in encoded_text:
            sample["token_type_ids"] = encoded_text["token_type_ids"]

        label = row["discourse_effectiveness"]

        sample["label"] = LABEL_MAPPING[label]

        training_samples.append(sample)
    return training_samples


def prepare_training_data(df, tokenizer, cfg, num_jobs, is_train):
    training_samples = []

    df_splits = np.array_split(df, num_jobs)

    results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(_prepare_training_data_helper)(cfg, tokenizer, df, is_train) for df in df_splits
    )
    for result in results:
        training_samples.extend(result)

    return training_samples

def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end

# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text

def get_essay(essay_id, is_train=True):
    parent_path = '../input/' + 'train' if is_train else INPUT_DIR + 'test'
    essay_path = os.path.join(parent_path, f"{essay_id}.txt")
    essay_text = open(essay_path, 'r', encoding='utf-8').read()
    return essay_text

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def get_score(y_true, y_pred):
    y_pred = softmax(y_pred)
    score = log_loss(y_true, y_pred)
    return round(score, 5)

def get_result(oof_df):
    labels = oof_df['target'].values
    preds = oof_df[['pred_0','pred_1','pred_2']].values
    score = get_score(labels, preds)
    LOGGER.info(f'Score: {score:<.4f}')

def freeze(module):
    """
    Freezes module's parameters.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False

def get_freezed_parameters(module):
    """
    Returns names of freezed parameters of the given module.
    """
    freezed_parameters = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            freezed_parameters.append(name)
            
    return freezed_parameters

# 8-bits optimizer
def set_embedding_parameters_bits(embeddings_path, optim_bits=32):
    """
    https://github.com/huggingface/transformers/issues/14819#issuecomment-1003427930
    """
    embedding_types = ("word", "position", "token_type")
    for embedding_type in embedding_types:
        attr_name = f"{embedding_type}_embeddings"
        
        if hasattr(embeddings_path, attr_name): 
            bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                getattr(embeddings_path, attr_name), 'weight', {'optim_bits': optim_bits}
            )