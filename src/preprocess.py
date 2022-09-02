import re
import os

def add_discourse_start_end(discourse_df, cfg, datatype="train"):
    idx = discourse_df.essay_id.values[0]
    filename = os.path.join(cfg.data.train_txt_path, idx + ".txt")
    with open(filename, "r", encoding='utf-8') as f:
        text = f.read()
    min_idx = 0
    starts = []
    ends = []
    for _, row in discourse_df.iterrows():
        discourse_text = row["discourse_text"]
        matches = list(re.finditer(re.escape(discourse_text.strip()), text))
        if len(matches) == 1:
            discourse_start = matches[0].span()[0] ## matchesで検索することで元の部分の位置情報を得ることが可能 . span に格納されている
            discourse_end = matches[0].span()[1]   ## spanは単語単位ではなく、文字単位のindexで取り出される
            min_idx = discourse_end
        elif len(matches) > 1:
            for match in matches:
                discourse_start = match.span()[0]
                discourse_end = match.span()[1]
                if discourse_start >= min_idx:
                    min_idx = discourse_end
                    break
        else:
            discourse_start = -1
            discourse_end = -1
        starts.append(discourse_start)
        ends.append(discourse_end)
    discourse_df.loc[:, "discourse_start"] = starts
    discourse_df.loc[:, "discourse_end"] = ends
    return discourse_df