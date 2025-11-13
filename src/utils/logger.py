import os
import json
import logging
from logging.handlers import WatchedFileHandler

def setup_live_logger(save_dir: str, filename: str = "bandit_live.log", reset = True):
    """
    单文件动态日志（人类可读）。支持 tail -f。
    也可同时打开一个 JSONL 附带日志，便于后续 pandas 分析（可选）。
    """
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, filename)

    logger = logging.getLogger("bandit_live")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if reset:
        # 以 'w' 打开并立即关闭 -> 清空文件
        open(log_path, "w", encoding="utf-8").close()

    # 用 WatchedFileHandler 便于 logrotate、tail -f 等
    fh = WatchedFileHandler(log_path, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False

    # 如果你也想要一个 JSONL（同一个文件更难读，这里默认不开）
    jsonl_path = os.path.join(save_dir, "bandit_live.jsonl")
    def log_jsonl(obj: dict):
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return logger, log_path, log_jsonl