from gpt_client import run_from_config

if __name__ == "__main__":
    text = run_from_config(
        config_path="config.json",
        kind="A",  # 选择 A/B/C
        variables={
            "number": 20
        },
        system="输出要简洁。",
        temperature=0.2,
    )
    print(text)