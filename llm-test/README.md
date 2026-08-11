# 中考真题 LLM 测评系统

用两个不同的 LLM 分别扮演"考生"和"阅卷老师"，自动完成答题和评分。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API

编辑 `config.py`，填入教师模型和裁判模型的 API 信息：

```python
TEACHER_MODEL = {
    "base_url": "https://api.openai.com/v1",  # API 地址
    "api_key": "sk-xxxxx",                     # API 密钥
    "model_name": "gpt-4o",                    # 模型名称
    ...
}

JUDGE_MODEL = {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-xxxxx",
    "model_name": "gpt-4o",
    ...
}
```

### 3. 运行

```bash
# 测评全部7科
python evaluator.py

# 只测评单科
python evaluator.py 数学
python evaluator.py 语文
```

## 目录结构

```
├── config.py          # 配置文件（API密钥等）
├── llm_client.py      # LLM 客户端封装
├── evaluator.py       # 主测评逻辑
├── testmd/paper/      # 试卷（7份）
├── testmd/key/        # 答案（7份）
└── results/           # 测评结果输出
```

## 工作流程

1. **考生模型** 读取试卷内容，逐题作答
2. **阅卷模型** 拿到考生答案 + 参考答案及评分标准，严格按标准打分
3. 结果保存为 JSON（结构化数据）和 Markdown（人类可读）
