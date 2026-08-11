# README 以 LLM-led Teacher 为主的更新设计

日期：2026-07-22

## 目标

把仓库根 README 从旧的结构化 Teacher Agent baseline 说明，更新为当前实际演示的 LLM-led Teacher Prototype 说明，同时保留旧 baseline 的研究背景和运行入口。

## 读者

- 想快速运行当前演示的开发者和教师；
- 想理解 Chroma、bge-m3、DeepSeek 和确定性审计边界的项目评审者；
- 需要部署网页但不应公开内部向量库端口的运维人员。

## 内容结构

1. 项目简介和当前默认演示；
2. 核心特性；
3. 当前架构与单轮数据流；
4. RAG 的实际分工；
5. 一键快速启动；
6. 环境变量和端口；
7. 教学节奏与后置审计；
8. 测试和离线评测；
9. 旧结构化 Agent baseline；
10. 项目目录和开发约束。

## 必须准确表达的事实

- 当前用户入口是 LLM-led Streamlit 页面 8502；
- `scripts/run_llm_led_teacher.py` 同时启动 Chroma 8002 和网页 8502；
- 8002 是服务器内部 Chroma HTTP API，不是第三方付费 API，不应公开；
- bge-m3 在本地生成 query embedding，Chroma Server 使用 HNSW 检索本地向量库；
- 当前本地检索保留 BM25、bge-m3 和 RRF，并在低相似度时使用原联网回退策略；
- RAG 参考只进入教师请求，不进入 observer，也不能覆盖 active problem 或 turn guidance；
- observer 是软判断，确定性规范化和后置审计是节奏防线；
- 只有审计后的最终回答才能逐字符展示；
- 当前教师生成默认仍使用 DeepSeek API；
- 本地 DeepSeek-V4 GGUF 是独立实验，不是运行当前网页的必要条件；
- 旧结构化 Agent baseline 仍可通过 FastAPI 8000 和 Streamlit 8501 运行。

## 快速启动设计

主路径只给一个命令：

```bash
python3 scripts/run_llm_led_teacher.py
```

同时说明 `.env` 中需要 `DEEPSEEK_API_KEY`，以及向量库和 bge-m3 本地模型路径必须存在。用户只访问或转发 8502；8002 留在服务器内部。

## 验证

- 检查 README 中所有相对链接对应文件存在；
- 检查启动命令与脚本参数一致；
- 检查端口、环境变量、模型角色和 RAG 数据流没有矛盾；
- 不运行真实付费 API；
- 不修改 README 之外的生产文件。

## 非目标

- 不改变任何代码、检索阈值或启动参数；
- 不把旧 baseline 删除；
- 不把本地 DeepSeek-V4 写成当前默认模型；
- 不承诺未经实测的公网生产可用性。
