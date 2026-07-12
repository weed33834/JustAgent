# AutoShip-CLI Project Guard 插件示例

一个真实可用的 AutoShip 插件示例，用于在提交前和验证前做项目健康检查。

## 功能

- **pre_commit**: 扫描暂存区 Python 文件，提示 TODO/FIXME/XXX/HACK、行尾空白、缺失末尾空行。
- **pre_verify**: 对项目 Python 文件做快速 `py_compile` 语法检查。
- **on_error**: 当 `autoship verify --fix` 失败时建议先运行 `autoship clean`。

## 安装

```bash
cd examples/custom-plugin
pip install -e .
```

`pyproject.toml` 中的 `[project.entry-points."autoship.plugins"]` 会自动注册该插件。

## 验证

```bash
# 在一个 Git 仓库中创建带 TODO 的 Python 文件
cd /path/to/your-project
echo "# TODO: fix this" > demo.py
git add demo.py
autoship commit -m "test"
```

你会看到 `[project-guard]` 提示暂存区中存在 TODO。

```bash
# 触发 verify --fix 建议
autoship verify --fix "python -c 'raise RuntimeError'"
```

## 测试

```bash
pytest
```
