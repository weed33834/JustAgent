# MyAgent SDK

Plugin development SDK for [MyAgent-CLI](https://myagent.dev).

## Install

```bash
pip install myagent-sdk
```

## Quick start

```python
from myagent_sdk import Plugin, hook
from myagent.core.context import CommandContext

class MyPlugin(Plugin):
    @hook
    def pre_commit(self, context: CommandContext) -> None:
        print("About to commit")
```

## Scaffolding a plugin

```python
from myagent_sdk import create_plugin

create_plugin(
    target_dir="./myagent-hello",
    plugin_name="hello",
    description="A hello-world MyAgent plugin",
)
```
