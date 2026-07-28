# JustAgent SDK

Plugin development SDK for [JustAgent-CLI](https://justagent.dev).

## Install

```bash
pip install justagent-sdk
```

## Quick start

```python
from justagent_sdk import Plugin, hook
from justagent.core.context import CommandContext

class MyPlugin(Plugin):
    @hook
    def pre_commit(self, context: CommandContext) -> None:
        print("About to commit")
```

## Scaffolding a plugin

```python
from justagent_sdk import create_plugin

create_plugin(
    target_dir="./justagent-hello",
    plugin_name="hello",
    description="A hello-world JustAgent plugin",
)
```
