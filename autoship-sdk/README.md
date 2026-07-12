# AutoShip SDK

Plugin development SDK for [AutoShip-CLI](https://autoship.dev).

## Install

```bash
pip install autoship-sdk
```

## Quick start

```python
from autoship_sdk import Plugin, hook
from autoship.core.context import CommandContext

class MyPlugin(Plugin):
    @hook
    def pre_commit(self, context: CommandContext) -> None:
        print("About to commit")
```

## Scaffolding a plugin

```python
from autoship_sdk import create_plugin

create_plugin(
    target_dir="./autoship-hello",
    plugin_name="hello",
    description="A hello-world AutoShip plugin",
)
```
