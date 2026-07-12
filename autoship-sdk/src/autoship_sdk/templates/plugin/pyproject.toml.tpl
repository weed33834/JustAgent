[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/$package_name"]

[project]
name = "$distribution_name"
version = "0.1.0"
description = "$description"
requires-python = ">=3.10"
dependencies = [
    "autoship>=0.2.0b1",
]

[project.entry-points."autoship.plugins"]
$plugin_name = "$package_name.plugin:register"

[project.urls]
Repository = "$repository_url"
