# JustAgent 双轨定位与架构设计文档

> 版本: v1.0 · 日期: 2026-08-22 · 状态: 已确认（用户拍板"两条都要"路线）
> 本文档是定位、架构分层、安全整改与路线图的单一事实来源。
> 执行冲突时以本文档为准；修改方向须先改此文档再动代码。

---

## 0. 定位陈述

**一句话**：JustAgent 是一个可审计的多 Agent 平台——开源引擎层卖技术（权限引擎、checkpoint、全程审计留痕），商业应用层卖场景（司法/法务垂直发行版，私有化部署）。

### 双轨定义

| 轨道 | 目标 | 受众 | 载体 |
|---|---|---|---|
| **开源轨** | 攒社区势能（star/贡献者/生态） | 开发者 | 引擎层：agent loop、MCP、checkpoint、审计 |
| **商业轨** | 变现验证 | 律所/企业法务（先）→ 法院（有渠道后） | 司法垂直发行版 + 未来闭源增值层 |

### 关键纪律（红线）

1. **叙事分离**：面向开发者的材料（README/文档首页）不出现法律领域词汇；面向法务的材料不暴露 repo_map、shadow git 等工程细节。
2. **代码分层**：引擎模块禁止 import judicial；judical 通过插件钩子注册进引擎。分层验收标准见 §3.4。
3. **安全先行**：任何"司法级/企业级"宣称在 §2 四项安全整改合并前禁止出现在任何对外文案中。

---

## 1. 现状盘点（2026-08-22，main = e33ca2e）

### 1.1 资产

- 成熟引擎内核：tool loop（`agent/runtime.py`）、Plan/Act/Yolo、checkpoint（shadow git）、session 持久化、权限规则（allow/deny/ask）、MCP 客户端、subagent、compaction
- Web 控制台：30+ `/api` 端点、SSE 流式聊天、多用户角色、多项目切换、ECharts 仪表盘
- 插件体系：pluggy + `justagent-sdk` 子包 + examples 五语言模板
- 测试规模大（README 称 2739 passed），CI 双 Python 版本矩阵全绿

### 1.2 债务（本设计要还的）

| # | 问题 | 证据 | 严重度 |
|---|---|---|---|
| D1 | 密码哈希为单次 HMAC-SHA256（无迭代因子），8 字节盐 | `src/justagent/web/users.py:55-58` | 高 |
| D2 | 未设 `JUSTAGENT_WEB_TOKEN` 时全部 `/api/*` 匿名开放 | `src/justagent/web/app.py:208-218` | 高 |
| D3 | `_require_write` 仅在"存在会话用户"时生效，匿名请求绕过角色检查 | `src/justagent/web/app.py:177-181` | 高 |
| D4 | 会话 token 存内存，进程重启全部失效 | `src/justagent/web/users.py:108-134` | 中 |
| D5 | 引擎与 judicial 存在 10+ 处直接耦合（清单见 §3.2） | `rg -l judicial src/justagent --glob '!judicial/**'` | 中 |
| D6 | CHANGELOG Unreleased 堆积 141 行未发版；pyproject 标注 `Production/Stable` 但无正式版本号 | `CHANGELOG.md` / `pyproject.toml` | 中 |
| D7 | README 测试数徽章为硬编码静态图片（tests-2739），非 CI 动态状态 | `README.md:13` | 低 |

---

## 2. 安全整改设计（P0，阻塞发版）

目标：Web 控制台达到"默认安全"，可作为商业化安全叙事的地基。全部改动限定在 `web/users.py` 与 `web/app.py`，不动 API 形状（前端零改动或极小改动）。

### 2.1 S1 密码哈希升级（修 D1）

方案：stdlib `hashlib.pbkdf2_hmac`（不引入新依赖，符合项目"优先标准库"约定）。

```
新存储格式: pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>
参数:       iterations=600_000（OWASP 2023 建议），盐 16 字节，输出 32 字节
```

- 兼容迁移：`_check()` 先识别格式前缀。遇到旧格式 `salt$digest` 校验通过后，立即用新格式重写该用户记录（透明升级，无需重置密码）。
- 新增常量集中在模块顶部，后续调参只改一处。
- 测试：`tests/test_web.py` 补三例——新格式登录成功 / 旧格式登录成功且触发迁移 / 错误密码不迁移。

### 2.2 S2 强制鉴权模式（修 D2/D3）

现状是"可选鉴权"，改为"至少一种鉴权必须激活"：

```
启动 justagent web 时按序判定：
1. users.json 存在任意用户          → 多用户模式（login 必需）
2. JUSTAGENT_WEB_TOKEN 非空        → 共享 token 模式
3. 两者皆无                         → 自动进入多用户模式：
                                     ensure_admin() 生成随机管理员密码，
                                     打印一次到终端，拒绝匿名访问
显式逃生口: --no-auth CLI flag（仅限本地开发，启动横幅红色警告）
```

中间件行为改为默认拒绝（default-deny）：

- 白名单路径仅保留 `/`、`/api/health`、`/api/auth/login`
- 其余 `/api/*` 无有效凭证一律 401
- 删除"`shared_token` 为空就放行"分支（app.py:214）；角色检查不再依赖"是否存在会话用户"的隐式判断（app.py:177-181 的 `_require_write` 重写为：解析不出用户 → 401；角色不足 → 403）

### 2.3 S3 Token 持久化（修 D4）

方案：token 落盘到 `~/.justagent/web_sessions.json`，复用已有 `utils/atomic_write.py`。

- 结构 `{token: {username, role, expires}}`；启动时惰性清理过期项
- 文件写入失败降级回内存模式并打警告（不阻塞启动）
- 明确不做的事：暂不上报注销事件、暂不做 refresh token——TTL 12h 保持不变，避免过度设计
- 安全备注：token 本身即凭证，文件权限继承用户目录；后续商业化若要求更高，换 signed cookie 方案，接口不变

### 2.4 S4 验收门禁

- [ ] 全部现有测试通过（`uv run pytest tests/ -v`）
- [ ] 新增安全测试 ≥ 8 例（S1 三例 + S2 匿名 401/低角色 403/admin 放行 + S3 重启后 token 仍有效）
- [ ] `--no-auth` 外的所有启动路径下，匿名请求 `/api/config` 返回 401（手工 curl 验证）
- [ ] ruff + mypy 干净

---

## 3. 架构分层设计

### 3.1 目标结构

```
justagent（单仓库，两阶段演进）
├── src/justagent/            ← 引擎层：零法律领域词汇
│   ├── agent/  core/  adapters/  checkpoint/  ...
│   └── plugins/              ← 内置插件（typecheck、security_scan…保持现状）
├── verticals/legal/          ← 司法垂直应用（阶段一：仓库内隔离）
│   ├── cases/ evidence/ documents/ knowledge/
│   ├── tools/                ← 注册为插件，经 hookspec 进入引擎
│   └── web_ext/              ← FastAPI router，由 web 挂载
└── justagent-sdk/            ← 保持独立子包（第三方插件作者入口）
```

命名说明：目录名用 `verticals/legal` 而非 `judicial`——"legal"对国际开发者更通用，也为未来其他垂直（如合规审查）留出命名空间。包内对外品牌仍可叫 JustAgent Legal。

### 3.2 现存耦合点清单（解耦工作项）

以下文件当前直接引用 judicial，需要逐个切断：

| 文件 | 耦合方式 | 解耦动作 |
|---|---|---|
| `agent/tools/builtin/__init__.py` | 直接注册 judicial 工具 | 改为插件 entry-point 注册 |
| `cli/commands/agent.py` | 启动时加载 judicial 工具 | 经插件管理器发现 |
| `cli/commands/judicial.py` | CLI 子命令本体 | 移入 verticals/legal，Typer 子应用挂载 |
| `web/app.py` | 司法端点内联在主 app | 抽成 `APIRouter`，vertical 提供、web 挂载 |
| `agent/evaluation.py` | 评估逻辑引用 judicial 场景 | 泛化为通用评估钩子 |
| `security/rbac.py`、`security/data_protection.py` | 角色示例/数据分级用法律术语举例 | 改为中性示例 |
| `locales/*.json` | 法律词条混入通用文案 | 拆分命名空间 |
| `__init__.py`、`utils/__init__.py` | 导出面提及 | 清理导出 |

### 3.3 插件化机制（复用已有 pluggy 基建）

- `hookspec.py` 增加 `register_tools(api)` / `register_cli(api)` / `register_web_routes(api)` 钩子（若已有等价钩子则复用）
- verticals/legal 在 `pyproject.toml` 声明 pluggy entry-point：`justagent_plugins = "justagent.verticals.legal:LegalPlugin"`
- 引擎启动流程只认 entry-point，不知道 legal 存在
- 收益：未来抽包（阶段二）只是移动目录 + 改发布配置，零逻辑改动

### 3.4 分层验收标准（CI 可执行）

```bash
# 引擎纯净性检查：以下命令必须零命中
rg -il "judicial|lawsuit|indictment|判决|卷宗|证据" src/justagent \
  --glob "!src/justagent/verticals/**"
```

加入 CI 作为独立 job（`layer-check`）。违规即红，防止分层腐化。

---

## 4. 发布与开源轨设计

### 4.1 v3.1.0（第一阶段收口）

触发条件：§2 安全整改合并 + 分层验收通过。

- CHANGELOG：Unreleased 全部归入 `[3.1.0] - <date>`；此后每个 PR 必须更新 Unreleased 区（CONTRIBUTING 加一条硬性要求）
- pyproject classifier 从 `5 - Production/Stable` 降为诚实值（`4 - Beta`，直到有外部用户反馈）
- 删除硬编码测试徽章（D7），替换为 GitHub Actions workflow status badge

### 4.2 README 重写要点（开源轨核心交付）

新叙事顺序：

1. 首屏：**可审计多 Agent 平台**——一句话卖点 = "每一次工具调用都有权限检查、checkpoint 和审计日志"
2. 30 秒 quickstart（编码场景，通用）
3. 特性表：引擎能力为主角
4. 垂直应用一节：JustAgent Legal 作为首个官方垂直发行版介绍，链接到单独文档页
5. 插件开发指南入口（SDK）

三语版本（en/zh/ja）同步改。

### 4.3 商业轨边界（现在只定规则，不写代码）

- 开源层：Apache-2.0 维持不变
- 未来增值层（闭源候选）：SSO/LDAP 对接、审计报表导出合规格式、法条数据订阅、技术支持
- 原则：增值层只做"集成与数据"，不做"引擎功能阉割"——避免社区反噬
- 首个付费验证场景：律所/企业法务的合同与证据审查私有化部署；法院渠道等资源到位再启动

---

## 5. 里程碑

| 里程碑 | 内容 | 完成标志 |
|---|---|---|
| **M0 安全整改** | §2 四项 | 验收门禁 §2.4 全绿，PR 合入 main |
| **M1 v3.1.0 发版** | 发版 + badge + classifier | tag v3.1.0，CHANGELOG 收口 |
| **M2 分层解耦** | §3.2 八项 + layer-check CI | CI 新 job 绿，引擎零法律词汇 |
| **M3 开源叙事** | README×3 重写 + 插件文档 | 文档合入，官网/README 不再有身份分裂表述 |
| **M4 垂直深耕** | 证据链审查做深（选定的打穿点） | 单一场景 demo 可跑通真实卷宗样例 |
| **M5 商业验证** | 1-3 家法务团队试点 | 一份试点反馈 + 报价单 |

依赖关系：M0 → M1；M2 可与 M1 并行但必须在 M3 前完成；M4/M5 依赖 M3。

---

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| 解耦 M2 触碰面广，可能引入回归 | 逐文件小 PR + 现有大测试盘兜底；layer-check 先加进 CI 再动代码（红→绿过程可见） |
| PBKDF2 60 万次迭代在高并发登录下 CPU 压力 | Web 控制台登录频率极低，可接受；预留参数化配置 |
| 双轨叙事再次漂移（历史惯性问题） | 本文档为唯一事实来源 + CI layer-check 机械兜底 |
| 个人维护者带宽有限，M4/M5 延期 | M0-M3 是自包含价值闭环（安全+发版+清晰叙事），允许在此暂停 |
