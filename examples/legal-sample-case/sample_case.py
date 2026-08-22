"""JustAgent Legal — 真实卷宗样例：证据链审计演示。

场景：买卖合同纠纷（借款替代场景同样适用）。
卷宗包含 5 项证据，人为埋入 4 类典型问题：

1. 扣押收集的账本没有保管链条记录          → 保管链条问题
2. 监控视频的保管链条环节日期乱序           → 保管链条问题
3. 一份证人证言的收集日期晚于立案日期        → 时间线问题
4. 两份合同扫描件同源（同一扫描文档）却被标为相互佐证 → 同源佐证警告
5. "逾期付款利息""解除租赁合同"两项诉讼请求无任何证据支持 → 诉请覆盖缺口

运行：
    uv run python examples/legal-sample-case/sample_case.py

无需 LLM、无网络——全部检查为确定性规则。
"""

from __future__ import annotations

from justagent.verticals.legal.case_manager import CaseFile, Claim, Party, PartyRole, TimelineEvent
from justagent.verticals.legal.evidence import (
    CustodyEvent,
    Evidence,
    EvidenceAuditor,
    EvidenceChain,
    EvidenceRelationType,
)


def build_sample() -> tuple[CaseFile, EvidenceChain]:
    case = CaseFile(
        case_number="（2025）京01民初777号",
        cause_of_action="买卖合同纠纷",
        court="北京市第一中级人民法院",
        domain="civil",
        parties=[
            Party(name="北京宏图贸易有限公司", role=PartyRole.PLAINTIFF),
            Party(name="李某", role=PartyRole.DEFENDANT),
        ],
        claims=[
            Claim(description="判令被告支付货款100万元"),
            Claim(description="判令被告承担逾期付款利息"),
            Claim(description="解除双方场地租赁合同"),
        ],
        timeline=[
            TimelineEvent(description="合同签订", timestamp=1735689600.0),  # 2025-01-01
            TimelineEvent(description="立案", timestamp=1767225600.0),  # 2026-01-01
        ],
    )

    chain = EvidenceChain()
    ev = []

    def add(name: str, **kw: object) -> None:
        e = chain.add_evidence(Evidence(name=name, case_id=case.id, **kw))  # type: ignore[arg-type]
        ev.append(e)

    # ① 合法完整的核心书证（对照组）
    add(
        "购销合同原件",
        type="documentary",
        description="双方签订的购销合同，约定货款100万元及付款期限。",
        collector="张律师",
        source="原告提供",
        collection_method="当事人提供",
        collection_date="2025-12-20",
        proving_object="买卖合同的订立与货款金额约定",
    )
    # ② 问题一：扣押收集但无保管链条
    add(
        "公司账簿（扣押）",
        type="documentary",
        description="从被告公司财务室扣押的原始账簿，记载欠款金额。",
        collector="办案队",
        source="财务室扣押",
        collection_method="扣押",
        collection_date="2025-12-25",
        proving_object="被告欠付货款的事实",
    )
    # ③ 问题二：保管链条乱序
    add(
        "监控录像（调取）",
        type="audio_visual",
        description="被告仓库门口监控，显示货物签收过程。",
        collector="办案队",
        source="物业机房调取",
        collection_method="调取",
        collection_date="2025-12-26",
        custody_chain=[
            CustodyEvent(date="2026-01-05", actor="法制科", action="移交"),
            CustodyEvent(date="2025-12-26", actor="办案队", action="收集"),
        ],
        proving_object="原告已交付货物的事实",
    )
    # ④ 问题三：收集日期晚于立案
    add(
        "证人证言（王某）",
        type="testimony",
        description="王某称曾听李某承认拖欠货款。",
        collector="孙法官助理",
        source="法院询问笔录",
        collection_method="当事人提供",
        collection_date="2026-02-10",
        proving_object="被告承认拖欠的意思表示",
    )
    # ⑤ 问题四：两份同源扫描件被互相佐证 + 问题五对照用不到它
    add(
        "合同扫描件A",
        type="documentary",
        description="合同首页扫描件。",
        source_document_id="scan_bundle_01",
        collector="张律师",
        source="原告提供",
        collection_method="当事人提供",
        collection_date="2025-12-21",
        proving_object="买卖合同的订立与货款金额约定",
    )
    add(
        "合同扫描件B",
        type="documentary",
        description="合同签章页扫描件。",
        source_document_id="scan_bundle_01",
        collector="张律师",
        source="原告提供",
        collection_method="当事人提供",
        collection_date="2025-12-21",
        proving_object="印章印文的真实性",
    )

    chain.add_relation(ev[4].id, ev[5].id, EvidenceRelationType.CORROBORATES)
    return case, chain


def main() -> None:
    case, chain = build_sample()
    filing = next(e.timestamp for e in case.timeline if e.description == "立案")
    filing_date = __import__("datetime").date.fromtimestamp(filing).isoformat()

    audit = EvidenceAuditor(chain).audit_case(
        case.id, claims=list(case.claims), filing_date=filing_date
    )

    print(f"案件：{case.case_number}  {case.cause_of_action}")
    print("=" * 60)
    for title, items in (
        ("保管链条问题", audit.custody_issues),
        ("时间线问题", audit.timeline_issues),
        ("同源佐证警告", audit.independence_warnings),
    ):
        print(f"\n[{title}] {len(items)} 项")
        for item in items:
            print(f"  • {item}")
    print(
        f"\n[诉请覆盖] {sum(c.covered for c in audit.claim_coverage)}/{len(audit.claim_coverage)}"
    )
    for c in audit.claim_coverage:
        mark = "✅" if c.covered else "❌"
        print(f"  {mark} {c.claim_description}" + (f" —— {c.note}" if c.note else ""))
    print("\n" + "=" * 60)
    print(audit.summary)


if __name__ == "__main__":
    main()
