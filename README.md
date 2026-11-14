# VideoProcessing-Project — x4 초해상화 개선 실험 파이프라인

> 상명대학교 영상처리 기말 프로젝트 (2인 팀).  
> 강의 개념에 근거해 ESPCN 베이스라인(RGB 입력, L1 손실)의 약점을 진단하고, 개선의 기여를 동일 공정 비교 조건·Ablation·시드 3회 반복으로 입증합니다.

## 문서 안내

- [요구사항 명세서 (`docs/SPEC.md`)](docs/SPEC.md)
- [도메인 용어집 (`docs/GLOSSARY.md`)](docs/GLOSSARY.md)
- [구현 계획·작업 분해도 (`docs/03-plan.md`)](docs/03-plan.md)
- 아키텍처 결정 기록 (`docs/adr/`)
  - [ADR-0001: 공정 비교 조건과 에폭 기반 학습 길이](docs/adr/0001-fair-comparison-conditions.md)
  - [ADR-0002: 베이스라인 ESPCN (RGB 입력, L1 손실)](docs/adr/0002-baseline-espcn-rgb-l1.md)
  - [ADR-0003: 평가 프로토콜 (BT.601 Y 기준 및 자기 베이스라인 대비 Δ)](docs/adr/0003-evaluation-protocol.md)
  - [ADR-0004: 개선 범위 (핵심 A·B, 탐색 C, 통제 B′)](docs/adr/0004-improvement-scope-core-and-exploratory.md)
  - [ADR-0005: 실행 환경(VS Code Colab T4) 및 결과 저장 규칙](docs/adr/0005-execution-environment-and-result-storage.md)
  - [ADR-0006: 보고서·발표의 모든 수치는 실행 기록에서 스크립트로 생성](docs/adr/0006-report-numbers-from-logs-only.md)
  - [ADR-0007: Skills 원본은 `.agents/skills/` 하나, `.claude/skills/`는 동기화 사본](docs/adr/0007-skills-single-source-with-claude-copy.md)

## 저장소 구조

```text
.
├── configs/                  # 공정 비교 조건(단일 진실 소스) 및 실험 구성 스위치 정의
├── data/                     # 런타임 로컬 데이터 경로 안내 (실제 데이터는 Git 미추적)
├── docs/                     # 요구사항 명세서, 용어집, ADR
│   └── adr/
├── notebooks/                # Colab T4 / Kaggle 얇은 실행 창구 노트북 (로직 미포함)
├── results/
│   ├── aggregated/           # main 병합 기록 기반 자동 집계 표·그림
│   ├── pilot/                # 1주차 진단 및 파일럿 실행 결과 (본 결과 표 제외)
│   └── runs/                 # 본 실행 기록 (summary, per_image, 대표 이미지)
├── scripts/                  # CLI 실행 진입점
├── src/sr_pipeline/          # 공용 규칙·데이터·모델·손실·평가·진단·집계·데모 모듈
└── tests/                    # 로컬 CPU 단위·경계 테스트
```

## 작업 흐름

1. 로컬에서 각자 이름 브랜치를 생성·체크아웃해 개발 및 로컬 CPU 테스트(`pytest`)를 실행합니다.
2. `notebooks/`의 얇은 실행 노트북으로 Colab T4 런타임에서 브랜치를 가져와 학습·평가를 실행합니다.
3. 실행 기록(`results/runs/`)은 실행자 브랜치에 커밋하고 PR 리뷰를 거쳐 `main`에 병합합니다.
4. 보고서·발표의 모든 수치와 그림은 `main`에 병합된 실행 기록을 `scripts/aggregate_results.py`로 자동 생성한 산출물만 사용합니다.

## Agent Skills

원본은 `.agents/skills/`, Claude Code용 사본은 `python scripts/sync_skills.py`로 만든 `.claude/skills/`입니다(사본 직접 수정 금지). 업데이트는 원본 저장소의 diff를 검토한 뒤에만 반영합니다.

| 스킬 | 원본 저장소 | 고정 커밋 | 라이선스 | 용도 |
| --- | --- | --- | --- | --- |
| `setup-matt-pocock-skills` | mattpocock/skills | `4ae2317` | MIT | 이슈 트래커·문서 위치 설정 (레포당 1회) |
| `to-tickets` | mattpocock/skills | `4ae2317` | MIT | SPEC → 작업 티켓 분해 |
| `implement` | mattpocock/skills | `4ae2317` | MIT | 티켓 기반 구현 |
| `tdd` | mattpocock/skills | `4ae2317` | MIT | 테스트 우선 구현 |
| `codebase-design` | mattpocock/skills | `4ae2317` | MIT | `tdd`가 참조하는 모듈·seam 용어 |
| `code-review` | mattpocock/skills | `4ae2317` | MIT | 표준·spec 두 축 리뷰 |
| `diagnosing-bugs` | mattpocock/skills | `4ae2317` | MIT | 버그 진단 루프 |
| `writing-for-agents` | mattpocock/skills | `4ae2317` | MIT | AGENTS.md·CLAUDE.md·스킬 작성 원칙 (참고용) |
| `verification-before-completion` | obra/superpowers | `22f0b46` | MIT | 완료 주장 전 검증 실행 |
| `finishing-a-development-branch` | obra/superpowers | `22f0b46` | MIT | 브랜치 마무리(병합·PR) |
| `pptx` | anthropics/skills | `2c5958e` | Apache-2.0 | 발표 자료 |

## 로컬 환경 확인

```bash
pip install -r requirements.txt
pytest
```
