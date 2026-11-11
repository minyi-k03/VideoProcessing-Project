# AGENTS.md

x4 초해상화 개선 실험 파이프라인. 이 과제의 산출물은 "성능"이 아니라 **입증**이다: 모든 구성이 같은 공정 비교 조건에서 학습되었고, 보고서의 모든 수치가 실행 기록까지 추적된다. 아래 규칙은 대부분 이 입증을 지키기 위한 것이다.

## 기준 문서

- 구현할 것과 완료 조건: `docs/SPEC.md`
- 용어: `docs/GLOSSARY.md`. 코드·주석·문서·PR에서 정의된 용어를 쓰고, 각 항목의 _Avoid_ 단어는 쓰지 않는다(예: "GT"가 아니라 HR 영상, "런"이 아니라 실행).
- 결정과 그 이유: `docs/adr/`. 구현이 ADR과 어긋나야 할 것 같으면 코드를 바꾸기 전에 멈추고 보고한다.
- Task·담당·상태: `docs/03-plan.md`. 세션을 시작하면 이 파일의 "현재 상태"와 해당 Task 행, 최근 git log를 먼저 읽는다.
- 문서끼리 충돌하면 추측하지 말고 충돌 내용을 보고한다.

## 공정 비교 조건 (가장 중요)

- 공정 비교 조건의 값은 `configs/fair_comparison.yaml` 한 곳에만 있다. 구성(`configs/experiments.yaml`)은 `trainable`, `promoted_only`, 개선 스위치(A·B·C·Bp)만 가진다.
- `pilot_determined`의 `null`은 파일럿 실행 후 Task T4.2에서 팀이 한 번에 고정한다. 그 전에는 테스트·합성 데이터용 값을 테스트 코드 안에서만 쓴다.
- 고정 후 공정 비교 조건을 바꾸면 이전 실행 전부가 비교 불가능해진다(ADR-0001). 편의·속도를 위한 변경이 필요해 보이면 제안만 하고 사람의 결정을 기다린다.
- **⚠ 변경**: 공정 비교 조건, 손실, 데이터 준비·샘플러, 평가, 색 변환·bicubic 규칙을 건드리는 변경. 커밋·PR 설명 첫 줄에 `⚠`와 바뀐 규칙을 적고, 상대 팀원 승인 전에는 main에 병합하지 않는다.

## 구현 규칙

- 모든 RGB↔YCbCr 변환은 `sr_pipeline.common.color`, 모든 bicubic 축소·확대는 `sr_pipeline.common.bicubic`을 거친다. `F.interpolate`·PIL·OpenCV·skimage의 resize/색 변환 기본값은 커널·좌표·안티에일리어싱·계수가 달라서, 그 차이가 개선 효과로 오해된다.
- 손실은 자르지 않은 출력으로 계산한다. [0, 1] 자르기와 8비트 양자화는 평가·저장 단계(`evaluation/postprocess.py`)에서만 한다.
- 모델의 바깥 인터페이스는 구성과 관계없이 "RGB LR → RGB SR(자르기 전)"이다. 스위치별 분기는 모듈 안에 둔다. 모든 스위치가 꺼진 구성이 곧 베이스라인이며, 베이스라인 전용 코드 경로는 만들지 않는다.
- 로직은 `src/sr_pipeline/`에 둔다. `notebooks/`는 브랜치 가져오기 → 데이터 준비 → `scripts/` 실행 → 실행 기록 커밋만 하는 얇은 실행 창구다.
- 데이터·출력·체크포인트 경로는 실행 인자로 받는다. Colab, Kaggle, 로컬 Windows가 같은 코드를 쓴다.
- 코드는 device를 인자로 받아 CPU에서도 돈다. 테스트는 로컬 CPU, 학습은 Colab T4, 데모는 CPU 대체 경로를 가진다.
- 실행 기록 필드 이름은 SPEC "실행 기록 형식"의 영문 식별자로 고정한다.

## 결과와 수치

- 보고서·발표·README에 들어가는 수치·표·그림은 `scripts/aggregate_results.py`가 main에 병합된 실행 기록에서 만든 `results/aggregated/` 산출물만 쓴다. 수치를 손으로 적거나 로그에서 옮겨 적지 않는다.
- 진단·파일럿 실행은 `results/pilot/`, 본 실행은 `results/runs/`에 둔다. 파일럿 결과는 결과 표에 넣지 않는다.
- 데이터, 패치 풀, 체크포인트(`*.pth`, `*.pt`)는 Drive·Kaggle에 둔다. 예외는 ADR-0005의 Drive 대체 경로뿐이다.
- 동결(12/5) 이후에는 코드와 실행 기록을 바꾸기 전에 사람의 승인을 받는다.

## 협업

- 각자 이름 브랜치에서 작업한다. main은 PR과 승인 1명으로만 바뀐다. 상대 팀원의 변경이 main에 병합되면 내 브랜치에 바로 병합한 뒤 다음 실행을 한다.
- 공통 기반(`configs/`, `src/sr_pipeline/common/`, `src/sr_pipeline/evaluation/`, 의존성, 이 파일)은 `docs/03-plan.md` 담당 표의 공통 기반 담당이 바꾼다. 담당이 아니면 변경을 제안으로 보고한다.
- Task 상태가 바뀌면 같은 커밋에서 `docs/03-plan.md`의 상태 열을 고친다. Task를 지우거나 완료 조건을 바꿔야 하면 사람에게 묻는다.
- 저장소는 Public이다. push 토큰은 실행할 때 `getpass`로만 받고, 파일·노트북 출력·로그에 남기지 않는다.

## 테스트와 검증

검증 명령 (로컬 CPU, 수 분 안에 끝나야 한다):

```bash
pip install -r requirements.txt
pytest
python scripts/sync_skills.py --check
```

- 현재 테스트 19개는 `@pytest.mark.skip` 상태다. 모듈을 구현하면 그 모듈의 테스트 `skip`을 풀고 통과시키는 것까지가 Task 완료다.
- 테스트는 바깥에서 보이는 동작을 작은 합성 데이터로 검증한다. 테스트 경계는 SPEC "Testing Decisions"를 따른다.
- 테스트를 통과시키려고 테스트의 기대값이나 공정 비교 조건을 바꾸지 않는다. 기대값이 틀렸다고 판단되면 근거와 함께 보고한다.
- GPU가 필요한 확인(학습, 추론 시간)은 로컬에서 실행할 수 없다. 이 경우 "Colab 실행 필요"로 따로 보고한다.

## Skills

- 원본은 `.agents/skills/`다. `.claude/skills/`는 `python scripts/sync_skills.py`로 만드는 사본이라 원본만 고친다.
- 새 Skill이나 MCP 서버는 직접 설치하지 않고 제안한다.
- 완료를 주장하기 전에 `verification-before-completion` 스킬을 쓴다.
- `setup-matt-pocock-skills`가 만드는 `## Agent skills` 블록은 이 파일에 둔다.

## 다중 에이전트

- 이 파일이 Antigravity·Claude Code·Codex 공통 규칙의 단일 기준이다. `CLAUDE.md`는 이 파일을 import하고 Claude Code 전용 규칙만 더한다.
- `GEMINI.md`, `AGENTS.override.md`는 팀 리뷰와 ADR 없이 만들지 않는다.

## 완료 보고

1. 바뀐 내용과 목적, 주요 파일
2. 실제로 실행한 검증 명령과 결과 (실행하지 못한 검사는 따로)
3. ⚠ 변경 여부와 승인이 필요한 사람
4. Colab 실행이 필요한 확인 항목
5. 남은 문제와 위험
