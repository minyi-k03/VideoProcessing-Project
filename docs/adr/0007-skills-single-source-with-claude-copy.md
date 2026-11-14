---
status: accepted
---

# Skills 원본은 `.agents/skills/` 하나에 두고, Claude Code용 `.claude/skills/`는 스크립트로 만든 사본으로 쓴다

세 에이전트(Antigravity·Codex·Claude Code)가 같은 Skills로 일하려면 프로젝트 폴더에 Skills가 있어야 한다. Antigravity와 Codex는 `.agents/skills/`를, Claude Code는 `.claude/skills/`를 읽는다. 에이전트마다 따로 설치하면 사본이 갈라져 에이전트마다 다른 규칙으로 일하게 된다. 그래서 원본을 `.agents/skills/` 하나로 정하고, `scripts/sync_skills.py`가 `.claude/skills/`를 만든다. 사본은 직접 고치지 않는다.

## Consequences

- 원본을 고친 뒤 `python scripts/sync_skills.py`로 사본을 갱신한다. `python scripts/sync_skills.py --check`는 검증 명령에 들어 있어서 사본이 어긋나면 실패한다.
- 레지스트리 CLI로 바로 설치하지 않는다. CLI는 에이전트마다 사본이나 심볼릭 링크를 만들어 단일 원본을 깨뜨릴 수 있다. 원본 저장소의 `SKILL.md`와 동봉 파일을 검토한 뒤 `.agents/skills/`에만 넣고 sync한다.
- 출처 저장소·고정 커밋·라이선스는 README "Agent Skills" 표에 기록한다. 갱신은 원본 저장소의 diff를 검토한 뒤에만 반영한다.
- 새 Skill과 MCP 서버는 에이전트가 직접 설치하지 않고 사람에게 제안한다(AGENTS.md).
- 개인 전역 경로(`~/.claude/skills/`)에는 `find-skills`만 둔다. 프로젝트 Skills는 위 절차로만 늘린다.
