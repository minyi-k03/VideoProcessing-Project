# 데이터 경로 안내 (ADR-0005)

이 디렉터리에는 대용량 원본 데이터(DIV2K, 테스트셋)와 패치 풀을 커밋하지 않습니다.

- 공유 Drive(및 백업용 Kaggle 비공개 Dataset)에 **패치 풀 + 테스트셋 압축 파일 1개**로 보관합니다.
- Colab/Kaggle 실행 시 세션 로컬 경로(예: `/content/data`)에 압축을 해제하고 CLI `--data-dir` 인자로 전달합니다.
- 패치 풀 생성 시 생성 조건(`scale`, `degradation`, `bicubic_rule`, `patch_size`, `num_patches`, `pool_seed`, `patch_pool_id`) 메타데이터를 함께 저장합니다.
