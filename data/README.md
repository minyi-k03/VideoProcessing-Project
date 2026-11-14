# 데이터 경로 안내 (ADR-0005)

이 디렉터리에는 대용량 원본 데이터(DIV2K, 테스트셋)와 패치 풀을 커밋하지 않습니다.

- 공유 Drive(및 백업용 Kaggle 비공개 Dataset)에 **패치 풀 + 테스트셋 압축 파일 1개**로 보관합니다.
- Colab/Kaggle 실행 시 세션 로컬 경로(예: `/content/data`)에 압축을 해제하고 CLI `--data-dir` 인자로 전달합니다.
- 압축 파일은 `scripts/prepare_data.py`로 만듭니다. 안의 구조:
  - `patch_pool/{hr.npy, lr.npy, metadata.json}`
  - `testsets/<데이터셋>/{HR,LR}/<image_id>.png` (HR은 배율의 배수로 자른 영상, 평가는 이 HR만 씁니다)
  - `testsets/metadata.json`
- 패치 풀 `metadata.json`에는 생성 조건과 식별자를 저장합니다: `patch_pool_id`, `scale`, `degradation`, `bicubic_rule`, `patch_size_hr`, `num_patches`, `pool_seed`, `source_image_count`, `content_sha256`.
