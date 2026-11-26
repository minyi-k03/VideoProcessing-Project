# 구현 계획 — 작업 분해도 (WBS)

> 기준 문서: [`docs/SPEC.md`](SPEC.md) (초안 v3), ADR-0001 ~ 0005. 용어는 [`docs/GLOSSARY.md`](GLOSSARY.md)를 따른다.
> 작성: 2026-11-11. ADR-0004는 제안 상태라 11/17 진단 결과에 따라 M3 이후 Task가 바뀔 수 있다.

## 1. 사용법

- 작업 단위는 이 파일 하나로 관리한다. Task 상태가 바뀌면 작업자가 자기 브랜치에서 이 표를 고쳐 함께 커밋한다.
- **상태 값:** `대기` · `진행` · `통과` · `보류`
- **통과 조건 (Harness §3 ①):**
  1. 완료 조건을 충족한다.
  2. 관련 테스트의 `skip`을 풀고 `pytest`가 통과한다.
  3. 실제 실행 결과(로그·실행 기록)로 확인한다.
  4. 변경 위험에 맞는 리뷰를 받는다.
- **⚠ 표시:** 공정 비교 조건·손실·데이터·평가·색/보간 규칙을 바꾸는 Task. PR에 그 사실을 적고 상대 팀원의 승인을 받는다(US69).
- **US:** SPEC 사용자 스토리 번호

### 현재 상태 (11/23 기준)

- T0.1 통과: 두 팀원 모두 VS Code Colab 확장에서 T4 연결, Drive 마운트, 원격 push를 확인했다(차가원 11/15, 점검용 브랜치 push·삭제). 대체 경로는 쓰지 않는다.
- `models/espcn.py` 베이스라인(RGB, 은닉층 tanh, 마지막 층 활성화 없음, x4 sub-pixel)이 구현되어 `tests/test_espcn.py`(바깥 동작만 검증)를 통과한다(T2.1 통과, PR #4 리뷰 반영, main 병합은 고용민 승인 대기). 파라미터 수는 RGB 37,200개, Y 24,752개다. 배율은 x4만 받고 float RGB 입력만 받는다. 초기화는 은닉층 N(0, √(2/(out·k²))), 마지막 층 N(0, 0.001), 편향 0으로 모든 구성이 같다. 스위치 B·C 경로는 T3.4에서 구현한다.
- `losses/sr_loss.py`의 기본 L1(자르지 않은 SR 결과와 HR RGB의 배치·채널·화소 평균)이 구현되어 `tests/test_sr_loss.py`를 통과한다(T2.2 ⚠ 통과, main 병합은 고용민 승인 대기). 스위치 A·B·B′를 켜면 생성 시점에 실패하며, 이 경로는 T3.5에서 구현한다.
- `diagnosis/diagnostics.py`의 진단 5종(Y 오차 지도, HR Y Sobel 상위 q% 에지·평탄 PSNR(Y), 2D FFT 반지름 스펙트럼·대역 에너지 비, YCbCr 채널별 MSE 비중, 테스트셋별 최악 영상)과 `run_diagnosis_suite`(입력 `<HR 루트>/<데이터셋>/HR/<image_id>.png`, `<SR 루트>/<데이터셋>/<image_id>.png`, 가장자리 4화소 자르기 후 CSV 3개·그림 저장)가 구현되어 `tests/test_diagnostics.py`를 통과한다(T2.3 통과, main 병합은 고용민 승인 대기). PSNR·에지 영역 계산은 T2.4 전이라 진단 모듈 안 비공개 함수로 두었고, T3.6에서 metrics와 공유할지 정한다.
- T2.5 중 진단 연결이 끝났다(차가원, PR #9로 main 병합 완료). `scripts/run_pilot_diagnosis.py` → `diagnostics.run_pilot_diagnosis`: `shave_border`는 `fair_comparison.yaml` `fixed_rules`에서만 읽고, `edge_top_q`는 `pilot_determined` 값이 `null`이면 `--edge-top-q`가 필수, 확정 후에는 파일 값과 달라서는 안 된다. 쓴 값과 출처는 `<output-dir>/diagnosis_settings.json`에 남는다(Public 저장소라 입력 경로는 기록하지 않음). 설정 파일 구조·`shave_border`(0 이상 정수)·`edge_top_q`(숫자, bool 제외)가 잘못되면 산출물을 만들기 전에 `ValueError`로 멈춘다. 파일럿 학습 경로도 구현했다(고용민, 로컬 CPU 합성 데이터 테스트 `tests/test_pilot.py` 통과): `engine/pilot.py`의 `train_pilot_baseline`(`ESPCN()`+`SRLoss()` L1, Adam, 스케줄 없음, 8방향 증강(90° 회전 4 × 좌우 뒤집기 2)을 HR·LR에 같게 적용, 데이터 난수는 `create_data_rng`만 사용)과 `generate_sr_results`(테스트셋 LR → 자르기·8비트 양자화만 거친 SR PNG, 가장자리는 진단이 자름. 출력 폴더가 이미 있으면 거부하고 임시 폴더에 다 쓴 뒤 옮겨 실패 시 반쪽 산출물이 남지 않음), `scripts/train_pilot_baseline.py`, `notebooks/t2_5_pilot_colab.ipynb`. 에폭·에폭당 패치·배치·학습률·옵티마이저·스케줄·증강은 `pilot_determined`가 `null`인 동안 기본값 없이 필수 인자이며, 쓴 값과 실행 시점 커밋은 `results/pilot/pilot_train_settings.json`에 남는다(경로 제외). `src/`·`scripts/`·`configs/`에 커밋되지 않은 변경이 있으면 학습을 시작하지 않으며(`--allow-dirty`면 `-dirty`를 붙여 기록), 진단 CLI와 같은 `diagnosis/provenance.resolve_code_commit`을 쓴다. CLI는 학습 전에 설정 파일 구조·테스트셋 LR 폴더·SR 폴더 부재를 확인한다. 다른 에이전트 교차 검증(19건)을 반영했다. 이 경로는 파일럿 전용이라 T3.3 샘플러·T3.7 트레이너가 구현되면 대체된다. Colab T4에서 파일럿 베이스라인(시드 0, 30에폭, 에폭당 10,000패치, 배치 16, 학습률 1e-3)을 학습하고 진단을 돌려 `results/pilot/`에 커밋했다(T2.5 통과, main 병합은 차가원 승인 대기). 에폭 30 등은 진단용 파일럿 값이며 `pilot_determined`의 확정값이 아니다(본 실행의 에폭은 T4.1·T4.2에서 정한다). 학습 곡선과 진단 수치는 `results/pilot/pilot_train_settings.json`·`summary_diagnosis.csv`에 있다(여기에 옮겨 적지 않는다). `results/pilot/`의 이전 1에폭 작동 확인 결과는 30에폭 결과로 덮어썼다. 진단은 자기 산출물만 원자적으로 교체하며(PR #11), 학습은 기존 SR 폴더를 거부하므로 노트북 셀 3-1이 재실행 전에 이전 SR 폴더(`SR_DIR`)만 지운다.
- `evaluation/postprocess.py`(`quantize_and_shave`, `postprocess_to_y`: [0, 1] 자르기 → 8비트 양자화 → 가장자리 잘라내기 → `common.color.rgb_to_y`, `border` 기본값 없이 호출자가 `fair_comparison.yaml` 값을 넘김)와 `evaluation/metrics.py`(`compute_psnr`, `compute_ssim_y`: BasicSR/MATLAB 호환 11×11 가우시안 σ=1.5 valid 합성곱, float64 지표 연산)가 구현되어 `tests/test_evaluation.py`의 T2.4 경계 테스트를 통과한다(T2.4 ⚠ 통과, main 병합은 차가원 승인 대기). 모든 구성이 같은 평가 경로를 거치는지 검증과 `compute_edge_psnr`·`compute_lpips`·`evaluator.py`는 T3.6에서 구현한다.
- 모델 스위치 B·C가 구현되었다(T3.4 통과, main 병합은 고용민 승인 대기). B: LR을 YCbCr로 바꿔 정규화된 Y만 네트워크로 복원하고 Cb·Cr는 bicubic 확대 후 RGB로 되돌리며 `sr_y_norm`을 함께 반환한다. C: 네트워크가 다루는 채널(RGB 또는 정규화된 Y)의 bicubic 확대 결과에 예측 차이를 더한다.
- 손실 스위치 A·B·B′가 구현되었다(T3.5 ⚠ 통과, main 병합은 고용민 승인 대기). A: L1 + λ·(채널별 Sobel 수평·수직 성분 차이 L1, 복제 패딩, 정규화하지 않은 표준 커널). B: 모델의 `sr_y_norm`, B′: RGB SR 결과를 정규화된 Y로 바꿔 손실 계산. 구성 스위치 대응은 A → `SRLoss(use_edge_loss=True, edge_loss_lambda=λ)`, B → `ESPCN(use_y_channel=True)` + `SRLoss(use_y_loss=True)`, C → `ESPCN(use_global_residual=True)`, Bp → `SRLoss(use_y_loss=True)`이며 T3.2·T3.7에서 이 대응을 쓴다. ADR-0004가 아직 proposed라 T3.1에서 개선이 바뀌면 해당 경로를 고친다.
- `data/sampler.py`·`dataset.py`의 샘플러가 구현되어 `tests/test_data_sampler.py`와 `tests/test_config_and_sampler.py`의 샘플러 테스트를 통과한다(T3.3 ⚠ 통과, main 병합은 차가원 승인 대기). `SeededEpochPatchSampler(pool_size, patches_per_epoch, seed, *, augmentation)`가 `create_data_rng`만으로 에폭마다 `(patch_idx, aug_code)`를 정하고, `epoch_plan(epoch)`은 에폭을 직접 지정해도 처음부터 순차로 돌린 것과 같다(T3.7 재개 대비). `PatchPoolDataset(data_dir, patch_pool_id)`는 풀 ID가 다르면 거부하고, 배율은 풀 `metadata.json`의 `scale`을 쓰며(없거나 잘못되면 거부, 모델 배율과의 대조는 호출자가 한다), `iter_epoch_batches`로 (LR, HR) 배치를 낸다. 난수 소비 순서는 T2.5 파일럿과 같다(테스트가 시드 0의 에폭 계획을 기준 값으로 고정하고, 리팩터링 전후 파일럿 학습의 손실·가중치가 합성 풀에서 비트 단위로 같음을 일회성 비교로 확인했다). `engine/pilot.py`는 이 공용 모듈을 쓰도록 바뀌었고 `sample_epoch_batches`는 사라졌다. `TestsetDataset`은 T3.6에서 구현한다.
- `config/schema.py`가 구현되어 `tests/test_config_and_sampler.py`의 T3.2 테스트를 통과한다(T3.2 ⚠ 통과, main 병합은 차가원 승인 대기). `load_fair_comparison`은 구조(최상위 3절, 필수·알 수 없는 키, 고정 값의 `null`, `representative_images`의 데이터셋 집합)만 검증하고 `pilot_determined`의 `null`은 허용한다. `find_undetermined`가 `null`인 항목의 점 경로를 돌려주며 T4.2의 "`null` 0개" 확인에 쓴다. `load_experiment_config`는 `trainable`·`promoted_only`·`switches`(A·B·C·Bp, bool)만 받고, 공정 비교 조건 이름 키는 덮어쓰기 시도로 거부한다. `resolve_run_config`는 시드가 `fixed_rules.seeds`에 없거나 `pilot_determined`에 `null`이 남아 있으면 `ValueError`로 멈추므로, **T4.2 전에는 실제 `fair_comparison.yaml`로 확정 실행 조건을 만들 수 없다**(T3.7·T3.8 개발과 T3.9 e2e는 테스트 코드 안 합성 값으로 채운 fair 파일을 쓴다). 확정 실행 조건의 `fair_comparison`은 깊은 복사본이다. `diagnostics.py`·`pilot.py`의 자체 YAML 검증은 그대로 둔다.
- `engine/trainer.py`가 구현되어 `tests/test_trainer.py`(로컬 CPU, 합성 패치 풀)를 통과한다(T3.7 통과: 로컬 CPU 테스트와 Colab T4 확인을 모두 통과, 확인용 노트북 `notebooks/t3_7_trainer_colab_check.ipynb` — 학습 구성 9개의 GPU 결정성, 재개 후 일치, 상한 `incomplete`). `train_run(run_config, data_dir, output_dir, checkpoint_dir, *, commit, resume, device, clock)`은 확정된 실행 조건으로 에폭 학습을 하고, 매 에폭 끝에 `<checkpoint_dir>/<run_id>/state.pt`(모델·옵티마이저·에폭 손실·누적 학습 시간·run_id·학습 설정 해시)를 원자적으로 저장하며 끝나면 마지막 에폭 가중치 `weights.pt`(평가 체크포인트)와 `<output_dir>/<run_id>/train_log.json`(`run_id`, `commit`, `config_id`, `seed`, `patch_pool_id`, `switches`, `params`, `status`, `epochs_done`, `epochs_total`, `train_minutes`, `epoch_losses` — 경로는 기록하지 않음)을 쓴다. 재개는 `resume=True`이고 run_id와 학습 설정 해시가 같을 때만 되며(런타임이 끊긴 뒤 재개한 가중치·손실이 한 번에 끝까지 돈 것과 비트 단위로 같음을 테스트가 확인), `resume=False`인데 같은 run_id의 `state.pt`·`weights.pt`·`train_log.json` 중 하나라도 있으면 거부한다. 커밋되지 않은 변경으로 시작한 `-dirty` 실행과 학습 없는 구성(bicubic)은 재개를 거부한다. `max_hours`(0보다 큰 유한한 수)와 `patch_size_hr`(패치 풀 HR 패치 크기와 일치, `PatchPoolDataset.hr_patch_size`로 확인)은 파일을 쓰기 전에 검증한다. 3시간 상한은 매 에폭 **시작 전** 누적 학습 시간으로 판정해(에폭 중간에는 끊지 않고, 마지막 에폭을 끝냈으면 `complete`) `incomplete`로 기록하며, 상한에 걸린 실행을 재개하면 더 학습하지 않고 같은 `incomplete`를 다시 기록한다. `build_model_and_loss`가 스위치를 모델·손실로 옮긴다(T3.5 항목의 대응; B와 Bp 동시 켜짐은 거부). `check_git_clean(repo_root, allow_dirty)`는 `diagnosis.provenance.resolve_code_commit`에 위임한다. `train_and_evaluate`는 커밋 확인 → 학습(bicubic은 건너뜀) → `evaluate_model(model|None, run_meta, data_dir, <output_dir>/<run_id>)`의 얇은 조합이며, 평가 자체는 T3.6이라 테스트에서는 평가기를 대체했다. 한계: 현재 패치 풀에는 학습 곡선 모니터링용 DIV2K 분할이 없어 학습 곡선은 학습 손실뿐이고, 옵티마이저·스케줄은 `adam`·`none`만 지원한다(T4.2가 다른 값을 고정하면 트레이너 확장 필요). 파일럿 경로(`engine/pilot.py`)는 이미 커밋된 결과를 낸 코드라 그대로 두었다. 다른 에이전트 교차 검증(중요 3·사소 2)을 반영했다. Colab T4 1차 확인(commit `909246c`, torch 2.14.1+cu130, Tesla T4): 확인 ③ 상한은 통과했고 ①②는 **A가 켜진 구성 4개(A, AB, AC, ABC)에서만 FAIL**(같은 시드 2회 가중치 최대 차이 1.3e-6~6.8e-5, 재개 후 3e-8~1.2e-4), baseline·B·C·Bp·BC는 비트 단위로 같았다. A의 Sobel 손실 복제 패딩 backward가 CUDA에서 비결정적인 것으로 추정되며(원인 연산은 따로 확정하지 않음), 결정적 알고리즘을 강제하면(진단 셀) 4개 모두 차이 0이고 경고도 없었다. 그래서 `common/seed.py`의 `set_model_seed`가 `torch.use_deterministic_algorithms(True)`(strict)와 `CUBLAS_WORKSPACE_CONFIG`(미지정 시 `:4096:8`)를 설정하도록 고쳤다(로컬 테스트 통과). Colab T4 재확인(commit `cd0c19a`, torch 2.14.1+cu130, CUDA 13.0, cuDNN 9.28, Tesla T4): ① 9개 구성 모두 같은 시드 2회 가중치·에폭 손실이 비트 단위로 같고 다른 시드는 다른 결과, ② 9개 구성 모두 2에폭 저장 뒤 끊고 재개한 결과가 한 번에 돈 결과와 비트 단위로 같음, ③ 상한 시 `incomplete`(1/3에폭)와 재개해도 추가 학습 없음 → 셀 7 전체 PASS. 참고: 결정적 알고리즘 적용 후 파일럿 규모(에폭당 10,000패치, 배치 16, HR 패치 96) 1에폭은 구성별 3.2~5.0초로, 3시간 상한에 약 2,200~3,400에폭이 들어간다(합성 데이터, T4.1·T4.2의 가늠용). 주의: strict 모드라서 평가·진단 코드를 같은 프로세스에서 CUDA로 돌릴 때 결정적 구현이 없는 연산이 있으면 오류가 날 수 있으니 T3.6·T3.8에서 확인한다.
- 평가가 완성되었다(T3.6 ⚠ 통과, 로컬 CPU 테스트 통과, main 병합은 차가원 승인 대기). `metrics.compute_edge_psnr`(HR Y Sobel 크기 상위 ceil(q%·N) 화소, 동률은 안정 정렬, 완전 평탄 영상에서도 정의됨)·`compute_lpips`([0, 1]→[-1, 1], AlexNet, 모델 주입 가능), `data/dataset.TestsetDataset`(`testsets/<데이터셋>/{LR,HR}` 로드, id·크기 대응 검증), `benchmark.measure_inference_time`(`model._network`만 / `model(lr)` 전체, bicubic은 `infer_ms_net` 0.0, 워밍업 후 평균·장치 동기화, 측정 조건·배율은 기본값 없이 `infer_benchmark`·`fixed_rules.scale`에서만 받음), `evaluator.evaluate_model(model|None, run_meta, data_dir, output_dir, *, fair_comparison, device, lpips_model)`이 구현되어 `tests/test_evaluation.py`·`test_evaluator.py`·`test_benchmark.py`·`test_testset_dataset.py`를 통과한다. 평가는 `fair_comparison`(shave_border, datasets, edge_top_q, LPIPS 백본 alex, infer_benchmark, 대표 영상)에서만 값을 읽고, `edge_top_q`가 `null`이거나 0 < q < 100 밖이면 아무것도 만들기 전에 평가를 거부한다. B 계열의 `psnr_y_pre_rgb`는 정규화 Y를 [0, 1]로 자르고 Y 스케일 복원 → 가장자리 자르기 후 HR Y와 비교한다(HR Y도 반올림하지 않으므로 정수 반올림하지 않음). `run_meta`의 스위치 B와 모델의 `sr_y_norm` 출력 여부가 다르면 거부하고, 평가 후 호출자 모델의 학습/평가 모드를 되돌린다. 산출물은 `<실행 폴더>/summary.csv`·`per_image.csv`(SPEC 필드 순서, `switches`는 `A+B` 형식·없으면 `none`)와 대표 영상이 지정돼 있으면 `images/<데이터셋>/<image_id>_{median,worst}.png`(Before(bicubic) | After | HR)·`_roi.png`(HR Y Sobel 평균이 가장 큰 32×32 창의 ×4 확대본)이며, 대표 영상 id가 `null`이면(T4.2 전) 이미지만 건너뛴다. 이미 기록이 있으면 덮어쓰기를 거부하고 임시 폴더에 다 쓴 뒤 옮긴다(옮기다 실패하면 옮긴 것을 지워 반쪽 기록을 남기지 않음). 위 세부 사항 일부는 교차 리뷰 반영으로 고쳤다. 트레이너는 `fair_comparison`·`device`를 넘기도록 연결했고 `FakeEvaluator` 시그니처만 맞췄다. 에지 PSNR은 진단 모듈과 코드를 공유하지 않고(진단은 변경하지 않음) 두 구현이 같은 값을 내는 테스트로 묶었다. 한계: AlexNet LPIPS는 입력 한 변이 약 64화소 이상이어야 해서(실제 테스트셋은 충분) 합성 영상 테스트는 LPIPS 대역을 쓴다. **Colab 실행 필요:** T4에서의 추론 시간 값, CUDA에서 LPIPS·평가가 `use_deterministic_algorithms(True)` 아래 오류 없이 도는지.
- `common/color.py`는 구현되어 테스트를 통과한다(T1.1 통과, 11/15 main 병합 완료).
- `common/bicubic.py`도 구현되어 로컬 CPU 테스트와 Colab 확인(CUDA 일치, BasicSR `imresize` 대조)을 통과한다(T1.2 통과, 11/15 main 병합 완료).
- `common/seed.py`도 구현되어 로컬 CPU 테스트와 Colab T4 확인(CUDA 초기화·난수 재현, 데이터 수열 격리, cuDNN 결정적 모드, 작은 ESPCN형 모델 2회 학습 가중치 일치)을 통과한다(T1.3 통과, 11/15 main 병합 완료). 실제 학습 루프의 GPU 결정성과 재개 후 일치는 T3.7에서 다시 확인한다.
- `data/prepare.py`(패치 풀·테스트셋 LR 생성·압축)와 `scripts/prepare_data.py`는 구현되어 합성 데이터 테스트와 Colab 실제 데이터 확인을 통과한다(T1.4 통과, 11/15 main 병합 완료). Gemini 교차 리뷰(클램핑 검증, 실패 시 반쪽 산출물, 입력 검증, OS별 파일 정렬)를 반영했다. Colab에서 DIV2K 800장과 테스트셋 4종(5·14·100·100장)으로 파일럿용 풀(패치 96, 10000개, 시드 42; `patch_pool_id` `div2k_x4_p96_n10000_s42_e8bc539e`, 압축 파일 387MB, 373초)을 만들고, 압축 해제 후 `content_sha256` 재계산과 테스트셋 HR/LR 대응을 확인했다. 시드 고정 합성 영상의 `patch_pool_id`는 Windows와 Colab(Linux)에서 같다. 패치 크기·개수는 필수 인자이며 값은 `fair_comparison.yaml`에 없다. 본 실행 풀은 T4.2에서 확정한 값으로 다시 만든다. 테스트셋은 HR을 4의 배수로 잘라(modcrop, 104장이 잘림) 저장하므로 평가(T2.4)는 저장된 HR만 써야 한다. 나머지 `src/sr_pipeline/` 모듈은 시그니처와 docstring만 있고, 본체는 `NotImplementedError("TODO")`다.
- 위 압축 파일(파일럿용 풀 `div2k_x4_p96_n10000_s42_e8bc539e`, 387MB)을 공유 Drive와 Kaggle 비공개 Dataset에 올리고 두 경로에서 내려받아 확인했다(T1.5 통과, `notebooks/t1_5_upload_check.ipynb`). Drive: 로그인 없이 공유 링크(`gdown`)로 받은 zip의 sha256이 원본(`77e4a08f77a9c651cfea613204c5bbc8a6031c63bf2df07b20941a3c5e4e87b8`)과 같고, 해제 후 `content_sha256` 재계산·`patch_pool_id`·테스트셋 4종 HR/LR 대응이 모두 통과했다(Colab). Kaggle: Kaggle이 zip을 자동 해제해 원본 zip 해시는 비교하지 못했고, 해제본의 `content_sha256` 재계산과 `patch_pool_id` 일치로 확인했다(Kaggle 노트북, CPU). 저장소가 Public이라 공유 링크와 Dataset 주소는 기록하지 않는다(팀원에게 따로 전달). 본 실행 풀은 T4.2에서 값을 고정한 뒤 같은 방식으로 다시 올려야 한다.
- 스켈레톤 테스트 6개가 `@pytest.mark.skip`이다.
- `configs/fair_comparison.yaml`의 파일럿 결정값은 모두 `null`이다.
- 이미 완료된 것:
  - `configs/experiments.yaml`(10개 구성)
  - `SummaryRecord`·`PerImageRecord` 필드
  - `build_run_id`, `count_parameters`
  - 스크립트 argparse, 노트북 뼈대
  - 프로젝트 Skills 설치

## 2. 담당 표

| 기간 | 고용민 (A) | 차가원 (B) |
| --- | --- | --- |
| 1~2주차 | 파이프라인: 공용 규칙, 데이터 준비, 샘플러, 실행, 평가, 집계 | 모델: 모델, 손실, 진단 (1주차는 합성 데이터로 병렬 개발) |
| 3주차~ | 개선별 담당 (M3에서 확정) | 개선별 담당 (M3에서 확정) |

| 역할 | 담당 |
| --- | --- |
| 공통 기반 담당 (`configs/`, `common/`, `evaluation/`, 의존성, `AGENTS.md`) | 고용민 |
| 병합 담당 (main이 항상 `pytest` 통과 상태 유지) | PR을 승인한 사람이 승인과 함께 병합한다 (11/15 합의). 승인자는 병합 전 검증 명령 통과를 확인한다 |

## 3. 마일스톤

| ID | 기간 | 마일스톤 | 종료 조건 |
| --- | --- | --- | --- |
| M0 | 11/11~12 | 환경·하네스 | T4·Drive·push 확인, `AGENTS.md`·verify 커밋 |
| M1 | ~11/13 | 공용 규칙·데이터 | 압축 파일 1개가 Drive·Kaggle에 올라감 |
| M2 | ~11/14 | **최소 진단 (최우선)** | 파일럿 베이스라인 1시드 + 진단 4종 결과가 `results/pilot/`에 있음 |
| M3 | 11/15~17 | 진단 분석·정식 파이프라인 | ADR-0004 갱신, 전 테스트 통과 |
| M4 | 11/18~19 | 파일럿·조건 고정 | `fair_comparison.yaml`에 `null` 없음 |
| M5 | 11/20 | 제안 자료 제출 | 제출 완료 |
| M6 | 11/20~27 | 핵심 Ablation | 18회 실행 기록이 main에 병합, 결과 표 자동 생성 |
| M7 | 11/28 (~12/4) | 승격 결정·확장 실행 | 결정 기록, 승격 시 9회 추가 |
| M8 | **12/5** | **동결** | 집계 스냅샷 보존 |
| M9 | ~12/9 | 데모 | 3분할 데모 동작 + 녹화 영상 |
| M10 | 12/5~10 | 보고·발표 | 12/10 발표 |

## 4. WBS 트리

```text
x4 SR 개선 실험
├── M0 환경·하네스 ─── T0.1 환경 점검 · T0.2 AGENTS.md·verify · T0.3 Skills ADR
├── M1 공용 규칙·데이터 ─ T1.1 색 변환 · T1.2 bicubic · T1.3 시드 · T1.4 패치 풀 · T1.5 업로드
├── M2 최소 진단 ───── T2.1 ESPCN · T2.2 L1 · T2.3 진단 함수 · T2.4 PSNR/SSIM(Y) · T2.5 파일럿 진단
├── M3 정식 파이프라인 ─ T3.1 개선 확정 · T3.2 구성 해석 · T3.3 샘플러 · T3.4 모델 스위치
│                       T3.5 손실 스위치 · T3.6 평가 완성 · T3.7 학습(실행) · T3.8 실행 창구 · T3.9 e2e
├── M4 파일럿·고정 ──── T4.1 에폭 파일럿 · T4.2 조건 고정 · T4.3 집계 핵심
├── M5 제안 자료 ───── T5.1
├── M6 핵심 실행 ───── T6.1 Bicubic 하한 · T6.2 18회 실행 · T6.3 집계 그림
├── M7 승격 ───────── T7.1 결정 · T7.2 확장 실행 또는 C 분석
├── M8 동결 ───────── T8.1 스냅샷
├── M9 데모 ───────── T9.1 3분할 데모
└── M10 보고·발표 ──── T10.1 보고서 · T10.2 발표 자료·리허설
```

## 5. Task 목록

### M0 환경·하네스 (11/11~12)

| ID | 작업 | 담당 | 선행 | US | 완료 조건 | 상태 |
| --- | --- | --- | --- | --- | --- | --- |
| T0.1 | Colab T4 연결, Drive 마운트, 원격 git push 점검. 노트북 01에 push 테스트 셀 추가 | 공통 | — | 1–3 | 두 팀원 모두 T4·마운트·push 성공(또는 대체 경로 확정) | 통과 |
| T0.2 | `AGENTS.md` 작성(용어집·ADR 준수, ⚠ 규칙, 병합 후 즉시 반영, 마감 직전 우회 병합 절차), `@AGENTS.md` 한 줄 `CLAUDE.md`, verify = `pytest` + `python scripts/sync_skills.py --check` | 공통 기반 | — | 69–74 | 세 에이전트에서 규칙 인식 확인, verify 로컬 CPU 수 분 내 통과 | 통과 |
| T0.3 | Skills 설치 결정 ADR 작성, `.agents/`·`.claude/`·`scripts/sync_skills.py` 커밋 | 공통 기반 | — | — | ADR 커밋, `/스킬명` 호출 확인 | 통과 |

### M1 공용 규칙·데이터 (~11/13)

| ID | 작업 | 담당 | 선행 | US | 완료 조건 | 상태 |
| --- | --- | --- | --- | --- | --- | --- |
| T1.1 ⚠ | `common/color.py`: BT.601(MATLAB `rgb2ycbcr` 계수) RGB↔YCbCr, Y 16~235 정규화 | 고용민 | — | 10 | `test_color_and_bicubic.py` 색 변환 항목 통과 | 통과 |
| T1.2 ⚠ | `common/bicubic.py`: 화소 중심 정렬, 축소 시 안티에일리어싱, GPU 텐서 연산 | 고용민 | — | 11–12 | BasicSR MATLAB 호환 `imresize`와 허용 오차 내 일치 테스트 통과 | 통과 |
| T1.3 | `common/seed.py`: 시드 하나 → 모델 초기화 난수 + 데이터 난수 | 고용민 | — | 32 | 같은 시드 재현 테스트 통과 | 통과 |
| T1.4 ⚠ | `data/prepare.py`: DIV2K HR 800장 → LR, 패치 풀 1회 생성 + 생성 조건·식별자 기록. 테스트셋 4종 LR 생성. 압축 파일 1개 | 고용민 | T1.2 | 4, 8, 9 | 압축 파일과 메타데이터(`patch_pool_id`) 생성 | 통과 |
| T1.5 | 압축 파일을 공유 Drive와 Kaggle 비공개 Dataset에 업로드, 공유 링크 확보 | 고용민 | T1.4 | 5–6 | 두 경로에서 내려받아 해제 확인 | 통과 |

### M2 최소 진단 (~11/14, 최우선)

| ID | 작업 | 담당 | 선행 | US | 완료 조건 | 상태 |
| --- | --- | --- | --- | --- | --- | --- |
| T2.1 | `models/espcn.py` 베이스라인: RGB 입력, sub-pixel, 은닉층 tanh, 마지막 층 활성화 없음 | 차가원 | — | 21, 75 | 합성 데이터로 순전파·출력 크기 테스트 통과 | 통과 |
| T2.2 ⚠ | `losses/sr_loss.py` L1 (자르지 않은 출력 사용) | 차가원 | — | 22(일부), 30 | 손실 테스트 통과 | 통과 |
| T2.3 | `diagnosis/diagnostics.py`: 오차 지도, 에지·평탄 PSNR(Y), 주파수 스펙트럼, YCbCr 채널별 오차, 최악 영상 목록 | 차가원 | T1.1 | 52–56 | 합성 SR/HR 쌍으로 5종 산출물 생성 | 통과 |
| T2.4 ⚠ | `evaluation/postprocess.py`(자르기 → 8비트 → 가장자리 4화소 → Y) + `metrics.py` PSNR(Y)·SSIM(Y) | 고용민 | T1.1 | 41 | 같은 영상 PSNR=∞·SSIM=1, 가장자리 무영향 테스트 통과 | 통과 |
| T2.5 | `scripts/run_pilot_diagnosis.py` 최소 기능: 파일럿 베이스라인 1시드 학습(`scripts/train_pilot_baseline.py`로 분리) + 진단 4종 실행 (Colab) | 고용민·차가원 | T1.4, T2.1–T2.4 | 51 | `results/pilot/`에 진단 그림·표 커밋 | 통과 |

### M3 진단 분석·정식 파이프라인 (11/15~17)

| ID | 작업 | 담당 | 선행 | US | 완료 조건 | 상태 |
| --- | --- | --- | --- | --- | --- | --- |
| T3.1 | 진단 분석 → 개선 A·B·C 확정, ADR-0004 갱신, 필요 시 SPEC 스토리·실행 목록 수정, 3주차 개선별 담당 확정 | 공통 | T2.5 | 57, 83 | ADR-0004 상태 "확정", 담당 표 갱신 | 대기 |
| T3.2 ⚠ | `config/schema.py`: 공정 비교 조건 로드, 구성 키는 `trainable`·`promoted_only`·스위치만 허용, 시드는 실행 인자로 받고 실행 조건 확정 | 고용민 | — | 13–15, 28–29 | 덮어쓰기 거부, SPEC의 확정 실행 조건 비교 테스트 통과 | 통과 |
| T3.3 ⚠ | `data/sampler.py`·`dataset.py`: 데이터 난수로 에폭별 패치·순서·증강 결정 | 고용민 | T1.3 | 32 | 같은 시드 → 구성 무관 같은 순서, 다른 시드 → 다른 순서 테스트 통과 | 통과 |
| T3.4 | 모델 스위치 B(1채널 Y + Cb·Cr bicubic 경로), C(bicubic 확대 + 예측 차이) | 차가원 | T2.1, T1.2 | 24, 26, 28–29 | B: 출력 RGB 3채널·파라미터 감소. C: 차이 0이면 bicubic과 동일 | 통과 |
| T3.5 ⚠ | 손실 스위치 A(Sobel 수평·수직 L1 × λ), B(정규화 Y 채널), B′(RGB 출력 → 정규화 Y 손실) | 차가원 | T2.2, T1.1 | 22, 25, 27 | λ=0이면 베이스라인과 동일, 평탄 영상에서 손실·기울기 유한, B′ 파라미터·입출력이 베이스라인과 같음 | 통과 |
| T3.6 ⚠ | 평가 완성: PSNR(RGB), 에지 PSNR(상위 q%), LPIPS(AlexNet, [-1,1]), B 계열 `psnr_y_pre_rgb`, 추론 시간(네트워크만/전체, 320×180→1280×720, 워밍업 10 + 100회), 대표 영상 Before/After·확대본, `evaluator.py` | 고용민 | T2.4 | 42–50 | 평가 경계 테스트 전부 통과, summary·per_image 행 생성 | 통과 |
| T3.7 | `engine/trainer.py`: 에폭 기반 학습, 중간·끝 체크포인트와 재개, 3시간 상한 → `incomplete`, 실행 ID(commit 포함), 커밋 안 된 변경 경고·거부, `train_minutes`·학습 곡선 기록 | 고용민 | T3.2, T3.3 | 16, 19–20, 31–37 | 재개 후 결과 동일, 상한 시 `incomplete` 기록 | 통과 |
| T3.8 | `scripts/run_experiment.py` + 노트북 02 완성: 경로 인자, Drive 복사·해제(대체: 공유 링크), `getpass` 토큰(저장 안 함), 실행 기록 커밋·push, Kaggle 경로 | 고용민 | T3.7 | 2–3, 7, 38–40 | Colab과 Kaggle에서 같은 명령으로 1에폭 실행 | 대기 |
| T3.9 | 전체 테스트 `skip` 해제, e2e: 모든 구성이 작은 데이터·1에폭으로 끝까지 실행, 형식 맞는 행 생성, 재현성 | 공통 | T3.2–T3.8 | 31, 33 | `pytest` 전부 통과(skip 0) | 대기 |

### M4 파일럿·조건 고정 (11/18~19)

| ID | 작업 | 담당 | 선행 | US | 완료 조건 | 상태 |
| --- | --- | --- | --- | --- | --- | --- |
| T4.1 | 가장 무거운 구성으로 에폭 파일럿, 에폭당 시간 측정 → 30분 안팎이 되도록 에폭 수 역산 | 공통 | T3.9 | 17 | 측정값과 역산 근거 기록 | 대기 |
| T4.2 ⚠ | `fair_comparison.yaml`의 `null` 일괄 고정: 에폭, 에폭당 패치, 패치 풀 크기, 학습률, 배치, 패치 크기, 옵티마이저, 스케줄, 증강, λ, q, 대표 영상 목록(파일럿 베이스라인 중앙값·최하위 규칙) | 공통 | T4.1 | 18, 23, 50 | 상대 승인 PR 병합, `null` 0개 | 대기 |
| T4.3 | `reporting/aggregate.py` 핵심: main 실행 기록 로드, 중복 (config_id, seed, patch_pool_id) 오류, 평균±표준편차, Δ, 시드별 Δ 부호 일관성, 평균 Δ vs 시드별 Δ 표준편차, 미완료 표시 (고정된 가짜 기록으로 병렬 개발) | 고용민 | — | 58–59, 62, 67 | 집계 경계 테스트 통과 | 대기 |

### M5 제안 자료 (11/20)

| ID | 작업 | 담당 | 선행 | US | 완료 조건 | 상태 |
| --- | --- | --- | --- | --- | --- | --- |
| T5.1 | 제안 자료: 진단 근거 → 개선 설계(강의 개념 연결), 실험 계획. 진단은 파일럿 베이스라인 기준이라고 명시 | 공통 | T3.1 | 83 | 제출 | 대기 |

### M6 핵심 실행 (11/20~27)

| ID | 작업 | 담당 | 선행 | US | 완료 조건 | 상태 |
| --- | --- | --- | --- | --- | --- | --- |
| T6.1 | Bicubic 하한을 같은 평가 경로로 측정 | 고용민 | T4.2 | 49 | `config_id=bicubic` 실행 기록 병합 | 대기 |
| T6.2 | 베이스라인, +A, +B, +A+B, +C, B′ × 시드 {0,1,2} = 18회 실행. 이름 브랜치 커밋 → PR → main 병합. 상대 변경 병합 시 즉시 반영 후 다음 실행 | 개선별 | T4.2 | 27, 31, 73 | 18회 모두 `complete`로 main에 병합 | 대기 |
| T6.3 | 집계 산출물: 결과 표, Ablation 표(A: RGB·Y 맥락, B: 정렬·입력/용량 분해), BSD100·Urban100 영상별 Δ 분포, 성공·실패 사례 자동 선정, 다중 구성 비교 그림·오차 지도, 학습 시간 표·학습 곡선, 구성×시드 진행 현황 표 | 고용민 | T4.3 | 60–61, 63–66, 76, 84 | 표·그림 파일이 `results/aggregated/`에 자동 생성 | 대기 |

### M7 승격 결정 (11/28, 확장 실행 ~12/4)

| ID | 작업 | 담당 | 선행 | US | 완료 조건 | 상태 |
| --- | --- | --- | --- | --- | --- | --- |
| T7.1 | 핵심 Ablation 완료 여부와 가장 느린 구성의 실행 시간으로 C 승격 결정, ADR-0004에 기록 | 공통 | T6.2 | 77 | 결정과 근거 기록 | 대기 |
| T7.2 | 승격 시 +A+C, +B+C, +A+B+C × 3 = 9회 실행. 미승격 시 +C 단독 결과 원인 분석 | 개선별 | T7.1 | 78 | 실행 기록 병합 또는 원인 분석 문서 | 대기 |

### M8 동결 (12/5)

| ID | 작업 | 담당 | 선행 | US | 완료 조건 | 상태 |
| --- | --- | --- | --- | --- | --- | --- |
| T8.1 | 동결 시점 집계 결과 스냅샷 보존(태그 + 산출물). 모든 수치가 실행 ID로 추적되는지 확인 | 병합 담당 | T6.3, T7.2 | 68, 82 | 태그 생성, 보고서·발표·데모가 같은 스냅샷 사용 | 대기 |

### M9 데모 (~12/9)

| ID | 작업 | 담당 | 선행 | US | 완료 조건 | 상태 |
| --- | --- | --- | --- | --- | --- | --- |
| T9.1 | `demo/app.py` + 노트북 03: 체크포인트로 Bicubic 하한 / 베이스라인 / 제안 구성 3분할 + 확대 영역, CPU 대체, 업로드 사진(선택), 녹화 영상 백업 | 차가원 | T8.1 | 79–81 | T4·CPU 양쪽에서 동작, 녹화 영상 확보 | 대기 |

### M10 보고·발표 (12/5~10)

| ID | 작업 | 담당 | 선행 | US | 완료 조건 | 상태 |
| --- | --- | --- | --- | --- | --- | --- |
| T10.1 | 보고서: 진단 → 강의 개념 → 개선 근거, 결과·Ablation, 성공·실패 사례 Before/After·오차 지도, SPEC "해석상 주의점" 5개 반영 | 공통 | T8.1 | 83–84 | 모든 수치가 동결 스냅샷에서 나옴 | 대기 |
| T10.2 | 발표 자료(15분, `/pptx`)와 리허설 2회, 12/10 발표 | 공통 | T9.1, T10.1 | 79 | 리허설 완료 | 대기 |
