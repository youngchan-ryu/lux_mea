(.venv) youngchanryu@Youngchans-MacBook-Pro sw % python3 bench_asr.py record --parts parts_poster.yaml
[i] parts_poster.yaml → clips_poster/
부위 5개 × 2회. 실제 발표처럼 문장으로 말하세요.

  [title] (럭스, lux, 룩스) 1/2 — Enter 후 말하세요
    저장 (peak=0.51)
  [title] (럭스, lux, 룩스) 2/2 — Enter 후 말하세요
    저장 (peak=0.52)
  [heading] (개요, 제품의 개요) 1/2 — Enter 후 말하세요
    저장 (peak=0.44)
  [heading] (개요, 제품의 개요) 2/2 — Enter 후 말하세요
    저장 (peak=0.50)
  [program] (프로그램, 내장, 짧고 흔한) 1/2 — Enter 후 말하세요
    저장 (peak=0.48)
  [program] (프로그램, 내장, 짧고 흔한) 2/2 — Enter 후 말하세요
    저장 (peak=0.54)
  [sustain] (안정성, 작동, 정해진 규칙) 1/2 — Enter 후 말하세요
    저장 (peak=0.50)
  [sustain] (안정성, 작동, 정해진 규칙) 2/2 — Enter 후 말하세요
    저장 (peak=0.36)
  [bridge] (브릿지, 브리지, 다리) 1/2 — Enter 후 말하세요
    저장 (peak=0.42)
  [bridge] (브릿지, 브리지, 다리) 2/2 — Enter 후 말하세요
    저장 (peak=0.51)

[i] 10개 클립 저장 → python3 bench_asr.py run --parts parts_poster.yaml
(.venv) youngchanryu@Youngchans-MacBook-Pro sw % python3 bench_asr.py run    --parts parts_poster.yaml
[i] ASR 엔진=mlx 모델=mlx-community/whisper-large-v3-turbo
Fetching 4 files: 100%|███████████| 4/4 [00:00<00:00, 2348.76it/s]
Reconstruction complete: |           |  0.00B /  0.00B            
Download complete: :                          |  0.00B            

mlx-community/whisper-large-v3-turbo
  적중 7/10 (70%)  중앙 지연 0.49s
    ✗ title → None   전사='오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만 오랜만'
    ✗ heading → None   전사='우리 재유의 우리 재통'
    ✗ bridge → None   전사=''
[i] ASR 엔진=mlx 모델=mlx-community/whisper-medium-mlx
Fetching 4 files: 100%|███████████| 4/4 [00:00<00:00, 2218.33it/s]
Reconstruction complete: |           |  0.00B /  0.00B            
Download complete: :                          |  0.00B            

mlx-community/whisper-medium-mlx
  적중 8/10 (80%)  중앙 지연 0.37s
    ✗ title → None   전사='이 제품을 러츠웨아 입니다.'
    ✗ heading → None   전사='우리 is is'
[i] ASR 엔진=mlx 모델=mlx-community/whisper-small-mlx
Fetching 4 files: 100%|███████████| 4/4 [00:00<00:00, 2793.41it/s]
Download complete: :                          |  0.00B            
Reconstruction complete: |           |  0.00B /  0.00B            

mlx-community/whisper-small-mlx
  적중 6/10 (60%)  중앙 지연 0.13s
    ✗ title → None   전사='감사합니다 .'
    ✗ title → None   전사='제품의 이름은 록팀웨어 입니다.'
    ✗ heading → None   전사='우리 제이홉의 우리 제이홉'
    ✗ bridge → None   전사='그래서 우리는 이걸 그리지라고 부르기'

==============================================================
모델                                             적중률      지연
mlx-community/whisper-large-v3-turbo           70%   0.49s
mlx-community/whisper-medium-mlx               80%   0.37s
mlx-community/whisper-small-mlx                60%   0.13s
==============================================================
적중률이 같다면 더 작은 모델을 쓴다 (지연·메모리 이득).
오답의 전사 결과를 보고 parts.yaml 별칭을 보강하는 것이
모델을 키우는 것보다 대개 효과가 크다.
(.venv) youngchanryu@Youngchans-MacBook-Pro sw % python3 bench_asr.py record --parts parts_poster.yaml
[i] parts_poster.yaml → clips_poster/
부위 5개 × 2회. 실제 발표처럼 문장으로 말하세요.

  [title] (럭스, lux, 룩스) 1/2 — Enter 후 말하세요
    저장 (peak=0.44)
  [title] (럭스, lux, 룩스) 2/2 — Enter 후 말하세요
    저장 (peak=0.28)
  [heading] (개요, 제품의 개요) 1/2 — Enter 후 말하세요
    저장 (peak=0.34)
  [heading] (개요, 제품의 개요) 2/2 — Enter 후 말하세요
    저장 (peak=0.35)
  [program] (프로그램, 내장, 짧고 흔한) 1/2 — Enter 후 말하세요
    저장 (peak=0.41)
  [program] (프로그램, 내장, 짧고 흔한) 2/2 — Enter 후 말하세요
    저장 (peak=0.66)
  [sustain] (안정성, 작동, 정해진 규칙) 1/2 — Enter 후 말하세요
    저장 (peak=0.29)
  [sustain] (안정성, 작동, 정해진 규칙) 2/2 — Enter 후 말하세요
    저장 (peak=0.35)
  [bridge] (브릿지, 브리지, 다리) 1/2 — Enter 후 말하세요
    저장 (peak=0.52)
  [bridge] (브릿지, 브리지, 다리) 2/2 — Enter 후 말하세요
    저장 (peak=0.42)

[i] 10개 클립 저장 → python3 bench_asr.py run --parts parts_poster.yaml
(.venv) youngchanryu@Youngchans-MacBook-Pro sw % python3 bench_asr.py run    --parts parts_poster.yaml
[i] ASR 엔진=mlx 모델=mlx-community/whisper-large-v3-turbo
Fetching 4 files: 100%|███████████| 4/4 [00:00<00:00, 1352.78it/s]
Reconstruction complete: |           |  0.00B /  0.00B            
Download complete: :                          |  0.00B            

mlx-community/whisper-large-v3-turbo
  적중 4/10 (40%)  중앙 지연 0.48s
    ✗ heading → None   전사='기효로 말씀드리겠습니다.'
    ✗ heading → None   전사='먼저 제품의 기후를 말씀드리겠습니다.'
    ✗ program → None   전사=''
    ✗ program → None   전사=''
    ✗ bridge → None   전사=''
[i] ASR 엔진=mlx 모델=mlx-community/whisper-medium-mlx
Fetching 4 files: 100%|███████████| 4/4 [00:00<00:00, 1070.32it/s]
Reconstruction complete: |           |  0.00B /  0.00B            
Download complete: :                          |  0.00B            

mlx-community/whisper-medium-mlx
  적중 8/10 (80%)  중앙 지연 0.39s
    ✗ heading → None   전사='기회를 말씀드리겠습니다.'
    ✗ heading → None   전사='먼저 제품의 기후를 말씀드리겠습니다.'
[i] ASR 엔진=mlx 모델=mlx-community/whisper-small-mlx
Fetching 4 files: 100%|███████████| 4/4 [00:00<00:00, 1383.69it/s]
Download complete: :                          |  0.00B            
Reconstruction complete: |           |  0.00B /  0.00B            

mlx-community/whisper-small-mlx
  적중 7/10 (70%)  중앙 지연 0.14s
    ✗ heading → None   전사='기회를 말씀드리겠습니다.'
    ✗ heading → None   전사='먼저 제품의 기후를 말씀드리겠습니다.'
    ✗ program → None   전사='ㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱ'

==============================================================
모델                                             적중률      지연
mlx-community/whisper-large-v3-turbo           40%   0.48s
mlx-community/whisper-medium-mlx               80%   0.39s
mlx-community/whisper-small-mlx                70%   0.14s
==============================================================
적중률이 같다면 더 작은 모델을 쓴다 (지연·메모리 이득).
오답의 전사 결과를 보고 parts.yaml 별칭을 보강하는 것이
모델을 키우는 것보다 대개 효과가 크다.