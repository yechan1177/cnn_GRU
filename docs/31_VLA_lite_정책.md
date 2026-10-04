# 31. VLA-lite 소형 정책(하위 정책 검증용)

- 코드: `src/vcp/vla/policy.py`
- 테스트: `tests/test_vla_policy.py`, `tests/test_vla_policy_v3.py`(v3 특징 토큰)
- 계약: `docs/28a_VLA_모듈_인터페이스_계약.md`의 "A4 소유" 절, "v3 추가 계약" 절(10절 참고)
- 상위 계획: `docs/28_VLA_논문_재설계_계획.md` 3절 "하위 정책(VLA-lite)"

## 1. 역할
큐레이션 방법(ours, random, oracle 등)이 고른 데이터로 **같은 구조·같은 학습 예산**의 소형 VLA 정책을 학습하고,
개루프 행동 오차와 폐루프 주행 지표로 데이터 선별의 효과를 비교한다. 정책 자체의 최고 성능이 목표가 아니라,
"데이터 선별 외의 변수를 고정한 측정 도구"가 목표다.

## 2. 구조

```text
 image uint8 [B,64,64,6]  (t RGB | t−2 RGB)          tokens int64 [B,L]           proprio float32 [B,1]
          │ /255 → (x−0.5)/0.25                              │                         (= ego_v / speed_scale)
          ▼                                                 ▼                                  │
 ┌──────────────────────────────┐            ┌───────────────────────────┐                     │
 │ Stem Conv 4×4 s4, 32ch       │            │ Embedding(VOCAB, 64, PAD=0)│                     │
 │  GN → ReLU        [32,16,16] │            │ 마스크 평균 → Linear → ReLU │                     │
 ├──────────────────────────────┤            └─────────────┬─────────────┘                     │
 │ Conv 3×3 s2, 64ch            │   FiLM γ,β ◀─────────────┤ lang [B,64]                      │
 │  GN → x(1+γ)+β → ReLU [64,8,8]│◀──────────┘             │                                   │
 ├──────────────────────────────┤   FiLM γ,β ◀─────────────┤                                   │
 │ Conv 3×3 s2, 128ch           │◀──────────┘             │                                   │
 │  GN → x(1+γ)+β → ReLU[128,4,4]│                         │                                   ▼
 └──────────────┬───────────────┘                         │                         MLP 1→32→32 (ReLU)
                │ flatten(2048) → Linear → ReLU            │                                   │
                ▼ vis [B,128]                              ▼ lang [B,64]                       ▼ prop [B,32]
                └───────────────────────── concat [B,224] ─┴───────────────────────────────────┘
                                                │
                                    MLP 224→256→256→chunk(8)
                                                ▼
                         정규화 가속도 청크 [B,8]  (× accel_scale → m/s²)
```

### 설계 결정
| 항목 | 선택 | 이유 |
|---|---|---|
| 시간 정보 | t와 t−2 프레임을 채널로 쌓음(6ch) | 접근 속도(객체 크기 변화)를 단일 순전파로 본다. 순환 구조 없이 배치 추론 가능 |
| 시각 줄기 | 4×4 stride 4 패치 conv(32ch) | 처음 검토한 3×3 s2 4단(32-64-96-128)은 CPU 1스레드에서 배치 128 순전파+역전파가 약 350 ms(측정)로 3000단계 학습이 20분을 넘었다. 패치 줄기는 모든 픽셀을 보면서(겹치지 않는 선형 투영) 앞단 연산을 크게 줄인다 |
| 정규화 | GroupNorm(8그룹) | 배치 크기·학습/평가 모드와 무관하게 같은 동작(폐루프 배치 크기가 학습과 다르다) |
| 풀링 | 4×4 격자 flatten + Linear | 전역 평균은 위치를, 공간 softmax는 크기·존재 강도를 지운다. 선행 객체의 "자기 차선 여부(위치)"와 "거리(크기)"를 모두 보존한다 |
| 언어 인코더 | 임베딩 마스크 평균(bag-of-words) | 지시문은 스타일·목표 속도 단어가 핵심인 짧은 템플릿이므로 어순 의존이 작다. GRU보다 싸다 |
| 언어 조건화 | FiLM(2·3번째 conv 블록, GN 뒤 ReLU 앞) + 헤드 concat | RT-1의 FiLM-EfficientNet과 같은 발상. FiLM 생성기를 0으로 초기화해 시작 시 항등 |
| 메모리 배치 | channels_last | CPU oneDNN conv가 더 빠르다(측정: 같은 구조 92 ms → 68 ms) |
| 절제 | `use_language=False` | 언어 모듈을 만들지 않고 FiLM 항등, 헤드의 언어 입력은 0 벡터. 토큰이 출력에 영향을 주지 않음을 테스트로 확인 |

주의: GroupNorm은 입력 전체 밝기의 크기를 정규화로 지운다. 전역 밝기만 다른 균일 영상 문제는 이 구조로 풀 수 없고
(테스트에서 확인), 대신 조명 변화에 둔감해진다. 실제 과제는 객체의 위치·크기에 의존하므로 문제가 되지 않는다고
판단했다. 합성 학습 테스트는 "밝은 사각형의 세로 위치 → 가속도"로 바꿨다.

## 3. 입력·출력 정규화
| 항목 | 정규화 | 출처 |
|---|---|---|
| 영상 | uint8 → [0,1] → (x−0.5)/0.25 | 모델 내부 |
| proprio | ego_v / speed_scale(domain) (주행 30, 로봇 3) | `vcp.vla.obs.proprio` |
| 행동 타깃 | action / accel_scale(domain) (주행 4, 로봇 1) | `vcp.vla.obs.accel_scale` |
| 정책 출력 | 청크 첫 원소 × accel_scale → m/s² | `make_policy_fn` |
| 토큰 | 0=PAD(마스크 제외), 어휘 밖 id는 UNK(1) | `vcp.vla.instructions.VOCAB` |

목표 속도는 proprio에 넣지 않는다. 지시문(언어)으로만 전달한다(계약).

### 그룹 경계 규칙(`PolicyData`)
- 그룹(`group` 배열)의 **연속 구간**을 하나의 에피소드·블록으로 본다.
- 관측: stack_frames(t, t−2). t−2가 구간 시작보다 앞이면 구간 첫 프레임을 쓴다(`prev_index`).
- 행동 청크: action[t : t+chunk], 구간 끝을 넘는 자리는 구간 마지막 값으로 채운다(`chunk_index`). 다른 에피소드의 행동이 섞이지 않는다.

## 4. 손실
청크 위치 가중 Huber(SmoothL1, β=1.0, 정규화 단위):

L = mean_{b,k} w_k · SmoothL1(ŷ_{b,k}, y_{b,k}),  w_k ∝ 1 + 0.5·(1 − k/(H−1)),  mean(w) = 1

- 폐루프에서는 청크 첫 원소만 실행하므로 앞쪽을 조금 더 중시한다(첫 원소 가중치가 마지막의 1.5배).
- 뒤쪽 원소는 미래 의도를 학습시키는 보조 신호로 남긴다.
- Huber는 급제동(정규화 −2 부근) 같은 드문 큰 값에 대한 기울기 폭주를 막으면서 작은 오차에서는 MSE처럼 동작한다.

## 5. 학습 통제 원칙
- **고정 경사 단계**: 모든 조건(큐레이션 방법·예산·시드)에서 `steps`(기본 3000)·`batch`(128)·`lr`(1e-3)·구조를 고정한다.
  미니배치는 `train_idx`에서 복원 추출한다. 따라서 데이터 양(예산 5·10·20%·full)만 바뀌고 계산량은 같다.
  에폭 기반 학습이면 큰 데이터가 더 많은 갱신을 받아 "데이터 품질"과 "계산량" 효과가 섞인다.
- 최적화: AdamW(weight_decay 1e-4), 선형 워밍업(min(100, steps/20) 단계) 후 cosine 감쇠(→0), 기울기 노름 1.0 자르기.
- 결정성: `torch.manual_seed(seed)`, `np.random.seed(seed)`, 배치 추출 `default_rng(seed)`, 평행이동 증강
  `default_rng(seed+1)`, 밝기·대비 증강 `torch.Generator(seed+2)`. 같은 시드 → 같은 가중치(테스트). 스레드 수는
  `cfg.threads`로 고정하고 학습 후 원래 값으로 되돌린다.
- 증강(`augment=True`): 밝기 ±0.06, 대비 ×[0.85, 1.15](두 시점 프레임에 같은 값), 평행이동 ±2 px(가장자리 복제 패딩,
  두 시점 함께). **좌우 반전은 하지 않는다**(차선·주행 방향의 비대칭성).
- 반환 정보: 손실 곡선(구간 평균 30개)과 구간 끝 단계, 마지막 5% 평균 손실, 학습 시간, samples/s, 파라미터 수, 어휘 크기.

## 6. 데이터 소스
| 클래스 | 용도 | 비고 |
|---|---|---|
| `PoolImageSource` | 시뮬레이터 풀 | `box_ptr`/`boxes`/`horizon_y`로 `vcp.sim.render.render_frame`을 지연 호출. LRU 캐시 상한 `max_cache`(기본 100,000 프레임 ≈ 1.2 GB) |
| `ArrayImageSource` | comma 64×64 프레임 | 미리 만든 `frames` 배열 |

`PoolImageSource.from_pool(arrays, domain)`으로 `load_pool` 결과에서 바로 만든다. 렌더러 실측(A2 렌더러, 이 세션의 공유 CPU):
cold 렌더링 약 0.20 ms/프레임. 학습 한 단계는 프레임 256장(t, t−2)을 요구하므로 캐시가 없으면 단계당 약 50 ms가 추가된다.
학습 프레임이 상한보다 많으면 캐시 적중률이 떨어지므로, 대형 풀에서는 `max_cache`를 학습 인덱스 수 이상으로 잡는 것을 권장한다.

## 7. 파라미터 수와 처리량(측정)
측정 환경: Intel Xeon 2.8 GHz 4코어(가상), torch 2.14 CPU, 배치 128, 64×64 입력. **다른 작업자 프로세스와 CPU를 공유하던
상태**(load average 1.5~2.3)라 단독 실행보다 낮을 수 있다. 처리량은 200단계 학습의 전체 시간(데이터 조회·증강·순전파·역전파·
옵티마이저 포함)으로 계산했다. 영상은 무작위 uint8 배열(`ArrayImageSource`, 실데이터 64×64와 같은 계산량)이다.

| 모델 | 파라미터 수 |
|---|---|
| use_language=True(어휘 64) | 517,736 |
| use_language=True(어휘 89, A2 실제 VOCAB) | 519,336 |
| use_language=False(절제) | 484,520 |

| 스레드 | 증강 | samples/s | 3000×128 예상 시간 |
|---|---|---|---|
| 1 | 켬 | 1,198 | 5.3분 |
| 1 | 끔 | 1,341 | 4.8분 |
| 4 | 켬 | 1,692 | 3.8분 |
| 4 | 끔 | 2,322 | 2.8분 |

실제 3000단계 1회 측정(1스레드, 증강 켬, A2 소형 풀 12 에피소드·3,600프레임을 `PoolImageSource`로 지연 렌더링, 첫 접근 시 렌더링 비용 포함):
**336초(5.6분), 1,141 samples/s**. 이 소형 풀에서 손실 곡선(구간 평균)은 0.080 → 0.0028로 감소했다(풀이 작아 일반화 성능을 뜻하지 않는다).
따라서 1스레드 3000×128 학습은 "수 분 이내" 목표를 만족하지만 여유는 크지 않다.

참고(구조 결정 근거, 같은 환경): 3×3 s2 4단(32-64-96-128) 구조는 1스레드 증강 포함 약 300 samples/s(3000×128에 약 21분)였다.
4스레드 효율이 낮은 것은 작은 텐서의 연산별 병렬화 이득이 작고 다른 프로세스와 코어를 나눠 쓰기 때문으로 보인다.
여러 조건을 돌릴 때는 1스레드 학습 여러 개를 병렬로 돌리는 편이 총 처리량이 높을 것으로 예상한다(미측정).

## 8. 인터페이스 요약
```python
cfg = PolicyConfig(chunk=8, use_language=True, steps=3000, batch=128, seed=0, threads=1)
data = PolicyData(images=PoolImageSource.from_pool(arr, "driving"), group=arr["ep"], ego_v=arr["ego_v"],
                  action=arr["expert_cmd"], tokens=tokens, domain="driving")
model, info = train_policy(data, train_idx, cfg)
pred = predict_open_loop(model, data, test_idx)          # [len, 8] m/s²
fn = make_policy_fn(model, "driving")                    # closed_loop.run_closed_loop(fn, specs)
```

계약 대비 추가 사항(시그니처 변경 없음)
- `PolicyConfig` 확장 필드(모두 기본값): `huber_beta`, `front_weight`, `warmup_steps`, `grad_clip`, `n_curve_bins`, `device`.
- `PolicyData` 보조 메서드: `prev_index`, `chunk_index`, `action_chunk`, `observation`.
- `PoolImageSource.from_pool`, `cache_info`, 모듈 함수 `chunk_weights`, `shift_images`, `photometric_jitter`, `augment_observation`, `count_parameters`.
- `predict_open_loop(..., batch_size=512)` 선택 인자.
- `vcp.vla.obs`(A2)가 없을 때만 계약 값과 같은 대체 관측 규격을 쓰고 경고한다. 두 구현의 일치는 테스트로 확인한다.

## 9. 검증
- `tests/test_vla_policy.py`(13개): 형태, 언어 절제 불변성, 청크 패딩 그룹 경계, t−2 그룹 경계, 합성 문제 학습
  (위치 → 가속도, 200단계에서 MAE가 기준선의 20% 미만), make_policy_fn 형태·개루프 일치, 결정성, 스레드 복원,
  증강 범위, LRU 상한, A2 obs 일치, 실제 렌더러 연결.
- A2 모듈 연결 확인(스크래치 스크립트): 12 에피소드 소형 풀 생성 → `PoolImageSource` + `encode_instruction` 토큰 →
  150단계 학습 → `predict_open_loop` → `run_closed_loop(make_policy_fn(...))` 4 에피소드가 오류 없이 동작했다.
  이 결과는 연결 확인용이며 성능 수치로 쓰지 않는다.

## 10. v3: 검출 특징 토큰(P1, docs/33)
v3 계획([33](33_v3_핵심지표_벤치마크_개선계획.md))의 P1이다. 64×64 영상만으로는 근거리 간격·정지 판단이 어렵다는 진단(AMR 실패,
실영상 AUROC)에 따라, 검출기에서 나온 프레임 특징을 정책 입력에 더한다. 기본값(`use_features=False`)은 v2 모델과 같다.

### 10.1 구조
```text
 features float32 [B,2,32]   (0번 = 프레임 t, 1번 = t−2; 각 행 = v1 16 + v2 16, X_v1v2 순서)
          │ NaN → 0, clamp(−3, 3)      (특징은 대부분 [−1,1] 근처라 별도 정규화 없이 이상치만 자른다)
          ▼ flatten [B,64]
 MLP 64 → 64 → 64 (ReLU 2회)  →  feat [B,64]
          │
 concat(vis 128, lang 64, prop 32, feat 64) = [B,288] → MLP 288→256→256→chunk
```
- 결합 위치: FiLM 이후 헤드 입력(영상·언어·속도와 나란히). 특징으로 FiLM을 조건화하지는 않는다.
- 초기화 순서: 특징 MLP는 다른 모든 모듈을 만든 뒤 마지막에 만든다. 그래서 같은 torch 시드에서 CNN·언어·proprio
  모듈의 초기 가중치가 특징 사용 여부와 무관하게 같다(헤드 첫 층은 입력 차원이 달라 예외).
- `use_features=False`: 특징 모듈을 만들지 않고, `"features"` 입력이 있어도 무시한다. 파라미터 수·초기 가중치·학습 결과가
  v2와 비트 단위로 같다(`test_use_features_false_is_identical_to_v2`, 증강 포함 학습 결과 동일).

### 10.2 설정·데이터·추론
| 항목 | 내용 |
|---|---|
| `PolicyConfig` | `use_features=False`, `feature_dim=32`, `feature_hidden=64`, `feature_noise=0.02` |
| `PolicyData.features` | [N, feature_dim] float32 또는 None. 있으면 `observation(idx)`가 `"features"` [B,2,D]를 만든다. t−2 인덱스는 영상과 같은 `prev_index`(같은 연속 구간 시작보다 앞이면 구간 첫 프레임) |
| 증강 | `augment=True`이고 `use_features=True`일 때만 특징에 N(0, 0.02²) 노이즈. 난수는 전용 `numpy Generator(seed+3)`라 영상 증강 난수 흐름과 분리된다 |
| `train_policy` | `use_features=True`인데 `data.features`가 없거나 차원이 `feature_dim`과 다르면 `ValueError` |
| `make_policy_fn` | 모델이 `use_features`이면 입력 dict의 `"features"`가 필수(없으면 `ValueError`, 메시지에 `features` 포함). 아니면 무시 |
| `predict_open_loop` | 모델이 `use_features`인데 `data.features`가 없으면 `ValueError` |
| 폐루프 | 시뮬레이터는 노이즈 검출로 온라인 계산한 특징을 obs에 넣는다(A10 `closed_loop.py`). 정책 쪽은 키만 읽는다 |

### 10.3 파라미터 수(측정)
| 모델 | 파라미터 수 |
|---|---|
| use_features=False(어휘 89) | 519,336 (v2와 같음) |
| use_features=True(어휘 89) | 544,040 (+24,704 = 헤드 첫 층 64×256 + 특징 MLP 64·64+64 + 64·64+64) |
| use_features=True, use_language=False(어휘 89) | 509,224 |

### 10.4 1스레드 처리량(측정)
측정 조건: 7절과 같은 개발 컨테이너(CPU 4코어 가상, torch 2.14 CPU), `threads=1`, 배치 128, 200단계, 무작위 uint8 영상
(`ArrayImageSource`, 4,000프레임), 특징은 균일 난수 [N,32]. 측정 중 load average 약 1~1.9(다른 작업과 CPU 공유 가능성 있음).
각 조건 2회 측정값이다. 처리량 = 200단계 전체 시간(데이터 조회·증강·순전파·역전파·옵티마이저 포함)으로 계산했다.

| 증강 | 특징 | samples/s(1회, 2회) | 3000×128 예상 시간 |
|---|---|---|---|
| 켬 | 끔 | 1,655 / 1,695 | 약 3.8분 |
| 켬 | 켬 | 1,671 / 1,716 | 약 3.8분 |
| 끔 | 끔 | 1,836 / 1,755 | 약 3.6분 |
| 끔 | 켬 | 1,805 / 1,777 | 약 3.6분 |

- 특징 MLP(64→64→64)는 CNN에 비해 연산량이 매우 작아, 특징 on/off 차이는 측정 반복 간 편차(약 ±3%) 안에 있다.
- 7절 표(1스레드 증강 켬 1,198 samples/s)보다 높은 것은 측정 시점의 CPU 부하 차이로 보인다. 두 표를 직접 비교하지 않는다.

### 10.5 검증(`tests/test_vla_policy_v3.py`, 7개)
- 형태: 모델 입력 [B,2,32] → [B,chunk], 특징 누락·차원 불일치 오류, `observation`의 t/t−2 특징이 영상과 같은 경계 규칙을 따름.
- 이상치: 100은 3과, NaN은 0과 같은 출력(clamp·nan_to_num).
- 동일성: `use_features=False`의 파라미터·초기 가중치가 v2와 같고, features 유무와 무관하게 학습 결과가 같음.
- 학습: 영상은 상수이고 특징 한 차원만 가속도를 결정하는 합성 문제에서, 200단계 후 특징 모델의 MAE가 기준선
  (평균 예측)의 25% 미만이다(측정 약 11%). 같은 조건의 특징 없는 모델은 70% 초과다(측정 약 100%).
- 결정성: 같은 시드 → 같은 가중치, 다른 시드·`feature_noise=0` → 다른 가중치.
- `make_policy_fn`: 개루프 예측과 일치, 특징 누락 시 `ValueError`, 특징 미사용 모델은 키를 무시.
