# video-understanding-core

오디오를 직접 받지 않는 비전 LLM에 전사와 타임스탬프 몽타주를 제공하고, 동일한 입력에서
세 가지 명시적 실행 방식을 비교하기 위한 video understanding PoC입니다.

## Pipeline

`run.mode`는 영상 길이와 무관하게 config에 적힌 값만 사용합니다. 자동 라우팅은 없습니다.

```text
baseline_full
  video ─► full faster-whisper ASR (180s chunks; cloud 600s)
        └► 1s frames ─► 3×3 montages ─► one VLM request ─► report

baseline_index_only
  video ─► full SenseVoice ASR + FSMN-VAD
        └► 15s frames ─► 3×3 montages ─► one VLM request ─► report

agentic
  video ─► full SenseVoice ASR + FSMN-VAD
        └► 15s frames ─► 3×3 montages ─► VLM tool loop ─► report
                                                    ├─ view_frames
                                                    └─ transcribe_segment
```

- `baseline_full`은 SenseVoice를 실행하지 않습니다. 전체 오디오를 advanced ASR로 전사하고,
  1초 간격 프레임을 3×3으로 묶어 모든 몽타주를 한 번의 VLM 요청에 넣습니다.
- `baseline_index_only`는 전체 SenseVoice 인덱스와 15초 간격 3×3 몽타주를 모두 한 번의
  VLM 요청에 넣으며 도구를 제공하지 않습니다.
- `agentic`은 `baseline_index_only`와 같은 초기 입력을 사용하고, 필요한 경우에만 프레임과
  advanced ASR을 추가 조회할 수 있습니다. SenseVoice만으로 충분하면 도구 호출 없이
  종료할 수 있습니다.
- 일반 VLM 요청의 이미지 수를 애플리케이션에서 제한하거나 균등 다운샘플하지 않습니다.

SenseVoice의 인라인 태그는 정제된 `text`에서 제거되고 `language`, `emotion`, `events`에
분리됩니다. 원문은 `raw_text`에 보존합니다. 세그먼트 `start`/`end`는 FSMN-VAD 경계이며,
세그먼트 내부 단어 위치의 정밀 타임스탬프는 아닙니다.

## Tools

agentic 모드에는 두 도구만 노출합니다.

```text
transcribe_segment(start_s, end_s)
view_frames(start_s, end_s, fps, n)
```

- `transcribe_segment`의 한 번 호출 구간은 `tools.transcribe_segment.max_duration_s`와 provider
  상한 중 작은 값까지이며 기본값은 60초입니다.
  기본 provider는 로컬 faster-whisper `large-v3-turbo`, CPU `int8`입니다. cloud provider는
  인터페이스만 있는 stub입니다.
- `view_frames`에서 에이전트가 시간 구간, fps, 정사각 몽타주의 한 변 `n`을 선택합니다.
  `fps`는 `0.1|0.2|0.5|1|2`, `n`은 `1|2|3|4` 중에서 선택합니다.
- 모든 몽타주의 캔버스 크기와 JPEG 품질은 전역 `montage` 설정을 사용하며 기본 캔버스는
  1344×756입니다. 셀 크기는 요청한 `n`으로 결정되며, 마지막
  묶음은 셀 크기를 유지하면서 프레임을 담을 수 있는 최소 정사각 grid로 축소합니다. 예를
  들어 3×3 요청의 마지막 4프레임은 빈칸 없는 896×504의 2×2 몽타주가 됩니다.
- 한 호출의 제한은 원본 프레임 16장이 아니라 반환되는 몽타주 이미지 16장입니다.
  예상 몽타주 수는 `ceil(ceil((end_s-start_s)*fps) / n²)`입니다. `n`은 양의 정수이며 별도
  연속값이 아닙니다.

보고서 citation은 `claim`, `start_s`, `end_s`만 가집니다. 조회 여부를 근거로 강제 판정하거나
보고서 섹션의 시간 범위를 강제하는 후처리는 없습니다.

## Setup

Python 3.11과 ffmpeg가 필요합니다.

```bash
uv sync --extra dev --extra sensevoice --extra advanced-asr
uv run vuc index /path/to/video.mp4
uv run vuc run /path/to/video.mp4 --query "핵심 주장을 요약해줘"
```

OpenAI-compatible VLM 접속 정보는 프로젝트 루트의 `.env`에서 읽습니다. `.env`는 Git에서
제외되며 `.env.example`을 복사해 사용할 수 있습니다. 이미 설정된 프로세스 환경 변수는
덮어쓰지 않습니다.

```dotenv
VLM_BASE_URL=https://provider.example/v1
VLM_MODEL=your-model
VLM_API_KEY=your-secret
```

실행 모드는 [`config.yaml`](config.yaml)에 명시합니다.

```yaml
run:
  mode: agentic  # baseline_full | baseline_index_only | agentic
```

다른 설정 파일은 `--config`, 인덱스 캐시 무시는 `--force` 또는 `--force-index`로 지정합니다.

```bash
uv run vuc index sample.webm --config config.yaml --force
uv run vuc run sample.webm --config config.yaml --force-index
```

모델 없이 개발 테스트만 실행할 때는 다음으로 충분합니다.

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## Configuration and accounting

주요 기본값은 다음과 같습니다.

| item | default |
|---|---:|
| index / agentic frame sampling | 15s |
| baseline_full frame sampling | 1s |
| montage reference canvas | 1344×756 |
| initial montage grid | 3×3 |
| tool grid choices | 1×1, 2×2, 3×3, 4×4 |
| tool fps choices | 0.1, 0.2, 0.5, 1, 2 |
| tool montage limit | 16 images/call |
| baseline_full ASR chunk | local 180s / cloud 600s |
| agentic ASR tool interval | 60s/call |
| agent tool calls | 12 |
| agent wall-clock budget | 600s excluding ASR |

`meta.cumulative_input_tokens`는 agentic의 모든 VLM turn에 보고된 입력 토큰을 합한 값입니다.
provider가 캐시 토큰을 별도로 보고하더라도 이 필드는 청구 토큰으로 재계산하지 않습니다.
`output_tokens`도 모든 turn의 합입니다. `vlm_cost_usd`는 config의 백만 토큰당 입출력 단가로
계산하며 두 단가가 모두 0이면 `null`입니다.

ASR 시간은 에이전트 wall-clock 예산과 분리합니다. 로컬 ASR은 `asr_processing_s`, cloud
ASR은 처리한 오디오 초와 `cloud_asr_cost_usd`로 각각 기록합니다.

캐시는 입력 파일 SHA-256을 키로 `.vuc-cache/<sha256>/` 아래에 저장됩니다. SenseVoice를 쓰는
모드는 `audio.wav`, `index.json`, `index.txt`, 프레임, 몽타주를 공유합니다. 각 `vuc run`의
보고서와 trace는 `runs/<run-id>/report.md`, `report.json`, `trace.jsonl`로 분리되어 이전 실행을
덮어쓰거나 서로 섞지 않습니다.

## Measurements

### M1 long-video validation

2026-09-08에 arm64 macOS, 12코어, RAM 16GB 환경에서 측정했습니다. 55.7초 실제 소개 영상을
반복해 만든 30분 MP4이므로 처리량과 메모리 측정에는 유효하지만 콘텐츠 다양성 평가는
아닙니다. 당시 프레임 설정은 현재 기본값과 달리 30초 간격이었습니다.

| video | wall time | realtime multiple | peak RSS | segments | mean segment |
|---:|---:|---:|---:|---:|---:|
| 1,800.0s | 199.59s | 9.02× | 3,106.1MB | 125 | 14.40s |

- 125개 세그먼트 모두에서 인라인 태그가 `text`에서 제거되고 별도 필드에 저장됐습니다.
- 첫 3×3 몽타주를 원본 크기로 열어 `00:00`부터 `04:00`까지 burn-in이 읽히는 것을
  확인했습니다.

### Vision input spike

2026-09-08 OpenAI-compatible VLM에 대해 3×3 타임스탬프 판독, 이미지와 tool schema의 동시
입력, 이미지 1/4/8/16장 입력을 확인했습니다.

| probe | result | latency | prompt tokens | total tokens |
|---|---:|---:|---:|---:|
| 3×3 timestamp reading | 9/9 | 12.208s | 525 | 658 |
| image + forced tool call | success | 6.682s | 745 | 901 |
| 1 image | success | 4.905s | 496 | 699 |
| 4 images | success | 3.573s | 1,846 | 1,970 |
| 8 images | success | 4.817s | 3,646 | 3,894 |
| 16 images | success | 6.137s | 7,246 | 7,674 |

16장까지 성공했다는 관측은 provider 입력 실험 결과이며 현재 파이프라인의 VLM 이미지 상한이
아닙니다. 재현 스크립트는 [`scripts/spike_vision.py`](scripts/spike_vision.py)입니다.

대상 영상 manifest는 [`eval/youtube-field-eval.yaml`](eval/youtube-field-eval.yaml)이고, 각 영상
옆의 `<video>.meta.json` 사이드카가 채널명·제목·챕터를 힌트로 제공합니다. 스윕 재현은
[`scripts/run_field_eval.py`](scripts/run_field_eval.py)이며 실행별 원자료는
[`eval/field-eval-results.jsonl`](eval/field-eval-results.jsonl)에 있습니다.

## Benchmark results

manifest 5편(551~1027초, en/ko/ja)에 세 모드를 돌린 결과입니다. 인덱싱은 SenseVoice CPU,
고급 ASR은 faster-whisper large-v3-turbo CPU, VLM은 OpenAI 호환 엔드포인트입니다.
실행별 원자료는 [`eval/field-eval-results.jsonl`](eval/field-eval-results.jsonl)(glm)과
[`eval/field-eval-deepseek.jsonl`](eval/field-eval-deepseek.jsonl)에 있습니다.

### 비용과 지연 (2026-09-09, z-ai/glm-5.3-flash)

| mode | videos | indexing | ASR | VLM | total | input/output tokens | cost | tool calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_index_only | 5 | 83.3s | — | 100.1s | 183.5s | 9,508 / 5,080 | $0.0040 | 0 |
| agentic | 5 | 83.3s | 20.1s | 103.2s | 206.8s | 16,385 / 5,678 | $0.0048 | 6 |
| baseline_full | 5 | — | 245.4s | 150.0s | 403.0s | 109,612 / 6,005 | $0.0169 | 0 |

영상별 평균입니다. `indexing`은 SenseVoice와 15초 프레임의 cold 비용이며 `index_only`와
`agentic`의 공통 전제 조건입니다. `agentic`은 앞선 실행의 인덱스 캐시를 재사용해 측정값이
0으로 찍히므로 `total`에는 이를 더한 cold 등가치를 적었습니다. `baseline_full`은 인덱스를
만들지 않습니다. `baseline_full`의 강의·요리 2건은 429와 JSON 스키마 이탈로 재실행되어 ASR이
hot이었고, `total`과 `ASR`은 첫 시도의 cold 측정치(342s/196.6s, 364s/222.1s)로 환산했습니다.
열 합이 `total`과 어긋나는 것은 1초 프레임 추출·몽타주 단계(평균 7.0초)를 표에서 뺐기
때문입니다.

`agentic`은 `baseline_full` 대비 입력 토큰 6.7배, 비용 3.5배, 지연 1.95배를 줄입니다. 지연의
지배 변수는 CPU Whisper로 `baseline_full` 총 지연의 61%입니다.

### 보고서 품질 (블라인드 판정)

보고서 15개를 영상별로 A/B/C 무작위 배치하고 `meta`를 제거해 모드를 가린 뒤, 캐시된 Whisper
전사를 근거로 인용을 검증하고 순위를 매겼습니다.

| 영상 | 1위 | 2위 | 3위 |
|---|---|---|---|
| 요리 (603s) | agentic | index_only | baseline_full *(결정적)* |
| 슬라이드 강의 (716s) | baseline_full | agentic | index_only |
| Talking head (551s) | baseline_full | index_only | agentic *(결정적)* |
| 한국어 브이로그 (1027s) | agentic | index_only | baseline_full *(결정적)* |
| 일본어 브이로그 (800s) | baseline_full | agentic | index_only *(결정적)* |

Borda 점수는 `agentic` 11, `baseline_full` 11, `baseline_index_only` 8입니다. **품질만 보면
`agentic`과 `baseline_full`은 동점**이고, `agentic`을 택할 근거는 같은 품질을 3.5배 싸게
얻는다는 점과 아래의 확장성 한계입니다.

**인용 개수는 품질 지표가 아닙니다.** 인용 개수 순위와 블라인드 품질 순위의 쌍대 일치율은
10/15(67%)로 무작위 기대(50%)를 겨우 넘습니다. 요리에서는 완전히 반대였습니다 —
`baseline_full`이 23인용으로 최다였지만 결정적 최하위였습니다.

그 요리 사례가 `baseline_full`의 실패 양상을 잘 보여줍니다. Whisper가 분수를 `3 1⁄4`로
깨뜨렸는데 SenseVoice는 `three quarter cup`으로 정확히 받아썼습니다. Whisper만 보는
`baseline_full`은 반죽 배합을 `밀가루 3¼컵`으로 적어 레시피를 망가뜨렸고, 인덱스를 가진 두
모드는 모두 `3/4컵`으로 맞췄습니다.

**세 모드 모두 없는 사실을 만들어냅니다.** 발표자 이름, 전쟁 국가명, 가격 견적, 음악 크레딧,
존재하지 않는 재료가 각 모드에서 최소 한 번씩 나왔습니다. 현재 품질은 연구 베이스라인으로는
유효하지만 요약을 검증 없이 신뢰하는 용도에는 쓸 수 없습니다.

### 도구 사용 실태

두 모델 합쳐 도구 호출은 8회이고 **전부 `transcribe_segment`**입니다. `view_frames`는 한 번도
호출되지 않았습니다.

실효는 제한적입니다. 8회 중 실질적으로 기여한 것은 고유명사 교정 2건(`Tenentrum`→`Centrum`,
`Public domainoma Day`→`Public Domain Day` 등)이고, 그중 1건은 도구가 `Piotr Dzentara`를
정확히 돌려줬는데도 보고서에 반영되지 않았습니다. 일본어 브이로그 [203-263] 호출은 32초를
쓰고 Whisper 환각(`ご視聴ありがとうございました`)만 받았습니다.

도구를 부르지 않은 것은 대체로 타당했습니다. 인덱스와 Whisper 기준의 글자 수를 비교하면
5편 중 4편이 0.95~1.11로 사실상 동등합니다.

| 영상 | 세그먼트 | 인덱스/Whisper 글자수 비 |
|---|---:|---:|
| talking-head | 42 | 0.98 |
| cooking | 45 | 1.00 |
| slide-lecture | 52 | 0.95 |
| japanese-vlog | 54 | 1.11 |
| korean-vlog | 6 | **0.02** |

한국어 브이로그만 인덱스가 비어 있는데, 음악과 화면 자막 위주라 재전사해도 얻을 것이 없어
두 모델 모두 도구를 부르지 않은 것이 옳았습니다. 이 영상은 `view_frames`가 유효했을 유일한
후보였지만 호출되지 않았습니다.

### 모델 교체 (2026-09-11, deepseek/deepseek-v4.1-flash)

동일 매니페스트·동일 캐시에 VLM만 바꿔 15회를 다시 돌렸습니다.

| mode | model | VLM | input | output | reasoning | cost | tools |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline_index_only | glm-5.3-flash | 100.1s | 9,508 | 5,080 | 2,175 | $0.0040 | 0 |
| | deepseek-v4.1-flash | 113.9s | 5,585 | 8,569 | 5,905 | $0.0120 | 0 |
| agentic | glm-5.3-flash | 103.2s | 16,385 | 5,678 | 2,246 | $0.0048 | 6 |
| | deepseek-v4.1-flash | **21.0s** | 7,487 | 3,360 | 726 | $0.0062 | 2 |
| baseline_full | glm-5.3-flash | 141.0s | 109,612 | 6,005 | 2,704 | $0.0169 | 0 |
| | deepseek-v4.1-flash | 80.2s | 47,004 | 6,349 | 3,460 | $0.0217 | 0 |

deepseek는 `baseline_full` 1건(한국어 브이로그)이 실패해 n=4입니다. 품질 판정은 아직 하지
않았고, 아래는 비용·지연 관찰입니다.

같은 프롬프트인데 **입력 토큰이 절반 이하**입니다(`baseline_full` 109,612 → 47,004). 이미지
토큰 환산 방식 차이로, 입력 단가가 2배인데도 총비용은 1.3배에 그칩니다. `agentic`은 VLM
지연이 103.2초 → 21.0초로 5배 빨라집니다.

대신 **추론 토큰 편차가 큽니다.** `index_only`에서 출력의 69%가 추론이고(5,905/8,569) 그 탓에
`agentic`보다 2배 비쌉니다. 입력이 부실할수록 추론을 늘리는 경향으로 보입니다.

### baseline_full의 확장 한계

`baseline_full`은 1초 간격이라 요청 본문이 영상 길이에 정비례합니다. 한국어 브이로그(1027초,
몽타주 115장, base64 32.3MB)는 OpenRouter에서 **413 Payload Too Large**로 거부됐고, 일본어
브이로그(800초, 89장, 27.6MB)는 통과했습니다. 한도는 그 사이에 있습니다. 20분 이상 영상에는
이 모드를 쓸 수 없습니다.

## Roadmap

- M2: 명시적 세 모드, 두 도구, tool-calling loop, JSONL trace, Markdown/JSON 보고서
- M3: 동일 manifest에서 세 모드 비교 실행과 report judge
- M4: 4개 이상 언어와 code-switching 평가 및 기본값 제안
