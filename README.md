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

2026-09-09, manifest 5편(551~1027초, en/ko/ja) × 3모드 = 15회. glm-5.3-flash,
SenseVoice CPU 인덱싱, faster-whisper large-v3-turbo CPU 고급 ASR.

| mode | videos | indexing | ASR | VLM | total | input/output tokens | cost | sections | citations | tool calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_index_only | 5 | 83.3s | — | 100.2s | 183.5s | 9,508 / 5,080 | $0.0040 | 6.2 | 21.0 | 0 |
| agentic | 5 | 83.3s | 20.1s | 103.2s | 206.8s | 16,385 / 5,678 | $0.0048 | 6.0 | 23.4 | 6 |
| baseline_full | 5 | — | 245.4s | 150.0s | 403.0s | 109,612 / 6,005 | $0.0169 | 6.0 | 25.0 | 0 |

영상별 평균입니다. `indexing`은 SenseVoice와 15초 프레임의 cold 비용이며, `index_only`와
`agentic`의 공통 전제 조건입니다. `agentic`은 앞선 `index_only` 실행의 캐시를 재사용해
측정값이 0으로 찍히므로 `total`에는 이 값을 더한 cold 등가치를 적었습니다. `baseline_full`은
인덱스를 만들지 않습니다. `baseline_full`의 강의·요리 2건은 429와 JSON 스키마 이탈로
재실행되어 ASR이 hot이었고, `total`과 `ASR`은 첫 시도의 cold 측정치(342s/196.6s,
364s/222.1s)로 환산했습니다. 열 합이 `total`과 다른 것은 1초 프레임 추출·몽타주
단계(`baseline_full` 평균 7.0초)를 표에 넣지 않았기 때문입니다.

`agentic`은 `baseline_full` 대비 입력 토큰을 6.7배, 비용을 3.5배, 지연을 1.95배 줄이면서
인용은 6% 적습니다(23.4 vs 25.0). 지연의 지배 변수는 CPU Whisper로, `baseline_full` 총
지연의 61%입니다. `agentic`은 필요한 구간만 재전사해 평균 20.1초를 씁니다.

인용 밀도는 영상에 따라 갈립니다. `baseline_full`이 앞선 것은 요리(23 vs 13), 일본어
브이로그(41 vs 35), 강의(26 vs 22)이고, 뒤진 것은 한국어 브이로그(16 vs 29)와 talking
head(19 vs 18)입니다. 한국어 브이로그(1027초)에서 `baseline_full`은 411초를 ASR에 쓰고
151,029토큰을 넣었지만 4섹션 16인용에 그쳤고 추론 토큰이 1개였습니다 — 1,027프레임에서
만든 115 몽타주가 오히려 품질을 떨어뜨렸습니다. 같은 영상에서 `agentic`은 63.9초와
11,514토큰으로 29인용을 냈습니다.

도구 호출은 15회 합계 6회이고 **전부 `transcribe_segment`**입니다(요리 2, talking head 2,
일본어 브이로그 2). `view_frames`는 한 번도 호출되지 않았습니다. 도구를 쓴다고 인용이
항상 늘지도 않습니다 — 요리 `agentic`은 도구 2회에도 13인용으로 `index_only`(14)보다
적었습니다.

## Roadmap

- M2: 명시적 세 모드, 두 도구, tool-calling loop, JSONL trace, Markdown/JSON 보고서
- M3: 동일 manifest에서 세 모드 비교 실행과 report judge
- M4: 4개 이상 언어와 code-switching 평가 및 기본값 제안
