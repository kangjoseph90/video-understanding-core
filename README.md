# video-understanding-core

오디오를 직접 받지 않는 비전 LLM을 위해 로컬 ASR 인덱스와 희소 프레임을 만들고,
필요한 구간만 다시 확인하는 agentic video understanding PoC입니다.

현재 구현 상태: **M2 완료** — M1 인덱서에 라우터, 하이브리드 고급 ASR 인터페이스,
세 가지 에이전트 도구, 캘리브레이션, 직접 구현한 tool-calling loop, JSONL trace 및
Markdown/JSON 보고서 출력을 추가했습니다.

M1 실기 검증은 음성을 포함한 125초 MP4로 수행했습니다. CPU SenseVoice가 VAD 세그먼트 7개와
언어·감정·오디오 이벤트 태그를 저장했고, `00:00`부터 30초 간격인 프레임 4장과 3x3
몽타주 1장을 생성했습니다. 같은 파일의 두 번째 실행은 콘텐츠 해시 캐시를 사용합니다.

## Architecture

```text
local video ─┬─ ffmpeg 16 kHz mono ─► SenseVoice-Small + FSMN-VAD ─┐
             │                                                     ├─► hash cache
             └─ ffmpeg sparse frames ─► timestamp burn-in ─► grid ┘
                                                                    │
                              ┌─ short (<10m) ─► full advanced ASR ─┤
                              └─ long  (≥10m) ─► calibrated agent ──┤
                                                                    ▼
                                                        Markdown + JSON report
```

인덱싱과 초기 프레임 추출은 두 작업으로 동시에 실행됩니다. 캐시는 입력 파일의 SHA-256을
키로 사용하며 `.vuc-cache/<sha256>/` 아래에 `audio.wav`, `index.json`, `index.txt`,
개별 프레임, 몽타주 및 `trace.jsonl`을 저장합니다. JSONL trace에는 각 단계의 인자,
반환 요약, 토큰 사용량 자리, 소요 시간이 기록됩니다.

SenseVoice의 태그는 정제된 텍스트와 별도로 `language`, `emotion`, `events`,
`raw_text`에 보존됩니다. 공식 FunASR Python 어댑터는 CPU 모드와 FSMN-VAD를 사용하며,
`indexer.hub`으로 ModelScope(`ms`)와 Hugging Face(`hf`)를 선택할 수 있습니다.

## Setup

Python 3.11과 ffmpeg가 필요합니다.

```bash
uv sync --extra dev --extra sensevoice --extra advanced-asr
uv run vuc index /path/to/video.mp4
uv run vuc run /path/to/video.mp4 --query "핵심 주장을 요약해줘"
```

OpenAI-compatible 모델 접속 정보는 프로젝트 루트의 `.env`에서 읽습니다. `.env`는 Git에서
제외되며, 새 환경에서는 `.env.example`을 복사해 사용합니다. 이미 프로세스 환경에 설정된
값은 `.env` 값으로 덮어쓰지 않습니다.

```dotenv
VLM_BASE_URL=https://provider.example/v1
VLM_MODEL=glm-5.3-flash
VLM_API_KEY=your-secret
```

최초 실제 실행은 SenseVoice 및 FSMN-VAD 모델을 내려받으므로 네트워크와 모델 캐시 공간이
필요합니다. 모델 없이 개발 테스트만 실행할 때는 다음으로 충분합니다.

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

다른 설정 파일은 `--config`, 기존 캐시 무시는 `--force`로 지정합니다.

```bash
uv run vuc index sample.webm --config config.yaml --force
uv run vuc run sample.webm --config config.yaml --force-index
```

## Configuration

모델 엔드포인트, 샘플링, 이미지, 도구, 에이전트 예산과 벤치마크 judge 설정을 모두
단일 [`config.yaml`](config.yaml)에 둡니다. 아직 구현되지 않은 M3 설정도 같은 스키마에
미리 고정해 이후 별도 설정 파일이 생기지 않게 했습니다.

## M2 behavior

- `route_video`는 600초 미만 영상을 baseline, 이상을 agentic 경로로 보냅니다.
- baseline은 고급 ASR을 제공자 상한 이하로 나눠 전체 전사하고 15초/512px 프레임과 함께
  한 번의 VLM 호출로 보고서를 만듭니다.
- agentic 경로는 영상의 앞·중간·뒤 시간층에서 무작위 30초 구간 3개를 고급 ASR로 비교해
  인덱스 신뢰도를 계산합니다. 인덱스 세그먼트가 창 경계를 넘으면 시간 비율로 텍스트를
  절단하고, 양쪽 텍스트에서 공백·문장부호를 제거한 CER의 `1-CER`를 사용합니다. 영상 해시를
  seed로 사용하므로 표본은 재현 가능합니다.
- 로컬 `transcribe_segment` 상한은 180초, cloud 상한은 600초입니다. ASR wall time은 에이전트
  600초 wall-clock 예산에서 제외하고 `asr_wall_clock_s`로 따로 기록합니다.
- 제공하는 도구는 정확히 `view_frames(start_s, end_s, fps, resolution)`,
  `transcribe_segment(start_s, end_s)`, `read_index(start_s, end_s)`입니다. `read_index`는 전체
  인덱스가 초기 20k 토큰 예산을 넘을 때만 모델에 노출됩니다.
- 고급 ASR 결과에는 `notes: str | null`이 있으며 local은 `null`입니다. M3의 Gemini 제공자가
  비언어 정보를 전달할 수 있도록 인터페이스를 미리 확보했습니다.
- 각 citation은 `evidence_span(start_s, end_s, source)`을 가집니다. claim 구간의 80% 이상이
  전체 조회 구간 합집합에 포함되고, claim과 evidence 구간이 80% 이상 겹치며, evidence 구간도
  선언한 source(`view_frames` 또는 `transcribe_segment`)로 80% 이상 실제 조회된 경우에만
  `verified=true`입니다. 모델이 반환한 `verified` 값은 사용하지 않습니다.
- 특정 구간의 시각적 주장은 같은 구간의 `view_frames` 근거만 사용하도록 시스템 프롬프트에
  강제합니다.
- 에이전트 예산은 도구 12회, 누적 입력 200k 토큰, ASR 제외 wall-clock 600초입니다. 한도에
  도달하면 도구를 중단하고 현재 근거만으로 최종 보고서를 요청합니다.
- 종료 전 섹션 시간 합집합을 계산합니다. 영상의 90% 미만이면 공백 구간을 모델에 전달해
  추가 조회 또는 명시를 요구하며, 최종 `meta.coverage_ratio`와 `uncovered_spans`에도 기록합니다.
- `meta.cumulative_input_tokens`는 모든 VLM 왕복의 입력 토큰 누적치이고 `output_tokens`도 모든
  왕복의 출력 누적치입니다. `vlm_cost_usd`는 config의 백만 토큰당 입출력 단가로 계산하며,
  두 단가가 모두 0이면 비용 미설정을 뜻하는 `null`입니다.

## Assumptions

- **D1:** 배포 타깃은 x86_64, CPU 8코어, RAM 16GB이며 로컬 모델 피크 메모리는 12GB
  이하여야 합니다. 현재 개발 호스트는 arm64/12코어/16GB라서 M1 기능 테스트만 이 환경에서
  수행하며, x86_64 메모리 측정은 별도 검증 항목으로 남깁니다.
- **D2:** 고급 ASR은 하이브리드입니다. 기본값은 로컬 faster-whisper
  `large-v3-turbo`(CPU `int8`)이고, `asr.advanced.provider: local|cloud`로 선택합니다.
  클라우드 구현은 OpenAI-compatible `/v1/audio/transcriptions`이며 M2에서는 stub이어도 됩니다.
- **D5:** 결과는 Markdown과 구조화 JSON을 모두 생성합니다.
- **D6:** Gemini 네이티브 결과는 선택 비교군으로만 두며 기본 judge는 환경변수로 주입한 별도
  강한 모델입니다.
- **D7:** 초기 인덱스 토큰 예산은 20,000입니다.
- **D8:** Python 3.11, uv, ruff를 사용합니다.
- **D9:** 기존 목표 단가는 유지합니다. 비용 집계에서 로컬 고급 ASR은 처리 시간(초),
  클라우드 고급 ASR은 USD로 분리합니다. 현재 전달된 명세에는 목표 단가의 숫자가 없어
  수치를 새로 가정하지 않습니다.
- 프레임 타임스탬프는 한 시간이 넘어도 총 분 수를 유지하는 `[mm:ss]` 형식입니다.
- FunASR 버전별 응답 차이를 고려해 `sentence_info`를 우선 사용하고, 없으면 전체 오디오를
  하나의 세그먼트로 안전하게 보존합니다.

## Measurements

### M1 long-video validation

2026-09-08에 arm64 macOS, 12코어, RAM 16GB 환경에서 측정했습니다. 로컬의 55.7초 실제
소개 영상을 반복하고 640px/5fps로 변환한 30분 MP4를 사용했으므로, 처리량과 메모리 측정에는
유효하지만 콘텐츠 다양성 평가는 아닙니다. 캐시를 강제로 무시한 전체 `vuc index` 프로세스를
`/usr/bin/time -l`로 계측했습니다.

| video | wall time | realtime multiple | peak RSS | segments | mean segment |
|---:|---:|---:|---:|---:|---:|
| 1,800.0s | 199.59s | 9.02× | 3,106.1MB (2,962.2MiB) | 125 | 14.40s |
| 1,800.0s, suffix fix | 197.59s | 9.11× | 3,128.1MB (2,983.2MiB) | 125 | 14.40s |

- 프레임 추출과 몽타주 생성은 60장/7장으로 약 1초였으며, 나머지 시간 대부분은 CPU
  SenseVoice 추론에 사용됐습니다.
- 125개 세그먼트 모두에서 인라인 태그가 정제된 `text`에서 제거됐고 `emotion`과
  `events[]`로 저장됐습니다. 원본 증거 보존용 `raw_text`에는 인라인 태그를 유지합니다.
  이 입력에서 관찰한 감정은 `neutral`, `emo_unknown`, 이벤트는 `Speech`였습니다.
- `start`/`end`는 FSMN-VAD 세그먼트 경계에서 온 값입니다. 세그먼트 내부 단어·문장 위치의
  정밀 타임스탬프는 제공하지 않습니다.
- 첫 3×3 몽타주를 원본 크기로 열어 `00:00`부터 `04:00`까지 30초 간격 burn-in을 육안
  확인했습니다. 검은 배경의 흰 글자가 각 셀 좌하단에서 선명해 기본 폰트 크기를 유지합니다.

### Index suffix duplication and calibration

`치는는`, `중에서에서`, `걸어주세요세요`를 조사한 결과 125개 VAD 세그먼트 사이 시간 겹침은
0건이었고, 각각 20·25·13회가 SenseVoice의 개별 `raw_text` 안에 이미 존재했습니다. 따라서
VAD overlap이나 세그먼트 merge 경계 중복이 아니라 동일 음원을 반복 입력했을 때 재현되는
SenseVoice 오류입니다. `raw_text`는 증거로 보존하고, 정제 `text`에만 보수적인 한국어
조사·어미 suffix 중복 제거를 적용했습니다. 재인덱싱 후 세 표현은 정제 `text`에서 모두
0건입니다.

동일한 층화 30초 창과 기존 faster-whisper 결과를 사용한 신뢰도 변화는 다음과 같습니다.
아래 전후 비교는 모두 새 CER 방식이므로 후처리 효과만 비교합니다.

| stage | window similarities | mean index reliability |
|---|---|---:|
| fix 전, 새 CER로 재계산 | 0.708955 / 0.812500 / 0.798507 | 0.773321 |
| fix 후 재인덱싱 | 0.708955 / 0.843750 / 0.828358 | 0.793688 |

신뢰도는 `+0.020367`(약 2.04%p) 상승했습니다. 기존 README의 0.739는 창과 겹치는 세그먼트
전체를 넣은 `SequenceMatcher` 값이라 새 수치와 직접 비교하지 않습니다.

### Vision LLM pre-spike

재현 스크립트는 `scripts/spike_vision.py`입니다. 첫 실행의 공급자 5시간 사용량 한도(HTTP
429)가 초기화된 뒤 2026-09-08 22:16 KST에 `glm-5.3-flash`로 다시 측정했습니다.

| probe | result | latency | prompt tokens | total tokens |
|---|---:|---:|---:|---:|
| 3×3 timestamp reading | 9/9 (100%) | 12.208s | 525 | 658 |
| image + forced `view_frames` call | success | 6.682s | 745 | 901 |
| 1 image | success | 4.905s | 496 | 699 |
| 4 images | success | 3.573s | 1,846 | 1,970 |
| 8 images | success | 4.817s | 3,646 | 3,894 |
| 16 images | success | 6.137s | 7,246 | 7,674 |

- 타임스탬프 응답은 `00:00`부터 `04:00`까지 30초 간격인 9개 값을 순서까지 정확히
  반환했습니다.
- 이미지 입력과 JSON schema를 함께 보낸 강제 tool-call 요청은
  `view_frames(start_s=0, end_s=30, fps=0.1, resolution=512)`를 반환했습니다.
- 1, 4, 8, 16장 모두 성공해 설정 상한인 16장까지 실패 지점이 관찰되지 않았습니다.
- **기본값 제안:** 256px 초기 프레임과 3×3 몽타주를 유지합니다. 가장 작은 현재 기본값에서
  burn-in 판독률이 100%였고 16장 입력도 성공했으므로 해상도나 이미지 수를 늘릴 근거가
  없습니다. 단, 30초 간격의 3시간 영상은 20개 몽타주가 되어 16장 상한을 넘으므로 M2에서
  초기 개요를 균등 다운샘플하거나 두 요청으로 나누는 정책이 필요합니다.

### M2 integration smoke

실제 55.7초 영상의 baseline 경로와 반복 생성한 30분 영상의 agentic 경로를
`glm-5.3-flash` 및 로컬 faster-whisper `large-v3-turbo` CPU int8로 실행했습니다.
아래 표는 evidence-span과 CER 보정 전의 원래 M2 실행 기록입니다.

| route | e2e | agent wall (ASR 제외) | ASR wall | tool calls | cumulative input tokens | citations |
|---|---:|---:|---:|---:|---:|---:|
| baseline, 55.7s | 134.7s | 52.2s | 67.7s | 0 | 2,538 | 5/6 verified |
| agentic, 30m | 186.6s | 83.4s | 103.1s | 8 | 80,348 | 9/9 verified |

baseline의 미검증 인용 1개는 모델이 영상 길이 55.7초를 넘어 `[00:55–01:00]`을 인용해
실제 조회 구간 겹침이 14.1%에 그친 사례입니다. agentic 실행은 층화 캘리브레이션 신뢰도
0.739를 얻었고, 프레임 수 상한 오류에 대해 구간을 줄여 재호출한 뒤 완전한 JSON 보고서를
생성했습니다. 첫 agentic 시도에서는 4,096 출력 토큰에서 JSON이 잘려 기본 상한을 8,192로
조정했으며 재실행으로 해결됐습니다.

보고서의 이름 `강요섭`은 파일명에서 가져온 값이 아닙니다. baseline의 0.000–55.706초 및
agentic의 0–60초 faster-whisper `large-v3-turbo` 전사에 실제로 `강요섭`이 있었고, VLM에는
원본 파일 경로를 보내지 않았습니다. SenseVoice 초안은 `각요셉`, 파일명은 `강요셉`이므로
현재 보고서 표기는 고급 ASR 근거에서 온 것입니다. 화면 텍스트 교차검증은 없어 보고서도
정확한 한글 표기를 미검증 항목으로 남겼습니다.

CER/evidence-span 수정 후 30분 보고서 재생성을 두 차례 시도했지만 VLM 제공자가 첫 응답 전에
HTTP 429를 반환했습니다. 따라서 디스크의 기존 30분 보고서는 위 표의 M2 원본이며 새 스키마로
덮어쓰지 않았습니다. 새 검증·자기점검·메타 경로는 mock VLM/ffmpeg 통합 테스트로 검증했습니다.

## Benchmark results

M4에서 다국어 평가셋으로 채웁니다.

| mode | videos | e2e latency | cloud ASR seconds | estimated ASR cost | report quality |
|---|---:|---:|---:|---:|---:|
| agentic | — | — | — | — | — |
| baseline_full | — | — | — | — | — |
| baseline_index_only | — | — | — | — | — |

## Roadmap

- M2: 완료 — 세 도구, 캘리브레이션, 직접 구현한 tool-calling loop, router, 보고서 출력
- M3: 세 비교군, manifest benchmark, citation/report judge
- M4: 4개 이상 언어와 코드 스위칭을 포함한 실제 평가 및 기본값 제안
