# video-understanding-core

오디오를 직접 받지 않는 비전 LLM을 위해 로컬 ASR 인덱스와 희소 프레임을 만들고,
필요한 구간만 다시 확인하는 agentic video understanding PoC입니다.

현재 구현 상태: **M1 완료** — SenseVoice 인덱서, 초기 프레임/몽타주, 콘텐츠 해시 캐시,
JSONL trace, `vuc index` 및 ffmpeg 스모크 테스트.

M1 실기 검증은 음성을 포함한 125초 MP4로 수행했습니다. CPU SenseVoice가 VAD 세그먼트 7개와
언어·감정·오디오 이벤트 태그를 저장했고, `00:00`부터 30초 간격인 프레임 4장과 3x3
몽타주 1장을 생성했습니다. 같은 파일의 두 번째 실행은 콘텐츠 해시 캐시를 사용합니다.

## Architecture

```text
local video ─┬─ ffmpeg 16 kHz mono ─► SenseVoice-Small + FSMN-VAD ─┐
             │                                                     ├─► hash cache
             └─ ffmpeg sparse frames ─► timestamp burn-in ─► grid ┘
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
uv sync --extra dev --extra sensevoice
uv run vuc index /path/to/video.mp4
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
```

## Configuration

모델 엔드포인트, 샘플링, 이미지, 도구, 에이전트 예산과 벤치마크 judge 설정을 모두
단일 [`config.yaml`](config.yaml)에 둡니다. 아직 구현되지 않은 M2/M3 설정도 같은 스키마에
미리 고정해 이후 별도 설정 파일이 생기지 않게 했습니다.

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

- 프레임 추출과 몽타주 생성은 60장/7장으로 약 1초였으며, 나머지 시간 대부분은 CPU
  SenseVoice 추론에 사용됐습니다.
- 125개 세그먼트 모두에서 인라인 태그가 정제된 `text`에서 제거됐고 `emotion`과
  `events[]`로 저장됐습니다. 원본 증거 보존용 `raw_text`에는 인라인 태그를 유지합니다.
  이 입력에서 관찰한 감정은 `neutral`, `emo_unknown`, 이벤트는 `Speech`였습니다.
- `start`/`end`는 FSMN-VAD 세그먼트 경계에서 온 값입니다. 세그먼트 내부 단어·문장 위치의
  정밀 타임스탬프는 제공하지 않습니다.
- 첫 3×3 몽타주를 원본 크기로 열어 `00:00`부터 `04:00`까지 30초 간격 burn-in을 육안
  확인했습니다. 검은 배경의 흰 글자가 각 셀 좌하단에서 선명해 기본 폰트 크기를 유지합니다.

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

## Benchmark results

M4에서 다국어 평가셋으로 채웁니다.

| mode | videos | e2e latency | cloud ASR seconds | estimated ASR cost | report quality |
|---|---:|---:|---:|---:|---:|
| agentic | — | — | — | — | — |
| baseline_full | — | — | — | — | — |
| baseline_index_only | — | — | — | — | — |

## Roadmap

- M2: 세 도구, 캘리브레이션, 직접 구현한 tool-calling loop, router, 보고서 출력
- M3: 세 비교군, manifest benchmark, citation/report judge
- M4: 4개 이상 언어와 코드 스위칭을 포함한 실제 평가 및 기본값 제안
