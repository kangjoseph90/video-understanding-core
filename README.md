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

SenseVoice의 태그는 정제된 텍스트와 별도로 `language`, `emotions`, `audio_events`,
`raw_text`에 보존됩니다. 공식 FunASR Python 어댑터는 CPU 모드와 FSMN-VAD를 사용하며,
`indexer.hub`으로 ModelScope(`ms`)와 Hugging Face(`hf`)를 선택할 수 있습니다.

## Setup

Python 3.11과 ffmpeg가 필요합니다.

```bash
uv sync --extra dev --extra sensevoice
uv run vuc index /path/to/video.mp4
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
- **D2:** 고급 ASR은 OpenAI-compatible `/v1/audio/transcriptions`이며 환경변수로 접속 정보를
  주입합니다.
- **D5:** 결과는 Markdown과 구조화 JSON을 모두 생성합니다.
- **D6:** Gemini 네이티브 결과는 선택 비교군으로만 두며 기본 judge는 환경변수로 주입한 별도
  강한 모델입니다.
- **D7:** 초기 인덱스 토큰 예산은 20,000입니다.
- **D8:** Python 3.11, uv, ruff를 사용합니다.
- 프레임 타임스탬프는 한 시간이 넘어도 총 분 수를 유지하는 `[mm:ss]` 형식입니다.
- FunASR 버전별 응답 차이를 고려해 `sentence_info`를 우선 사용하고, 없으면 전체 오디오를
  하나의 세그먼트로 안전하게 보존합니다.

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
