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
  video ─► index ─► one VLM request ─► report

agentic
  video ─► index ─► VLM tool loop ─► report
                              ├─ view_frames
                              └─ transcribe_segment
```

프롬프트를 구성하는 것은 네 가지이고, 데이터 모델과 캐시 파일도 같은 네 가지로
나뉩니다. 서로를 덮어쓰거나 정정하지 않으며, 합치는 fusion 단계는 없습니다.

```text
메타데이터     <video>.meta.json ─► 채널명, 제목, 길이, 챕터

자막           <video>.subs.json ─► 원본 언어 트랙 하나 ─► 귀속 판정
                                                        ├─ 음성 전사  ─► 음성 인덱스 교정
                                                        └─ 하드섭 사본 ─► 텍스트 인덱스 교정

음성 인덱스     video ─► FSMN-VAD ─┬─ speech     ─► SenseVoice ─► [start-end] <lang> 전사
                                  └─ non-speech ─► PANNs      ─► [start-end] <tag> <tag>
               index.audio  ·  audio_index.txt

텍스트 인덱스   video ─► 1fps 후보·4s 재확인 ─► 기본 OCR
                     ─► 변하는 글자 영역만 4fps 스캔·추가 OCR ─► 줄 추적
                     ─► 프레임 내 문단 ─► 문단 추적·중복 정리
               index.text  ·  text_index.txt  ·  text_frames/observations.jsonl

몽타주         컷 검출 ─► 30s 넘는 공백만 채움 ─► 안정된 프레임으로 정착
                     ─► dedupe ─► 타임스탬프 굽기 ─► 3×3 montages
               index.visual
```

두 인덱스는 단위도 줄 형식도 실패 양상도 다릅니다. 한 파일에 이어 붙이면 화면 텍스트가
전사의 연장처럼 읽히므로, 저장도 렌더링도 프롬프트 블록도 각각 분리합니다.

### 오디오: VAD가 먼저 타임라인을 나눈다

VAD는 더 이상 SenseVoice 내부에 숨어 있지 않고 독립 단계로 먼저 실행됩니다. 그 결과가
인덱스의 뼈대입니다. 모든 초는 speech 구간이거나 non-speech 구간이며, 겹침도 빈 곳도
없습니다. speech 구간은 SenseVoice가 전사하고, non-speech 구간은 event tagger가 이름을
붙입니다. 침묵은 정보의 부재가 아니라 관찰 결과입니다.

구간 하나가 줄 하나입니다. 어느 모델이 그 줄을 만들었는지는 `index.json`에만 남고
모델에게는 보이지 않습니다. 에이전트는 무엇이 들렸는지를 받지, 누가 들었는지를 받지
않습니다.

VAD 후처리는 네 가지만 합니다.

- 검출 구간을 초 단위로 **바깥쪽으로** 반올림합니다. 시작은 내림, 끝은 올림이므로 구간은
  넓어지기만 하고 음절이 잘려나가지 않습니다.
- 짧은 침묵을 사이에 둔 이웃을 합칩니다. 다만 `merge_gap_s`를 넘지 않고, **원래 검출
  길이** 중 짧은 쪽보다도 짧은 간격만 메웁니다. 0.5초짜리 검출 둘을 1.4초 간격으로
  이으면 기침 두 번이 3초짜리 발화가 되기 때문입니다.
- `min_region_s`(기본 2초)보다 짧게 남은 침묵은 구간이 되지 않고 옆의 speech에
  흡수됩니다. 1초짜리 방 소음에 대해 tagger가 할 말은 없고, 그걸 한 줄로 내보내면
  화자가 숨 쉴 때마다 빈 줄이 하나씩 생깁니다. 버리면 타임라인에 구멍이 나므로 버리지
  않고 옆에 붙입니다. 실측(603초 요리 영상)으로 93줄 → 70줄입니다.
- 어느 구간도 `window_max_s`를 넘지 않게 자릅니다.

### 시각: 시계가 아니라 변화가 프레임을 고른다

고정 15초 샘플링은 4분째 바뀌지 않은 슬라이드에 프레임을 쓰면서 3초짜리 자막 카드는
놓쳤습니다. 이제 ffmpeg scene score로 컷을 먼저 찾고 그 직후를 샘플링합니다.
`max_interval_s`(기본 30초)는 샘플링 시계가 아니라 **허용 최대 공백**입니다. 컷을 모두
잡은 뒤 남은 공백 중 그보다 넓은 곳만 균등 분할해 채우므로, 컷이 촘촘한 영상은 채움이
거의 필요 없고 정지된 talking head도 30초를 넘겨 비지 않습니다. 프레임 예산이 모자라면
채움이 아니라 약한 컷부터 버리며, 그러고 나서 다시 공백을 채웁니다.

컷 직후 0.25초는 아직 디졸브인 경우가 많아, 연속한 두 프레임이 같아 보일 때까지
최대 `settle_max_s`만큼 앞으로 밀며 찾습니다. 페이드에 걸린 단색 프레임은 후보에서
빼고, 끝까지 안정되지 않으면 첫 번째 쓸 만한 프레임을 씁니다 — 커버리지가 우선입니다.

중복 제거는 직전에 남긴 프레임과 구분되지 않는 프레임을 버리되, 컷이거나 버렸을 때
30초보다 넓은 공백이 생기는 프레임은 남깁니다.

음성 인덱스의 segment는 VAD 분할 전체를 그대로 담습니다. 아무것도 이름 붙일 수 없었던
구간까지 포함하는데, 이게 나중에 ASR 도구가 요청 구간을 speech/non-speech로 나누는
근거이기 때문입니다. 다만 렌더링할 때는 빠집니다 — 전사도 태그도 없는 줄은 정보가 없고,
앞뒤 줄의 타임스탬프가 이미 그 공백을 보여줍니다.

### 태그: PANNs의 분류체계가 아니라 이름

PANNs는 AudioSet의 527개 라벨로 말합니다. 그건 어휘가 아니라 온톨로지라서 추상 상위
노드, 같은 소리의 여러 철자, 괄호 속 한정어가 섞여 있습니다. 그대로 내보내면
`<chopping_(food)> <water_tap,_faucet>` 같은 줄이 나옵니다. 출력 전에 정리합니다.

- 괄호와 첫 쉼표 뒤는 온톨로지가 자기를 설명하는 말이지 정보가 아닙니다.
  `Chopping (food)` → `chopping`, `Chewing, mastication` → `chewing`.
- 한 소리에 이름 하나. `Water tap, faucet`와 `Sink (filling or washing)`는 둘 다
  `running_water`입니다.
- **범주는 소리가 아닙니다.** `Animal`은 Dog와 Cat의 상위 노드입니다. 맞을 때조차
  자기 자식들보다 말하는 게 적고 — 개와 고래와 귀뚜라미에 동시에 들어맞습니다 —
  도마질 5초에 0.405로 떴습니다. 같은 구간에서 동물 계열 자식 라벨은 전부 0.14
  아래였고 영상에 동물은 없었습니다. 상위 노드와 음향 장면 라벨(`Inside, small room`,
  `Reverberation`)은 버립니다. 그 아래 잎 라벨(`Cat`, `Meow`)은 그대로 오므로,
  점수가 난 것을 잃지는 않습니다.
- 별칭이 없는 라벨은 버리지 않고 정리된 이름으로 내보냅니다. 어휘에 없다고 실제로 들린
  소리가 사라지지는 않습니다.

그리고 라벨은 tagger 자신의 최고 점수의 `relative_floor`(기본 0.4) 이상이어야 합니다.
이건 모델 내부에 대한 설명이 아니라 측정에 근거한 heuristic입니다. BGM 6초 구간이
`Music=0.834` 뒤에 `Animal=0.316`, `Snake=0.247`, `Squish=0.221`을 달고 나왔는데
영상에 그런 건 없었고, 반대로 서로 맞장구치는 라벨들은 최고 점수 가까이 붙어 있어
그대로 살아남습니다(한 입에 `Biting=0.549`, `Crunch=0.389`, `Chewing=0.254`).
절대 기준만으로는 앞의 것을 거르면서 `Chopping=0.321`을 살릴 수 없습니다 — 둘의
점수가 같은 자리에 있기 때문입니다.

### 화면 텍스트: 관측 → 추적 → 출력

전체 화면 OCR은 기존처럼 1초 간격 후보 중 변화가 있거나 4초 재확인 주기가 된
시점을 읽습니다. 몽타주 시점과 컷 +0.75초에 가까운 후보도 포함합니다.
추가 확인용 448px 스캔은 4fps이며, 전체 화면과 같은 샘플 중심에 맞춰 기존 기본
프레임을 다른 순간의 이미지로 바꾸지 않습니다.

추가 탐색은 **가까운 두 기본 관측에서 같은 자리의 문구가 바뀐 영역 안**에서만
합니다. 두 관측 사이가 4초를 넘으면 연결하지 않습니다. 같은 로고의 반복 판독이나
비슷한 오독만으로는 영역을 열지 않으며, 화면의 특정 위치·영상 종류·언어를 가정하지
않습니다. 그 영역의 이미지가 바뀌었을 때만 최대 2fps로 주변을 잘라 OCR합니다.
겹친 영역은 한 번만 읽습니다. 잘린 문장의 경계 판독과 원래 줄의 위치·글자 높이에
맞지 않는 배경 판독은 버리고, 박스·글자 높이는 전체 화면 좌표로 돌려놓습니다.
작은 crop을 OCR 엔진이 거대하게 확대하지 않도록 추가 확인에서만 확대를 끕니다.
검출·인식 모델의 ONNX 세션은 재사용합니다.

이 변하는 영역에서 근거가 부족한 정상 크기의 3글자 이상 후보는 앞뒤 0.5초 안의
최대 네 스캔에서 해당
영역을 추가 확인합니다. 같은 글자·위치·크기가 두 시점에서 확인되고 한 판독의
신뢰도가 0.95 이상이면 인정합니다. 모두 그 아래라면 기본 신뢰도 0.7을 넘는 동일
판독이 세 시점에서 일치해야 합니다. 숫자가 다른 판독이나 다른 위치의 같은 글자는
근거가 되지 않습니다. 움직이는 글자는 가까운 시점의 실제 박스와 비교합니다.
검증 전용 프레임의 다른 판독은 새 행을 만들지 않고, crop 밖의 글자를 사라졌다고
판단하지도 않습니다. 처음 나타난 다른 위치의 짧은 글자는 여전히 놓칠 수 있습니다.
모든 입력은 타임스탬프를 굽기 전 이미지입니다.

검출은 `PP-OCRv6_tiny_det`, 인식은 다음 모델을 사용합니다.

| 영상 언어 | 인식 모델 |
|---|---|
| `ko`, `ko-KR` | `korean_PP-OCRv5_mobile_rec` — 한글·영어·숫자 |
| `en`, `zh`, `ja` 및 지역 태그 | `PP-OCRv6_small_rec` — 영어·중국어 간체/번체·일본어·라틴 문자 |
| 언어 없음 / 매핑되지 않은 언어 | 같은 검출 crop에서 v6 small과 Korean v5 비교 |

언어는 `<video>.meta.json`에서 읽습니다. 화면에 여러 언어가 섞이면 해당
`rec_by_language` 매핑을 제거하여 기본 두 인식기를 사용할 수 있습니다. 오디오 전사로
화면 글자를 정정하지 않습니다. `scripts/fetch_ocr_models.py`가 검출기, 두 인식기와
각 모델에 맞는 전처리 설정·문자 사전을 받습니다.

`use_angle_cls: false`가 기본입니다. RapidOCR 휠의 중국어 방향 분류기가 정상적인
한국어 자막을 180도 뒤집는 경우를 확인했습니다. 이 상태에서는 해상도를 높여도
오히려 인식이 나빠집니다. 회전된 입력임을 아는 경우에만 켤 수 있습니다.

후처리는 언어·영상 이름·특정 문구에 의존하지 않습니다.

- 실제 박스 겹침으로 일대일 추적합니다. 3×3 칸 경계를 넘거나 같은 문구가 이동해도
  바로 새 구간이 되지 않습니다. 서로 다른 곳에 동시에 있는 같은 표기는 보존합니다.
- 띄어쓰기·유니코드를 정규화하고, 한글은 자모 유사도도 비교합니다. 숫자 변화는
  퍼지 병합하지 않습니다. 확실히 다른 문장은 분리하고, crop의 시각적 일치도도 씁니다.
- 대표 표기는 **실제로 읽은 표기** 중 가장 높은 신뢰도를 우선합니다. 반복 횟수는
  동점일 때만 쓰며, 반복해서 관측한 완전한 행을 짧게 잘린 판독으로 대체하지 않습니다.
  한 번 나온 긴 오독이 반복된 정상 판독을 밀어내지는 못합니다. 사전 교정이나 문장 생성은 없습니다.
- 먼저 각 줄의 시간적 정체성을 추적하고 크기·관측 근거를 판정합니다. 같은 프레임에서
  인접하고 관측 시점도 충분히 겹치는 줄끼리만 문단을 만듭니다. 다른 수명의 이웃,
  떨어진 문단, 별도 열은 분리합니다. 실제 프레임의 문단을 다시 추적하므로 서로 다른
  시점의 줄을 모아 가상의 화면을 만들지 않습니다. 자막/슬라이드/간판 분류는 하지 않습니다.
- 잠시 놓친 글자는 4초까지 재연결합니다. 가림으로 짧아진 판독이 계속되면 완전한 문장의
  구간을 가림이 시작된 시점에서 끝냅니다. 겹치는 문단과 구성 줄은 같은 시간·공간에
  중복 출력하지 않습니다. 3×3 위치와 크기 라벨은 출력용 부가 정보이며 병합 경계가 아닙니다.
- 기본 신뢰도는 0.7입니다. 같은 문구·수평 위치·폭의 관측을 모아 **중앙 글자 높이가
  화면의 2.5% 이하면 줄 전체를 제외**합니다. 720p에서 약 18px입니다. 높이는 기울어진
  검출 사각형의 짧은 변으로 측정하여, 기울기나 여러 줄을 감싼 박스 때문에 커지지 않습니다.
  블록을 만들기 전에 판정하므로 작은 줄과 큰 줄의 높이가 섞이지 않습니다. 정상 크기 행
  바로 아래의 정렬된 줄바꿈은 같은 문단으로 보존합니다.
- 같은 판독이 두 시점 이상에서 확인되어야 합니다. 한 번만 읽힌 글자는 신뢰도 0.95 이상이고,
  앞뒤 4초 안의 다른 두 시점에서도 같은 자리·크기의 텍스트가 확인될 때만 남깁니다.
  같은 문단에 반복 확인된 이웃 줄이 두 개 이상 있는 경우도 근거로 인정하여,
  슬라이드나 재료 목록의 한 줄을 한 번 놓쳤다고 내용에 구멍을 내지 않습니다.
  빠르게 바뀌는 자막은 이 근거나 위의 주변 프레임 검증을 사용할 수 있습니다.
  언어별 짧은 글자 예외는 없으며,
  고립된 한 프레임짜리 진짜 글자도 누락될 수 있습니다. 정갈함을 우선한 선택입니다.
  제외된 결과까지 `text_frames/observations.jsonl`에 박스·글자 높이·신뢰도를 기록합니다.
  추가 확인 프레임에는 `verification: true`를 기록하여 리플레이에서도 용도를 유지합니다.
  확인된 영역을 추가 탐색한 프레임은 `discovery: true`이며, `regions`에 읽은 범위를 기록합니다. 이 프레임만으로는 짧은
  글자나 텍스트 자리를 확정하지 않으며, 3글자 이상이고 위의 직접 재확인이 있어야
  새로운 텍스트로 출력합니다. 기본 탐색·필수 시점·정기 확인의 관측과 구분합니다.
  초 미만 구간이 `[1-1]`처럼 표시되지 않도록 출력 시작은 내림, 끝은 올림합니다.

모델을 다시 실행하지 않고 후처리만 검증하려면 별도의 새 디렉터리에 재생합니다.
입력 관측과 기존 인덱스를 수정하지 않으며 결과·입력 해시·처리 시간을 기록합니다.

```bash
python scripts/replay_ocr.py /path/to/text_frames/observations.jsonl \
  --duration 716 --output-dir /tmp/ocr-replay
```

사람과 에이전트가 읽는 파일에는 시간·3×3 위치·대략적 크기를 영어로 표시합니다.
크기는 대표 줄 높이 기준 `small`(<4%), `medium`(4~8%), `large`(≥8%)입니다. 정밀 박스는 관측 파일에
남습니다. 두 번 이상 반복한 동일 문구는 **실제 구간을 모두 보존하여** 한 줄로 모읍니다.
위치·크기가 다르면 각 구간 옆에 따로 표시합니다. 사이의 공백을 연속 노출로 바꾸지 않습니다. 작은 UI나 표의 세부 값이
질문의 핵심이면 에이전트가 `view_frames`로 확인하도록 프롬프트에 명시합니다.

```text
[0-8, top center, medium] What are they used for?
[12-17, bottom center, small] 東京にある日本語学校で日本語を教える仕事です
[18-21; 38-41; 60-64, top left, small] Channel name
```

### 자막: 무엇인지 먼저 판정하고, 그 다음에 쓴다

채널이 올린 자막 트랙은 네 번째 전사기가 아닙니다. 영상과 함께 도착한 출처 불명의
텍스트이고, **음성의 전사인지 화면 글자의 사본인지 먼저 판정**합니다. 우선순위 사다리를
쓰지 않는 이유는 등급표 자체에 있습니다 — 기대 정확도가 가장 높은 소스가 기계적 검증
가능성이 가장 낮습니다. 사다리는 곧 가장 검증되지 않은 텍스트가 기본으로 이긴다는 뜻입니다.

유튜브는 사람이 자막을 올린 언어에 대해 그 언어의 ASR 트랙을 노출하지 않습니다. eval
5편에서 `automatic_captions[lang]`은 사람 트랙의 포맷 별칭이었고, `&nbsp;`를 정제하면
**단어 유사도 1.0000, 불일치 0**이었습니다. 자막 슬롯에는 실제 소스가 최대 하나뿐이며,
두 트랙이 일치한다는 사실은 어떤 근거도 되지 않습니다. 그래서 `--write-auto-sub` 같은
편의 플래그를 쓰지 않고 info JSON의 자막 딕셔너리에서 URL을 직접 읽습니다.

판정은 세 축입니다. 트랙 텍스트가 선언된 문자 체계로 쓰였는가(L), 단어가 VAD speech 안에
떨어지는가(V), 같은 시각 OCR이 읽은 것과 일치하는가(O).

| 조건 | 귀속 | 실측 |
|---|---|---|
| L 실패 | 버림 — 번역·오태깅 | 한글 비율 0.0000 |
| V ≥ 0.6 | **음성 전사** | talking-head 0.998, cooking 0.917, slide 0.999 |
| V < 0.6, O ≥ 0.7 | **하드섭 사본** | korean-vlog V 0.011 / O 0.894 |
| 그 외 | 버림 — 싱크 깨짐·홍보 | 미관측 |

**V축이 먼저 결정합니다.** 슬라이드를 낭독하는 강연은 O가 0.501까지 올라가지만, 그 트랙은
음성의 전사이지 슬라이드의 사본이 아닙니다. 그것으로 화면 글자를 고치면 화자가 풀어 말한
문장으로 슬라이드를 덮어쓰게 됩니다. 번역 트랙은 음성에 맞춰 타이밍이 잡혀 있어 V를
통과하므로, 원본 언어 트랙이 아니면 애초에 받지 않습니다.

음성 인덱스 융합은 VAD region 단위입니다. 각 토큰은 시간 중심점이 속한 region 하나에만
배정되므로 중복도 누락도 구조적으로 불가능합니다. **자막이 닿는 region은 자막이, 닿지 않는
region은 ASR이 차지합니다.** 처음에는 opcode로 병합하며 자막이 생략한 토큰을 ASR에서
되살렸는데, Whisper 준거 측정에서 그게 인덱스를 그냥 두는 것보다 나빴습니다(talking-head
6.25% 대 5.93%). 자막이 생략하는 것은 읽기 속도에 맞춘 축약이 아니라 대부분 필러와
오인식이었습니다. 버리도록 바꾸자 같은 영상이 17% 개선, 슬라이드 강의가 41% 개선이
됐습니다. 그렇게 하면 융합된 72개 region 전부에서 출력이 자막과 글자 단위로 같아지므로,
정렬은 병합이 아니라 **가드로만** 씁니다 — 두 진술이 서로 닮지 않은 region은 어느 쪽도
말하지 않은 문장으로 이어붙이지 않고 그대로 둡니다.

텍스트 인덱스 융합은 하드섭 사본으로 판정된 트랙만 합니다. 이것은 오디오 전사가 화면
글자를 고치는 것이 아닙니다 — 그 트랙은 화면 글자 자신의 원본이고, 픽셀 판독보다 상위
근거입니다. OCR이 `영양 균형 행기기`, `듬뿐`, `오본`, `시간X`로 읽은 줄을 트랙은 바르게
갖고 있습니다. 교체 후보의 길이가 원본의 0.8~1.25배를 벗어나면 거부합니다. 교정은 오독을
고치는 것이지 이웃 행을 흡수하거나 이 행의 절반을 버리는 것이 아닙니다.

OCR만 본 줄은 남깁니다 — 그릇의 `pyrex`는 실제로 화면에 있습니다. 자막에만 있고 OCR이
보지 못한 줄은 **추가하지 않습니다.** 위치도 크기도 없고, 텍스트 인덱스는 OCR이 그 자리에서
실제로 본 것을 뜻하기 때문입니다.

출력 형식은 바뀌지 않습니다. 출처도 등급도 신뢰도 표시도 인덱스에 나타나지 않으며,
판정 결과와 융합 통계는 `index.json`과 trace에만 남습니다. 에이전트는 여전히 무엇이
들렸는지를 받지, 누가 들었는지를 받지 않습니다.

```bash
python scripts/fetch_captions.py --manifest eval/youtube-field-eval.yaml
```

알려진 한계: **VAD가 놓친 발화는 자막이 있어도 복구되지 않습니다.** VAD 분할은 인덱스
전체의 척추이고 이번 변경은 그것을 움직이지 않으므로, speech region 밖으로 떨어진 자막
토큰은 버려집니다. 원본 언어 트랙이 없으면(일본어 브이로그는 번역 15개만 있습니다) 자막에서
아무것도 얻지 않습니다.

### 저하는 하되 실패하지 않는다

event tagger가 없으면 SenseVoice의 태그로 물러서고, OCR 엔진이 없거나 읽기 중 실패하면 화면 텍스트만
잃습니다. 자막 사이드카가 없거나 깨져 있으면 교정만 잃습니다. 어느 쪽도 인덱스 전체를 무너뜨리지 않으며 그 사실은 trace에 남습니다.

OCR은 별도 프로세스에서 돌립니다. macOS에서 onnxruntime과 torch를 한 인터프리터에서
내리면 종료 시점에 SIGABRT가 납니다. 인덱스는 이미 저장된 뒤였지만 `vuc index`의 종료
코드는 134였습니다.

### 이번 범위 밖

visual chapter 추출, 음악 분석, 설명·카드 등 나머지 메타데이터는 포함하지 않습니다.
메타데이터는 채널명·제목·길이·챕터만 씁니다. 자막은 원본 언어 트랙 하나만 쓰며, 번역
트랙과 여러 트랙의 동시 사용, 자막에 의한 VAD 분할 수정, 에이전트가 호출한 Whisper 결과를
인덱스에 누적하는 레이어는 포함하지 않습니다.

알려진 한계: 작은 간판·장식 글꼴의 오독은 높은 confidence에서도 남을 수 있습니다.
0.25초 스캔 사이에 나타났다 사라지는 글자는 놓칠 수 있고, 단독·작은 관측의 제외는 정밀도와
재현율 사이의 선택입니다. 전체 영상의 정답 전사가 없으므로 cue 수를 정확도나 recall로
해석하지 않습니다. 원본 관측은 이 기준을 바꿔 재처리할 수 있도록 보존합니다.

## Tools

agentic 모드에는 두 도구만 노출합니다.

```text
transcribe_segment(start_s, end_s)
view_frames(start_s, end_s, fps, n)
```

- `transcribe_segment`는 요청 구간을 인덱스와 **같은 VAD 분할**로 나눠서 처리합니다. speech
  부분은 faster-whisper가 다시 전사하고, non-speech 부분은 event tagger가 태그를 답니다.
  에이전트는 어느 쪽이 어느 모델에서 왔는지 알 필요 없이 인덱스와 동일한 `[start-end]`
  형식의 더 정밀한 결과를 하나로 받습니다. 분할을 여기서 다시 계산하지 않고 인덱스의
  것을 그대로 쓰므로, 에이전트가 인용하는 타임라인과 어긋날 일이 없습니다.
- 한 번 호출 구간은 `tools.transcribe_segment.max_duration_s`와 provider 상한 중 작은
  값까지이며 기본값은 60초입니다. 기본 provider는 로컬 faster-whisper `large-v3-turbo`,
  CPU `int8`입니다. cloud provider는 인터페이스만 있는 stub입니다.
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
uv sync --extra dev --extra sensevoice --extra advanced-asr --extra ocr --extra audio-events
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
| caption VAD overlap floor | 0.6 |
| caption OCR match floor | 0.7 |
| caption alignment guard | 0.2 |
| caption correction scope | 0.8–1.25× |
| index frame sampling | cuts + 30s coverage grid |
| baseline_full frame sampling | 1s |
| VAD region limit | 30s |
| VAD merge gap | 1.5s, capped by the shorter detection |
| VAD minimum region | 2s |
| event tagger | PANNs top-4, ≥0.2 and ≥0.4× its own top score |
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

캐시는 입력 파일 SHA-256을 키로 `.vuc-cache/<sha256>/` 아래에 저장됩니다. 인덱스를 쓰는
모드는 `audio.wav`, `index.json`, `audio_index.txt`, `text_index.txt`, 프레임, 몽타주를
공유합니다. 두 인덱스 파일은 비어 있더라도 항상 씁니다. `text_index.txt`가 비었다는 것은
프레임을 읽었고 글자가 없었다는 뜻이고, 파일이 없다면 OCR이 아예 돌지 않은 것과
구분되지 않습니다. 인덱스 출력을
바꾸는 설정이 하나라도 달라지면 캐시를 재사용하지 않고 다시 만듭니다. 각 `vuc run`의
보고서와 trace는 `runs/<run-id>/report.md`, `report.json`, `trace.jsonl`로 분리되어 이전 실행을
덮어쓰거나 서로 섞지 않습니다.

## Measurements

### Local model calibration

2026-09-13에 arm64 Mac14,10(12코어, RAM 16GB)에서 `.vuc-cache`와 분리된 임시
작업 공간으로 모델과 스레드 수를 다시 측정했습니다.

- FLEURS 한·영·중·일 각 6개, 총 262.7초에서 SenseVoiceSmall은 CER 5.90%, 처리
  36.1초였습니다. faster-whisper `large-v3-turbo`는 beam 1이 CER 4.06%/109.0초,
  beam 5가 3.84%/115.4초였습니다. 전체 인덱스는 빠른 SenseVoice, 요청 구간은 더
  정확한 Whisper beam 5로 처리하는 현재 구성을 유지합니다.
- SenseVoiceSmall의 같은 43.9초 입력은 2/4/6/8/12 CPU thread에서 각각
  2.54/3.18/4.75/6.13/8.30초였습니다. OCR과 병렬인 완전 인덱싱에서도 2 thread가
  구간당 약 1.5초, 8 thread가 4~5초였고 두 실행의 audio/text index는 byte 단위로
  같았습니다. 그래서 indexer 기본값은 2입니다.
- FSMN-VAD는 1,027초 입력에서 1/2/4/8/12 thread가
  5.70/3.67/2.72/2.44/2.63초였고 검출 결과가 같아 8 thread를 유지합니다.
  Whisper도 8 thread가 가장 빨라 그대로 둡니다.
- VAD가 고른 실제 non-speech 48창(257.1초)에서 PANNs Cnn14와 EfficientAT
  `dymn10_as`를 비교했습니다. EfficientAT은 더 작은 최신 후보지만 CPU 처리 시간이
  7.06초로 PANNs의 2.77초보다 길었고, 현재 필터 뒤 유효 태그도 43개 대 63개였습니다.
  `sizzle`, `frying`, `train`처럼 영상에서 유용한 세부 라벨을 더 자주 놓쳐 PANNs와
  기존 threshold를 유지합니다. 이 비교에는 완전한 event 정답 라벨이 없으므로 모델
  자체의 일반 정확도 순위로 해석하지 않습니다.

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

아래 "Benchmark results"의 수치는 고정 15초 샘플링과 SenseVoice 내부 VAD를 쓰던 이전
인덱스에서 측정한 것으로, 현재 파이프라인의 것이 아닙니다.
