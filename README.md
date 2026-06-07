# Korean-Book-Reader
<img width="600" alt="20260607121817" src="https://github.com/user-attachments/assets/3b415b4f-3ab5-43f3-bd8b-6d2ae21c55d1" />

**책의 특정 줄을 조준하면 실시간 이미지 보정, OCR 글자 추출, AI 오타 수리를 거쳐 한국어 음성으로 읽어주는 독서 보조 프로그램.**

## 1. 주요 기능 (Key Features)
* **실시간 이미지 평탄화 보정:** 비스듬하게 촬영되거나 기울어진 책 줄을 정면에서 본 것처럼 바르게 펴서 인식률을 높임
* **AI 기반 문맥 오타 교정:** 글자가 흐릿하게 인식되어 발생한 오타를 앞뒤 흐름을 분석하는 AI(GPT)를 통해 정상적인 문장으로 자동 수정
* **즉각적인 음성 제어(인터럽트):** `S` 키를 눌러 소리를 중단한 후, 다른 줄을 조준하면 지연 시간이나 매칭 대기 없이 새로운 문장을 즉시 재생
* **목소리 출력 속도 가속:** 음성 파일 생성 서버를 한국 지역 엔드포인트로 고정하여 끊김 현상을 방지하고 데이터 처리 속도를 향상

  
## 2. 시스템 구조 (Architecture)
* **메인 (Main Thread):** 실시간 카메라 프레임 수집, 레이저 가이드라인 기반의 영역 검출, 투시 변환 왜곡 보정 및 화면 상태 자막 렌더링 수행
* **텍스트 분석 및 교정 (OCR Worker Thread):** 관심 영역 이미지 크롭, EasyOCR 기반 텍스트 추출, OpenAI API를 활용한 오타 교정 및 문장 유사도 비교 필터링 처리
* **음성 출력 제어 (TTS Worker Thread):** gTTS를 이용한 한국어 오디오 파일 생성 및 Pygame Mixer를 통한 독립적인 오디오 장치 제어 및 재생 수행

## 3. 시작하기 (Getting Started)

### 필수 패키지 설치
프로그램 실행을 위해 아래 라이브러리를 설치해야 합니다.
```bash
pip install opencv-python numpy easyocr gTTS pygame openai
```

### API 키 설정
코드 내부의 OPENAI_API_KEY 변수에 본인의 OpenAI API 발급 키를 입력합니다.

## 4. 사용 방법 (Usage)
프로그램을 실행한 뒤, 책의 읽고자 하는 줄에 가이드라인(빨간색 프레임)을 조준합니다.

* **S 키**: 현재 읽고 있는 음성을 즉시 중단하고 대기열을 비웁니다. 이후 새로운 줄을 조준하면 즉시 새로 읽기 시작합니다.

* **R 키**: 연결된 다른 카메라 장치로 화면을 전환합니다.

* **Q 키**: 프로그램을 안전하게 종료합니다.

## 5. 개발 환경
**Language**: Python

**Computer Vision**: OpenCV

**OCR Engine**: EasyOCR

**AI Engine**: OpenAI API (gpt-4o-mini)

**Audio Engine**: gTTS, Pygame (mixer)

## 6. 작동 예시


https://github.com/user-attachments/assets/97d273fe-25b3-439a-9eee-28c2d011fa67

