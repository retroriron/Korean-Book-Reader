import cv2
import numpy as np
import easyocr
import time
import re
import threading
import queue
import os
from difflib import SequenceMatcher
from gtts import gTTS 
import pygame 
from openai import OpenAI

# ==========================================
# 1. 프로그램 기본 설정
# ==========================================
OPENAI_API_KEY = ""
client = OpenAI(api_key=OPENAI_API_KEY)

# 작업창끼리 데이터를 주고받는 큐
ocr_queue = queue.Queue(maxsize=1)
tts_queue = queue.Queue()

# 글자 읽는 엔진 및 오디오 재생기
reader = easyocr.Reader(['ko', 'en'])
pygame.mixer.init()

# 현재 상태 변수
current_state = "STANDBY"
is_muted_flag = False  # s 키로 소리를 끈 후, 새로운 줄을 조준하면 바로 읽어주기 위한 스위치


# ==========================================
# 2. 백그라운드 작업
# ==========================================

def tts_worker():
    """글자를 음성 파일로 만들고 스피커로 재생하는 작업"""
    global current_state
    temp_file = "tts_temp.mp3"
    
    while True:
        text = tts_queue.get()
        if text is None:  # 종료 신호 -> 중단
            break
        
        try:
            current_state = "SPEAKING"
            
            # 한국 전용 구글 주소로 연결하여 음성 파일 생성 속도 단축
            tts = gTTS(text=text, lang='ko', tld='co.kr', slow=False)
            tts.save(temp_file)
            
            # 이미 음성이 출력되는 경우
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # 소리 재생이 끝날 때까지 대기
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
                
            pygame.mixer.music.unload()
            
            # 사용이 끝난 임시 파일 삭제
            if os.path.exists(temp_file):
                try: os.remove(temp_file)
                except: pass
                
        except Exception as e:
            print(f"음성 재생 오류: {e}")
            
        # 소리가 다 나온 뒤 대기 상태로 변경
        if current_state == "SPEAKING":
            current_state = "STANDBY"
                
        tts_queue.task_done()


def ocr_worker():
    """책 이미지를 읽고 AI에게 오타 수리를 요청하는 작업"""
    global current_state, is_muted_flag
    last_text = ""
    last_ocr_time = 0
    
    while True:
        warped_cropped = ocr_queue.get()
        if warped_cropped is None:  # 종료 신호 -> 중단
            break
            
        current_time = time.time()
        # AI 호출 제한 (1.2초 간격)
        if current_time - last_ocr_time > 1.2:
            try:
                if current_state != "SPEAKING" and current_state != "MUTED":
                    current_state = "PROCESSING"
                
                # 글자를 잘 인식하도록 이미지를 흑백으로 바꾸고 4배 크게 키우기
                gray = cv2.cvtColor(warped_cropped, cv2.COLOR_BGR2GRAY)
                ocr_input = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                
                ocr_results = reader.readtext(ocr_input, detail=0)
                
                is_sent_to_tts = False
                if ocr_results:
                    raw_text = " ".join(ocr_results)
                    # 특수문자를 지우고 한글, 영어, 숫자, 기본 부호만 남기기
                    clean_text = re.sub(r'[^가-힣a-zA-Z0-9\s\.\,\?\!]', '', raw_text)
                    clean_text = " ".join(clean_text.split())

                    # 글자 수가 1자 이상일 때 전송
                    if len(clean_text) >= 1: 
                        response = client.chat.completions.create(
                            model="gpt-4o-mini", 
                            messages=[
                                {
                                    "role": "system", 
                                    "content": (
                                        "당신은 카메라로 촬영된 책 이미지에서 추출된 OCR 텍스트의 오타를 교정하는 전문가이다.\n"
                                        "입력되는 텍스트는 글자가 뭉개지거나 모양/발음이 유사한 단어로 잘못 인식된 상태이다. "
                                        "어떤 장르의 책이든 상관없이 아래의 규칙을 엄격히 적용하라:\n\n"
                                        "1. 문맥이 완전히 깨져 보일 때는, 입력된 단어의 초성/중성/종성 파편과 실제 발음을 소리 내어 읽었을 때 가장 가깝고 자연스러운 정상적인 한국어 단어로 복원하라.\n"
                                        "2. 오타 때문에 단어 자체로 해석이 안 된다면, 앞뒤 문장의 흐름(서술어, 조사 호응)을 보고 책에 인쇄되었을 법한 올바른 단어를 유추하라.\n"
                                        "3. 입력된 문장이 중간에 끊겨 있다면(예: '바라보'), 절대로 뒤에 말을 임의로 지어내서 문장을 완성하지 마라. 끊긴 상태 그대로 오타만 고쳐라.\n"
                                        "4. 인사말, 부연 설명, 마침표(.) 등은 모두 생략하고 오직 교정된 문장 딱 한 줄만 출력하라."
                                    )
                                },
                                {"role": "user", "content": f"이 문장의 오타를 문맥과 글자 형태 기반으로 올바르게 고쳐라: {clean_text}"}
                            ],
                            temperature=0.1
                        )
                        
                        fixed_text = response.choices[0].message.content.strip()
                        similarity = get_sentence_similarity(fixed_text, last_text)
                        
                        # 강제 정지를 눌렀었거나, 이전 문장과 내용이 다를 때 소리 대기열에 주입
                        if is_muted_flag or similarity < 0.5:
                            if is_muted_flag:
                                print(f"\n[🔄 새 조준 인식] 이전 문장을 버리고 새 문장을 읽습니다.")
                                is_muted_flag = False  # 새 문장 읽기 시작 -> 스위치 off
                            
                            print(f"[★ 새 문장 읽기] : {fixed_text}") 
                            current_state = "SPEAKING"
                            tts_queue.put(fixed_text)                
                            last_text = fixed_text
                            last_ocr_time = time.time()  
                            is_sent_to_tts = True
                
                if not is_sent_to_tts and current_state == "PROCESSING":
                    current_state = "STANDBY"
                    
            except Exception as e:
                print(f"글자 판독 오류: {e}")
                if current_state != "SPEAKING" and current_state != "MUTED":
                    current_state = "STANDBY"
                
        ocr_queue.task_done()


def get_sentence_similarity(str1, str2):
    """띄어쓰기를 지우고 두 문장이 얼마나 비슷한지 비율 계산"""
    if not str1 or not str2: return 0.0
    s1 = str1.replace(" ", "")
    s2 = str2.replace(" ", "")
    return SequenceMatcher(None, s1, s2).ratio()

# 백그라운드 작업 시작
threading.Thread(target=tts_worker, daemon=True).start()
threading.Thread(target=ocr_worker, daemon=True).start()


# ==========================================
# 3. 카메라 및 실시간 화면 설정
# ==========================================
camera_index = 0
cap = cv2.VideoCapture(camera_index)

def switch_camera():
    """r 키를 누르면 연결된 다른 카메라로 교체"""
    global cap, camera_index
    print("\n[카메라 전환 중...] 기존 카메라를 해제합니다.")
    cap.release()
    time.sleep(0.5) 
    for i in range(1, 5):
        next_idx = (camera_index + i) % 5
        test = cv2.VideoCapture(next_idx)
        if test.isOpened():
            get_ret, _ = test.read()
            if get_ret:
                camera_index = next_idx
                return test
        test.release()
        time.sleep(0.2)     
    return cv2.VideoCapture(0)

def order_points(pts):
    """검출한 사각형 모서리 4곳의 위치를 위아래 순서대로 정렬"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 카메라 화면 분석 반복 루프
while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue

    # 빨간색 테두리 영역만 찾아내기 위한 색상 분리
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1, upper1 = np.array([0, 50, 50]), np.array([25, 255, 255])
    lower2, upper2 = np.array([170, 50, 50]), np.array([180, 255, 255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

    # 화면 정돈
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        best_contour = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1500:
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect
                actual_w = max(w, h)
                actual_h = min(w, h)
                if actual_h == 0: continue
                aspect_ratio = actual_w / actual_h
                
                # 가로가 세로보다 4.5배 이상 긴 형태만 고르기
                if aspect_ratio > 4.5:
                    if area > max_area:
                        max_area = area
                        best_contour = cnt

        if best_contour is not None:
            box = np.int32(cv2.boxPoints(cv2.minAreaRect(best_contour)))
            pts = order_points(box.astype("float32"))
            (tl, tr, br, bl) = pts
            
            # 사각형 모양을 똑바로 펴기 위한 규격 계산
            maxWidth = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            maxHeight = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

            # 기울어진 사각형 영역 보정
            dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
            M = cv2.getPerspectiveTransform(pts, dst)
            warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
            
            # 잘라낸 글자 영역 가장자리에 빨간 레이저 선이 남지 않도록 테두리 살짝 깎아내기
            margin_y, margin_x = int(maxHeight * 0.12), int(maxWidth * 0.03)
            warped_cropped = warped[margin_y:-margin_y, margin_x:-margin_x] if maxHeight > margin_y * 2 and maxWidth > margin_x * 2 else warped

            # 최종 완성된 글자 줄 이미지를 글자 인식 대기열로 전달
            if warped_cropped is not None and warped_cropped.shape[1] > 0 and warped_cropped.shape[0] > 0:
                cv2.imshow("ROI_Cropped", warped_cropped)
                if ocr_queue.empty(): 
                    ocr_queue.put(warped_cropped)

    # ==========================================
    # 4. 화면 위 상태창 자막 표시
    # ==========================================
    if current_state == "PROCESSING":
        msg, color = "[ AI PROCESSING ] Correcting text...", (255, 150, 0)      
    elif current_state == "SPEAKING":
        msg, color = "[ SPEAKING NOW ] Reading... (Press 'S' to Mute)", (0, 220, 0) 
    elif current_state == "MUTED":
        msg, color = "[ AUDIO MUTED ] Aim at a NEW line to resume.", (0, 0, 255)  
    else:
        msg, color = "[ STANDBY MODE ] Aim at the target book line.", (160, 160, 160) 

    # 검은색 안내 바 위에 상태 출력
    cv2.rectangle(frame, (15, 15), (520, 55), (0, 0, 0), -1)
    cv2.putText(frame, msg, (25, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"CAM {camera_index}", (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)

    # 키보드 입력 대기
    key = cv2.waitKey(1) & 0xFF
    
    # s 키: 소리를 끄고 다음번에 조준한 줄을 읽도록 설정
    if key == ord('s'):
        if pygame.mixer.music.get_busy() or not tts_queue.empty():
            print("\n[🛑 문장 버림 / 중단] 현재 오디오를 종료하고 새 조준을 대기합니다.")
            current_state = "MUTED"
            is_muted_flag = True  # 스위치 켜기
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            
            # 대기 중이던 오디오 목록 전체 삭제
            while not tts_queue.empty():
                try:
                    tts_queue.get_nowait()
                    tts_queue.task_done()
                except queue.Empty:
                    break

    elif key == ord('r'):      
        cap = switch_camera()
    elif key == ord('q'):    
        break

# 프로그램 종료
ocr_queue.put(None)
tts_queue.put(None)
cap.release()
cv2.destroyAllWindows()