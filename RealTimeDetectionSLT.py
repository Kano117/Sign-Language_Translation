import cv2
import pickle
import torch
import mmcv
import yaml
import time
import numpy as np
from pathlib import Path
from mmpose.apis.inference import inference_topdown, init_model
from model import SignLanguageModel
from Tokenizer import GlossTokenizer_G2T, GlossTokenizer_S2G, TextTokenizer
from datasets import S2T_Dataset

def extract_keypoints_from_video(video_path, output_test_path):
    """
    Trích xuất keypoints từ video sử dụng HRNet model
    """
    # Load HRNet model
    config_file = './Khaosattongthe/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py'
    checkpoint_file = './Khaosattongthe/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hrnet_model = init_model(config_file, checkpoint_file, device=device)

    video = mmcv.VideoReader(video_path)
    keypoints = []

    for frame in video:
        results = inference_topdown(hrnet_model, frame)
        if results and results[0].pred_instances.keypoints.shape[0] > 0:
            kp_xy = results[0].pred_instances.keypoints[0]
            scores = results[0].pred_instances.keypoint_scores[0]
            kp_combined = np.concatenate([kp_xy, scores[..., None]], axis=-1)
            keypoints.append(torch.tensor(kp_combined, dtype=torch.float32))
        else:
            keypoints.append(torch.zeros((133, 3), dtype=torch.float32))

    name = Path(video_path).stem
    sample_dict = {
        'keypoint': torch.stack(keypoints),
        'gloss': 'NULL',
        'text': '',
        'num_frames': len(keypoints),
        'name': name
    }

    data_dict = {f'test/{name}': sample_dict}
    with open(output_test_path, 'wb') as f:
        pickle.dump(data_dict, f)
    return name

def update_config_test_path(config_path, new_test_path):
    """
    Cập nhật đường dẫn test file trong config
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['data']['test_label_path'] = new_test_path
    return config

def run_slt_model_on_test(config, test_index=0):
    """
    Chạy model SLT (Sign Language Translation) để nhận diện camera input
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Khởi tạo Tokenizers
    gloss_cfg = config['model']['RecognitionNetwork']['GlossTokenizer']
    gloss_tokenizer = GlossTokenizer_S2G(gloss_cfg)
    
    text_cfg = config['model']['TranslationNetwork']['TextTokenizer']
    text_tokenizer = TextTokenizer(text_cfg)

    # Khởi tạo Dataset
    test_data = S2T_Dataset(
        path=config['data']['test_label_path'],
        tokenizer=gloss_tokenizer,
        config=config,
        args=None,      
        phase='test'
    )

    sample = test_data[test_index]
    sample_input = test_data.collate_fn([sample])

    # Khởi tạo mô hình SLT 
    model = SignLanguageModel(cfg=config, args=None)
    
    checkpoint_path = './Khaosattongthe/pretrained_models/best.pth'
    if not Path(checkpoint_path).exists():
        print(f"Lỗi: Không tìm thấy file trọng số tại {checkpoint_path}")
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False) 
    model.to(device)
    model.eval()

    with torch.no_grad():
        for key in sample_input:
            if isinstance(sample_input[key], torch.Tensor):
                sample_input[key] = sample_input[key].to(device)

        # Định dạng lại input cho nhánh Translation của MSKA
        if 'keypoint' in sample_input:
            sample_input['video'] = sample_input['keypoint']
            # MSKA cần 'translation_inputs' để dịch câu
            # Giả lập cấu hình rỗng để model không báo lỗi thiếu key
            if 'translation_inputs' not in sample_input:
                sample_input['translation_inputs'] = {
                    'labels': torch.zeros((1, 1), dtype=torch.long, device=device) 
                }

        # CHẠY MÔ HÌNH
        output = model(sample_input)
        
        # Giải mã kết quả Nhận diện (Glosses)
        gloss_logits = output['ensemble_last_gloss_logits']
        input_lengths = output['input_lengths']
        decoded_ids = model.recognition_network.decode(gloss_logits, beam_size=5, input_lengths=input_lengths)
        gloss_prediction = gloss_tokenizer.convert_ids_to_tokens(decoded_ids)[0]

        # Giải mã kết quả Dịch thuật (Text) - Dùng hàm generate_txt từ model.py
        if 'transformer_inputs' in output:
            gen_out = model.generate_txt(transformer_inputs=output['transformer_inputs'])
            text_prediction = gen_out['decoded_sequences']
        else:
            text_prediction = ["Dữ liệu không đủ để dịch thành câu"]

        return gloss_prediction, text_prediction

def display_results(gloss_pred, text_pred, frame_index):
    """
    Hiển thị kết quả nhận diện lên console và file
    """
    glosses_text = ' '.join(gloss_pred) if gloss_pred else "[Không có kết quả]"
    text_translation = text_pred[0] if text_pred and text_pred[0] else "[Không có kết quả]"
    
    print(f"Dịch hoàn chỉnh: {text_translation}\n")

    # Lưu kết quả vào file
    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(f"Từ khóa nhận diện (GLOSSES): {glosses_text}\n")
        f.write(f"Dịch hoàn chỉnh: {text_translation}")

# Loop Webcam - Ghi 4s - Nhận Diện SLT 
if __name__ == "__main__":
    config_path = './Khaosattongthe/configs/phoenix-2014t_s2t.yaml'
    test_file = 'temp_webcam.test'
    cam_index = 0
    
    # Cấu hình webcam
    VIDEO_DURATION = 4  # giây
    FPS = 15
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    # Mở camera
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    print("\n" + "-"*60)
    print("NHẬN DIỆN CAMERA INPUT CHO MODEL SLT")
    print("-"*60)
    print(f"Cấu hình:")
    print(f"  - Thời lượng ghi mỗi lần: {VIDEO_DURATION}s")
    print(f"  - FPS: {FPS}")
    print(f"  - Độ phân giải: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print("\nNhấn:")
    print("  - SPACE: Bắt đầu ghi video")
    print("  - Q: Thoát chương trình")
    print("-"*60 + "\n")

    frame_count = 0

    try:
        while True:
            # Chờ người dùng nhấn SPACE để bắt đầu ghi
            print("Đang chờ... Nhấn SPACE để bắt đầu ghi video (Q để thoát)")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                cv2.putText(frame, "Press SPACE to record or Q to quit", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Sign Language Translation - Real Time", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # SPACE
                    break
                elif key == ord('q'):  # Q
                    raise KeyboardInterrupt

            # Ghi video
            out_path = 'webcam_temp.mp4'
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 
                                 FPS, (FRAME_WIDTH, FRAME_HEIGHT))

            frame_count += 1
            print(f"\n  [Lần {frame_count}] Đang ghi video {VIDEO_DURATION}s...")
            
            start_time = time.time()
            frames_captured = 0
            
            while time.time() - start_time < VIDEO_DURATION:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                frames_captured += 1
                
                # Hiển thị countdown
                elapsed = time.time() - start_time
                remaining = VIDEO_DURATION - elapsed
                
                cv2.putText(frame, f"Recording... {remaining:.1f}s", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(frame, f"Frames: {frames_captured}", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Sign Language Translation - Real Time", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    out.release()
                    raise KeyboardInterrupt

            out.release()
            print(f"Hoàn thành ghi video ({frames_captured} frames)")

            # Xử lý keypoints từ video
            print("Đang trích xuất keypoints...")
            name = extract_keypoints_from_video(out_path, test_file)
            
            # Cập nhật config
            config = update_config_test_path(config_path, test_file)
            
            # Chạy model SLT
            print("Đang chạy model SLT...")
            gloss_pred, text_pred = run_slt_model_on_test(config)
            
            if gloss_pred is not None:
                # ➤ Hiển thị kết quả
                display_results(gloss_pred, text_pred, frame_count)
            else:
                print("Lỗi: Không thể chạy model")

    except KeyboardInterrupt:
        print("\n\nThoát chương trình")
    except Exception as e:
        print(f"\nLỗi: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
