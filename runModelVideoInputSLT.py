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
    print(f"Đã tạo file .test tại: {output_test_path}")
    return name

def update_config_test_path(config_path, new_test_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['data']['test_label_path'] = new_test_path
    return config

def run_model_on_test(config, test_index=0):
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
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False) 
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Chuyển dữ liệu sang GPU/CPU
        for key in sample_input:
            if isinstance(sample_input[key], torch.Tensor):
                sample_input[key] = sample_input[key].to(device)

        # Định dạng lại input cho nhánh Translation của MSKA
        if 'keypoint' in sample_input:
            sample_input['video'] = sample_input['keypoint']
            # MSKA cần 'translation_inputs' để dịch câu
            # Ta giả lập cấu hình rỗng để model không báo lỗi thiếu key
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

        # Giải mã kết quả Dịch thuật (Text) - Dùng generate_txt của model.py
        if 'transformer_inputs' in output:
            gen_out = model.generate_txt(transformer_inputs=output['transformer_inputs'])
            text_prediction = gen_out['decoded_sequences']
        else:
            text_prediction = ["Dữ liệu không đủ để dịch thành câu"]

        print("\n" + "-"*40)
        print("PHÂN TÍCH NGÔN NGỮ KÝ HIỆU")
        print("-" * 40)    
        print(f"Từ khóa (Glosses): {' '.join(gloss_prediction)}")
        print(f"Dịch hoàn chỉnh:   {text_prediction[0]}")
        print("-"*40)

        # Lưu kết quả
        with open('result.txt', 'w', encoding='utf-8') as f:
            f.write(f"Glosses: {' '.join(gloss_prediction)}\n")
            f.write(f"Text: {text_prediction[0]}")

if __name__ == "__main__":
    # Nhập đường dẫn video từ người dùng
    video_path = input("Nhập đường dẫn video (ví dụ: video.mp4): ").strip()

    # Kiểm tra file tồn tại
    if not Path(video_path).exists():
        print(f"Lỗi: File '{video_path}' không tồn tại!")
        exit(1)

    print(f"Đang xử lý video: {video_path}")

    # Cấu hình
    config_path = './Khaosattongthe/configs/phoenix-2014t_s2t.yaml'
    test_file = 'temp_video.test'

    # Xử lý video
    try:
        name = extract_keypoints_from_video(video_path, test_file)
        config = update_config_test_path(config_path, test_file)
        run_model_on_test(config)
        print(f"\nHoàn thành xử lý video: {video_path}")
    except Exception as e:
        print(f"Lỗi khi xử lý video: {e}")
        exit(1)
