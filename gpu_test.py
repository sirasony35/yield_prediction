import torch

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"총 {num_gpus}개의 GPU를 찾았습니다.")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("사용 가능한 GPU를 찾을 수 없습니다.")