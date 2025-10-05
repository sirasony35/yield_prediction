import torch

# 파일 경로
problem_batch_path = 'problem_batch.pt'

try:
    # 1. 파일 불러오기
    problem_tensor = torch.load(problem_batch_path)
    print(f"✅ '{problem_batch_path}' 파일을 성공적으로 불러왔습니다.\n")

    # 2. 기본 정보 출력
    print("--- 기본 정보 ---")
    # 텐서의 형태 출력: (배치 사이즈, 시간, 높이, 너비, 채널)
    print(f"텐서 형태 (shape): {problem_tensor.shape}")

    # 3. 통계 정보 확인
    print("\n--- 통계 정보 ---")
    # 텐서 내의 값들이 정상 범위인지 확인합니다.
    print(f"최댓값 (max): {problem_tensor.max().item():.4f}")
    print(f"최솟값 (min): {problem_tensor.min().item():.4f}")
    print(f"평균값 (mean): {problem_tensor.mean().item():.4f}")
    print(f"표준편차 (std): {problem_tensor.std().item():.4f}")

    # 4. 비정상 값 포함 여부 확인
    print("\n--- 비정상 값 확인 ---")
    # 데이터 자체에 NaN 또는 Inf가 있는지 확인합니다.
    has_nan = torch.isnan(problem_tensor).any().item()
    has_inf = torch.isinf(problem_tensor).any().item()
    print(f"NaN 포함 여부: {has_nan}")
    print(f"Inf 포함 여부: {has_inf}")

except FileNotFoundError:
    print(f"'{problem_batch_path}' 파일을 찾을 수 없습니다. 학습 코드에서 파일이 생성되었는지 확인해주세요.")
except Exception as e:
    print(f"파일을 읽는 중 오류가 발생했습니다: {e}")