import torch  # PyTorch 라이브러리 임포트
import torch.nn as nn  # 신경망 모듈 임포트
from torch.utils.data import Dataset, DataLoader  # 데이터셋 및 데이터로더 클래스 임포트
from einops import rearrange  # 텐서 차원 조작을 위한 einops 라이브러리 임포트
import numpy as np  # 수치 연산을 위한 NumPy 라이브러리 임포트
import os  # 파일 및 폴더 경로 관리를 위한 OS 모듈 임포트
import rasterio  # TIFF 파일과 같은 래스터 데이터를 읽기 위한 라이브러리 임포트
import pandas as pd  # CSV 파일 처리를 위한 Pandas 라이브러리 임포트
from sklearn.model_selection import train_test_split  # 데이터셋 분할을 위한 Scikit-learn 함수 임포트
from torchvision.transforms import functional as F  # 이미지 리사이징을 위한 torchvision.transforms 임포트

# 논문에서 사용된 하이퍼파라미터 정의
PATCH_SIZE = (1, 128, 128)
TIMESTEPS = 1
HEIGHT = 128
WIDTH = 128
CHANNELS = 5

NUM_ENCODER_LAYERS = 4
NUM_HEADS = 8
EMBED_DIM = 512
MLP_DIM = 2048


# ViViT 모델 클래스 (이전과 동일)
class ViViT(nn.Module):
    def __init__(self,
                 timesteps=TIMESTEPS,
                 height=HEIGHT,
                 width=WIDTH,
                 channels=CHANNELS,
                 patch_size=PATCH_SIZE,
                 num_encoder_layers=NUM_ENCODER_LAYERS,
                 num_heads=NUM_HEADS,
                 embed_dim=EMBED_DIM,
                 mlp_dim=MLP_DIM):
        super().__init__()

        patch_t, patch_h, patch_w = patch_size
        num_patches = (timesteps // patch_t) * (height // patch_h) * (width // patch_w)
        patch_dim = channels * patch_t * patch_h * patch_w

        self.tubelet_embedding = nn.Conv3d(in_channels=channels,
                                           out_channels=embed_dim,
                                           kernel_size=patch_size,
                                           stride=patch_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=mlp_dim,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.fc_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = rearrange(x, 'b t h w c -> b c t h w')
        x = self.tubelet_embedding(x)
        x = rearrange(x, 'b d t h w -> b (t h w) d')
        x += self.pos_embedding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc_head(x)
        return x


# 사용자 정의 데이터셋 클래스 (이전과 동일)
class RiceYieldDataset(Dataset):
    def __init__(self, base_dir, filenames, yield_map):
        self.base_dir = base_dir
        self.filenames = filenames
        self.yield_map = yield_map
        self.timesteps = TIMESTEPS
        self.channels = CHANNELS
        self.patch_h = PATCH_SIZE[1]
        self.patch_w = PATCH_SIZE[2]
        self.band_abbreviations = {
            'Blue': 'Blue', 'Green': 'Green', 'Red': 'Red',
            'Red_Edge': 'RE', 'NIR': 'NIR'
        }

        self.data = []
        for field_name in filenames:
            field_path = os.path.join(base_dir, field_name)
            all_bands_raw = []
            image_files = [f for f in os.listdir(field_path) if f.endswith('.data.tif')]
            ordered_bands = ['Blue', 'Green', 'Red', 'Red_Edge', 'NIR']

            for band_name in ordered_bands:
                abbreviation = self.band_abbreviations[band_name]
                found_file = next((f for f in image_files if field_name in f and abbreviation in f), None)

                if found_file:
                    band_path = os.path.join(field_path, found_file)
                    try:
                        with rasterio.open(band_path) as src:
                            img = src.read(1)
                            all_bands_raw.append(torch.from_numpy(img).float())
                    except rasterio.errors.RasterioIOError:
                        print(f"Error loading {band_path}. Skipping...")
                        all_bands_raw.append(None)
                else:
                    print(f"File for field {field_name} and band {band_name} not found. Skipping...")
                    all_bands_raw.append(None)

            valid_bands = [b for b in all_bands_raw if b is not None]
            if not valid_bands:
                print(f"No valid band data found for field {field_name}. Skipping field.")
                continue

            min_h = min(b.shape[0] for b in valid_bands)
            min_w = min(b.shape[1] for b in valid_bands)

            bands_by_channel = []
            for b in all_bands_raw:
                if b is not None:
                    cropped_band = F.center_crop(b.unsqueeze(0), (min_h, min_w)).squeeze(0)
                    bands_by_channel.append(cropped_band)
                else:
                    bands_by_channel.append(torch.zeros((min_h, min_w), dtype=torch.float32))

            full_image_tensor = torch.stack(bands_by_channel, dim=2)

            h, w, c = full_image_tensor.shape
            for i in range(0, h, self.patch_h):
                for j in range(0, w, self.patch_w):
                    patch = full_image_tensor[i:i + self.patch_h, j:j + self.patch_w, :]

                    if patch.shape[0] == self.patch_h and patch.shape[1] == self.patch_w:
                        self.data.append({
                            'image': patch,
                            'yield': self.yield_map[field_name]
                        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_tensor = item['image']

        image_tensor = image_tensor.unsqueeze(0)
        yield_value = torch.tensor(item['yield'], dtype=torch.float32)

        return image_tensor, yield_value


# 메인 실행 블록
if __name__ == '__main__':
    DATA_DIR = 'data'
    TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train_data')
    TEST_DATA_DIR = os.path.join(DATA_DIR, 'test_data')
    CSV_PATH = os.path.join(DATA_DIR, 'yield_data.csv')

    try:
        yield_data = pd.read_csv(CSV_PATH)
        yield_data['yield'] = yield_data['yield'].astype(float)
        yield_map = dict(zip(yield_data['field_id'], yield_data['yield']))
    except FileNotFoundError:
        print(f"Error: {CSV_PATH} not found.")
        print("Please ensure the CSV file is in the correct path.")
        exit()

    try:
        train_filenames = [d for d in os.listdir(TRAIN_DATA_DIR) if os.path.isdir(os.path.join(TRAIN_DATA_DIR, d))]
        if not train_filenames:
            print(f"Error: No field directories found in {TRAIN_DATA_DIR}.")
            exit()
    except FileNotFoundError:
        print(f"Error: {TRAIN_DATA_DIR} not found. Please create this folder.")
        exit()

    try:
        test_filenames = [d for d in os.listdir(TEST_DATA_DIR) if os.path.isdir(os.path.join(TEST_DATA_DIR, d))]
        if not test_filenames:
            print(f"Error: No field directories found in {TEST_DATA_DIR}.")
            exit()
    except FileNotFoundError:
        print(f"Error: {TEST_DATA_DIR} not found. Please create this folder.")
        exit()

    BATCH_SIZE = 1

    train_dataset = RiceYieldDataset(TRAIN_DATA_DIR, train_filenames, yield_map)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = RiceYieldDataset(TEST_DATA_DIR, test_filenames, yield_map)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    EPOCHS = 100
    # === 원래의 학습률로 되돌림 ===
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # === device를 CPU로 강제 설정 ===
    device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViViT().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # === 학습률 스케줄러 제거 ===
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    print("모델 학습 시작 (총 수확량 예측, 패치 기반)...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        print(f"--- Epoch {epoch + 1}/{EPOCHS} 시작 ---")
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()

            # === 경사 클리핑 제거 ===
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # === nan 감지 로직 제거 ===
            # if not torch.isfinite(loss):
            #     print(f"경고: 배치 [{i}/{len(train_loader)}]에서 손실이 nan 또는 inf입니다. 이 배치는 건너뜁니다.")
            #     continue

            optimizer.step()
            train_loss += loss.item()

            if i % 10 == 0:
                print(f"  배치 [{i}/{len(train_loader)}], 배치 손실: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # scheduler.step(avg_train_loss)

        print(f'Epoch [{epoch + 1}/{EPOCHS}] 완료, 평균 훈련 손실: {avg_train_loss:.4f}')
        print(f'현재 학습률: {optimizer.param_groups[0]["lr"]:.6f}')

    print("모델 학습 완료!")
    print("-" * 30)

    model_save_path = "vivit_trained_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"모델 가중치가 '{model_save_path}'에 저장되었습니다.")

    print("예측 시작...")
    model.eval()
    with torch.no_grad():
        test_predictions_per_field = []

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predicted_yield = model(inputs)
            test_predictions_per_field.append(predicted_yield.item())

        test_predictions_per_field = np.array(test_predictions_per_field)

        final_predicted_yield = np.mean(test_predictions_per_field)

        actual_yield = test_dataset.data[0]['yield']

        print(f"테스트 필지 코드: {test_filenames[0]}")
        print(f"실제 수확량: {actual_yield:.2f} kg")
        print(f"예측 수확량 (패치 평균): {final_predicted_yield:.2f} kg")
        print(f"예측 오차: {abs(actual_yield - final_predicted_yield):.2f} kg")

        test_labels = np.array([actual_yield] * len(test_predictions_per_field))
        rmse = np.sqrt(np.mean((test_predictions_per_field - test_labels) ** 2))
        mape = np.mean(np.abs((test_labels - test_predictions_per_field) / (test_labels + 1e-8))) * 100

        print(f"\n테스트 세트 RMSE: {rmse:.2f} kg")
        print(f"테스트 세트 MAPE: {mape:.2f} %")

    print("예측 완료!")