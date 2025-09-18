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
PATCH_SIZE = (1, 32, 32)
TIMESTEPS = 1
HEIGHT = 256
WIDTH = 256
CHANNELS = 5
NUM_ENCODER_LAYERS = 4
NUM_HEADS = 8
EMBED_DIM = 512
MLP_DIM = 2048


# ViViT 모델 클래스 (총 수확량을 예측하도록 사용됨)
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


# 사용자 정의 데이터셋 클래스 (총 수확량 데이터를 사용)
class RiceYieldDataset(Dataset):
    def __init__(self, base_dir, filenames, yield_map):
        self.base_dir = base_dir
        self.filenames = filenames
        self.yield_map = yield_map
        self.timesteps = TIMESTEPS
        self.height = HEIGHT
        self.width = WIDTH
        self.channels = CHANNELS
        self.band_abbreviations = {
            'Blue': 'Blue',
            'Green': 'Green',
            'Red': 'Red',
            'Red_Edge': 'RE',
            'NIR': 'NIR'
        }

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        field_name = self.filenames[idx]
        field_path = os.path.join(self.base_dir, field_name)

        bands_by_channel = []
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
                        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
                        resized_img_tensor = F.resize(img_tensor, (self.height, self.width))
                        bands_by_channel.append(resized_img_tensor.squeeze(0))
                except rasterio.errors.RasterioIOError:
                    print(f"Error loading {band_path}. Skipping...")
                    bands_by_channel.append(torch.zeros((self.height, self.width), dtype=torch.float32))
            else:
                print(f"File for field {field_name} and band {band_name} not found. Skipping...")
                bands_by_channel.append(torch.zeros((self.height, self.width), dtype=torch.float32))

        stacked_bands = torch.stack(bands_by_channel, dim=2)
        video_tensor = stacked_bands.unsqueeze(0)

        yield_value = torch.tensor(self.yield_map[field_name], dtype=torch.float32)

        # # ====== 추가된 출력문 ======
        # print(f"\n--- 데이터 로딩 확인 ---")
        # print(f"필지 코드: {field_name}")
        # print(f"불러온 이미지 텐서 크기: {video_tensor.shape}")
        # print(f"매핑된 수확량 값: {yield_value.item():.2f} kg")
        # print(f"----------------------\n")
        # # ============================

        return video_tensor, yield_value


# 메인 실행 블록
if __name__ == '__main__':
    DATA_DIR = 'data'
    CSV_PATH = os.path.join(DATA_DIR, 'yield_data.csv')

    try:
        yield_data = pd.read_csv(CSV_PATH)
        yield_data['yield'] = yield_data['yield'].astype(float)
        yield_map = dict(zip(yield_data['field_id'], yield_data['yield']))
    except FileNotFoundError:
        print(f"Error: {CSV_PATH} not found.")
        print("Please check the file path and file name.")
        exit()

    field_filenames = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

    if not field_filenames:
        print(f"Error: No field directories found in {DATA_DIR}.")
        exit()

    train_filenames, test_filenames = train_test_split(
        field_filenames, test_size=1, random_state=42
    )

    BATCH_SIZE = 1
    train_dataset = RiceYieldDataset(DATA_DIR, train_filenames, yield_map)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = RiceYieldDataset(DATA_DIR, test_filenames, yield_map)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ViViT().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("모델 학습 시작 (총 수확량 예측)...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}')

    print("모델 학습 완료!")
    print("-" * 30)

    print("예측 시작...")
    model.eval()
    with torch.no_grad():
        test_predictions = []
        test_labels = []

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predicted_yield = model(inputs)

            test_predictions.append(predicted_yield.item())
            test_labels.append(labels.item())

        test_predictions = np.array(test_predictions)
        test_labels = np.array(test_labels)

        rmse = np.sqrt(np.mean((test_predictions - test_labels) ** 2))
        mape = np.mean(np.abs((test_labels - test_predictions) / (test_labels + 1e-8))) * 100

        field_name = test_filenames[0]
        actual_yield = test_labels[0]
        predicted_yield = test_predictions[0]

        print(f"필지 코드: {field_name}")
        print(f"실제 수확량: {actual_yield:.2f} kg")
        print(f"예측 수확량: {predicted_yield:.2f} kg")
        print(f"예측 오차: {abs(actual_yield - predicted_yield):.2f} kg")
        print(f"\n테스트 세트 RMSE: {rmse:.2f} kg")
        print(f"테스트 세트 MAPE: {mape:.2f} %")

    print("예측 완료!")