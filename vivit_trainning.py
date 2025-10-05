import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import numpy as np
import os
import rasterio
import pandas as pd
from torchvision.transforms import functional as F
from collections import defaultdict
from tqdm import tqdm

# --- 하이퍼파라미터 정의 ---
PATCH_SIZE = (1, 128, 128)
TIMESTEPS = 1
HEIGHT = 128
WIDTH = 128
CHANNELS = 5
NUM_ENCODER_LAYERS = 4
NUM_HEADS = 8
EMBED_DIM = 512
MLP_DIM = 2048


# --- ViViT 모델 클래스 (전체 정의) ---
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

        self.tubelet_embedding = nn.Conv3d(in_channels=channels,
                                           out_channels=embed_dim,
                                           kernel_size=patch_size,
                                           stride=patch_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=mlp_dim,
                                                   batch_first=True,
                                                   dropout=0.1)

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


# --- 사용자 정의 데이터셋 클래스 (전체 정의) ---
class RiceYieldDataset(Dataset):
    def __init__(self, base_dir, filenames, yield_map, is_train=True):
        self.base_dir = base_dir
        self.filenames = filenames
        self.yield_map = yield_map
        self.patch_h = PATCH_SIZE[1]
        self.patch_w = PATCH_SIZE[2]
        self.band_abbreviations = {
            'Blue': 'Blue', 'Green': 'Green', 'Red': 'Red',
            'Red_Edge': 'RE', 'NIR': 'NIR'
        }

        if is_train:
            self.means, self.stds = self._calculate_stats()
        else:
            self.means = None
            self.stds = None

        print(f"--- 데이터 패치 생성 시작: {base_dir} ---")
        self.data = self._create_patches()
        print(f"--- 데이터 패치 생성 완료: 총 {len(self.data)}개 ---")

    def _calculate_stats(self):
        print("  > 훈련 데이터셋의 평균 및 표준편차를 계산합니다 (시간이 걸릴 수 있습니다)...")
        count = 0
        mean = torch.zeros(CHANNELS)
        std = torch.zeros(CHANNELS)

        for field_name in tqdm(self.filenames, desc="통계 계산 중"):
            field_path = os.path.join(self.base_dir, field_name)
            all_bands_raw = []
            image_files = [f for f in os.listdir(field_path) if f.endswith('.data.tif')]
            ordered_bands = ['Blue', 'Green', 'Red', 'Red_Edge', 'NIR']

            for band_name in ordered_bands:
                abbreviation = self.band_abbreviations[band_name]
                found_file = next((f for f in image_files if field_name in f and abbreviation in f), None)
                if found_file:
                    try:
                        with rasterio.open(os.path.join(field_path, found_file)) as src:
                            all_bands_raw.append(torch.from_numpy(src.read(1).astype(np.float32)))
                    except rasterio.errors.RasterioIOError:
                        all_bands_raw.append(None)
                else:
                    all_bands_raw.append(None)

            valid_bands = [b for b in all_bands_raw if b is not None]
            if not valid_bands: continue

            min_h = min(b.shape[0] for b in valid_bands)
            min_w = min(b.shape[1] for b in valid_bands)

            bands_by_channel = []
            for b in all_bands_raw:
                if b is not None:
                    bands_by_channel.append(F.center_crop(b.unsqueeze(0), (min_h, min_w)).squeeze(0))
                else:
                    bands_by_channel.append(torch.zeros((min_h, min_w), dtype=torch.float32))

            full_image_tensor = torch.stack(bands_by_channel, dim=-1)

            full_image_tensor = torch.nan_to_num(full_image_tensor, nan=0.0, posinf=0.0, neginf=0.0)

            mean += full_image_tensor.mean(dim=[0, 1])
            std += full_image_tensor.std(dim=[0, 1])
            count += 1

        if count == 0:
            return torch.zeros(CHANNELS), torch.ones(CHANNELS)

        final_mean = mean / count
        final_std = std / count

        print(f"  > 계산 완료!")
        print(f"  > 계산된 평균 (Means): {final_mean.tolist()}")
        print(f"  > 계산된 표준편차 (Stds): {final_std.tolist()}")

        return final_mean, final_std

    def _create_patches(self):
        data_list = []
        for field_name in tqdm(self.filenames, desc="데이터 패치 생성 중"):
            field_path = os.path.join(self.base_dir, field_name)
            all_bands_raw = []
            image_files = [f for f in os.listdir(field_path) if f.endswith('.data.tif')]
            ordered_bands = ['Blue', 'Green', 'Red', 'Red_Edge', 'NIR']
            for band_name in ordered_bands:
                abbreviation = self.band_abbreviations[band_name]
                found_file = next((f for f in image_files if field_name in f and abbreviation in f), None)
                if found_file:
                    with rasterio.open(os.path.join(field_path, found_file)) as src:
                        all_bands_raw.append(torch.from_numpy(src.read(1).astype(np.float32)))
                else:
                    all_bands_raw.append(None)

            valid_bands = [b for b in all_bands_raw if b is not None]
            if not valid_bands: continue

            min_h = min(b.shape[0] for b in valid_bands)
            min_w = min(b.shape[1] for b in valid_bands)
            bands_by_channel = []
            for b in all_bands_raw:
                if b is not None:
                    bands_by_channel.append(F.center_crop(b.unsqueeze(0), (min_h, min_w)).squeeze(0))
                else:
                    bands_by_channel.append(torch.zeros((min_h, min_w), dtype=torch.float32))
            full_image_tensor = torch.stack(bands_by_channel, dim=2)

            h, w, _ = full_image_tensor.shape
            for i in range(0, h, self.patch_h):
                for j in range(0, w, self.patch_w):
                    patch = full_image_tensor[i:i + self.patch_h, j:j + self.patch_w, :]
                    if patch.shape[0] == self.patch_h and patch.shape[1] == self.patch_w:
                        data_list.append(
                            {'image': patch, 'yield': self.yield_map[field_name], 'field_name': field_name})
        return data_list

    def set_stats(self, means, stds):
        self.means = means
        self.stds = stds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_tensor = item['image'].clone()
        if not torch.isfinite(image_tensor).all():
            for c in range(image_tensor.shape[2]):
                channel_data = image_tensor[:, :, c]
                if not torch.isfinite(channel_data).all():
                    finite_vals = channel_data[torch.isfinite(channel_data)]
                    mean_val = finite_vals.mean() if finite_vals.numel() > 0 else 0
                    image_tensor[:, :, c] = torch.nan_to_num(channel_data, nan=mean_val, posinf=mean_val,
                                                             neginf=mean_val)

        if self.means is not None and self.stds is not None:
            stds_eps = self.stds + 1e-6
            image_tensor = (image_tensor - self.means) / stds_eps

        image_tensor = image_tensor.unsqueeze(0)
        yield_value = torch.tensor(item['yield'], dtype=torch.float32)
        field_name = item['field_name']
        return image_tensor, yield_value, field_name


# --- 메인 실행 블록 ---
if __name__ == '__main__':
    DATA_DIR = 'data'
    TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train_data')
    TEST_DATA_DIR = os.path.join(DATA_DIR, 'test_data')
    CSV_PATH = os.path.join(DATA_DIR, 'yield_data.csv')

    print("=== 데이터셋 및 모델 초기화 ===")
    try:
        yield_data = pd.read_csv(CSV_PATH)
        yield_data.dropna(subset=['yield'], inplace=True)
        yield_map = dict(zip(yield_data['field_id'], yield_data['yield'].astype(float)))
    except FileNotFoundError:
        exit(f"오류: '{CSV_PATH}' 파일을 찾을 수 없습니다.")

    train_filenames = [d for d in os.listdir(TRAIN_DATA_DIR) if os.path.isdir(os.path.join(TRAIN_DATA_DIR, d))]
    test_filenames = [d for d in os.listdir(TEST_DATA_DIR) if os.path.isdir(os.path.join(TEST_DATA_DIR, d))]

    BATCH_SIZE = 4

    train_dataset = RiceYieldDataset(TRAIN_DATA_DIR, train_filenames, yield_map, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    test_dataset = RiceYieldDataset(TEST_DATA_DIR, test_filenames, yield_map, is_train=False)
    test_dataset.set_stats(train_dataset.means, train_dataset.stds)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"--- 장치: {device} ---")

    model = ViViT().to(device)
    best_model_path = 'best_model.pth'

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    criterion = nn.MSELoss()
    start_epoch = 0
    best_loss = float('inf')

    print("=== 학습 시작 ===")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0
        for i, (inputs, labels, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            if not torch.isfinite(loss):
                print(f"  경고: 배치 [{i + 1}]에서 nan/inf 손실이 발생하여 건너뜁니다.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] 완료, 평균 훈련 손실: {avg_loss:.4f}")

        # 가장 좋은 모델 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> 새로운 최적 모델 저장됨 (손실: {best_loss:.4f})")

    print("=== 학습 완료 ===")

    # --- 평가 로직 시작 ---
    print("\n=== 예측 및 평가 시작 ===")
    # 저장된 최고의 모델 가중치를 불러옵니다.
    model.load_state_dict(torch.load(best_model_path))
    model.eval()  # 모델을 평가 모드로 전환

    predictions = defaultdict(list)
    actuals = {}

    with torch.no_grad():  # 그래디언트 계산을 비활성화하여 메모리 사용량을 줄이고 계산 속도를 높입니다.
        for inputs, labels, field_names in tqdm(test_loader, desc="평가 중"):
            inputs = inputs.to(device)
            # 모델을 통해 예측값(output)을 얻습니다.
            outputs = model(inputs)

            # 배치 내의 각 데이터에 대해 필지 이름별로 예측값과 실제값을 저장합니다.
            for idx, field_name in enumerate(field_names):
                predictions[field_name].append(outputs[idx].item())
                if field_name not in actuals:
                    actuals[field_name] = labels[idx].item()

    print("\n--- 필지별 예측 결과 ---")
    all_predictions = []
    all_actuals = []

    for field_name, preds in predictions.items():
        # 각 필지별로 모든 패치의 예측값 평균을 계산합니다.
        avg_prediction = np.mean(preds)
        actual_yield = actuals[field_name]

        all_predictions.append(avg_prediction)
        all_actuals.append(actual_yield)

        print(
            f"  > 필지: {field_name}, 실제: {actual_yield:.2f} kg, 예측: {avg_prediction:.2f} kg, 오차: {abs(actual_yield - avg_prediction):.2f} kg")

    # 전체 테스트 세트에 대한 평가 지표를 계산합니다.
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((all_actuals - all_predictions) ** 2))

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((all_actuals - all_predictions) / (all_actuals + 1e-8))) * 100

    print("\n--- 전체 테스트 세트 최종 평가 ---")
    print(f"  > RMSE: {rmse:.2f} kg")
    print(f"  > MAPE: {mape:.2f} %")
    print("=== 평가 완료 ===")