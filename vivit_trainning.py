import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import numpy as np
import os
import rasterio
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.transforms import functional as F
from torch.cuda.amp import autocast, GradScaler

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


# 사용자 정의 데이터셋 클래스 (nan/inf 픽셀을 평균으로 대체)
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

        print(f"--- 데이터 로딩 시작: {base_dir} ---")

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
                        print(f"  경고: 파일 로딩 오류 - {band_path}. 0 텐서로 대체합니다.")
                        all_bands_raw.append(None)
                else:
                    print(f"  경고: 파일 없음 - 필지 {field_name}, 밴드 {band_name}. 0 텐서로 대체합니다.")
                    all_bands_raw.append(None)

            valid_bands = [b for b in all_bands_raw if b is not None]
            if not valid_bands:
                print(f"  경고: 필지 {field_name}에 유효한 밴드 데이터가 없습니다. 이 필지를 건너뜁니다.")
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
        print(f"--- 데이터 로딩 완료: 총 {len(self.data)}개의 패치를 준비했습니다. ---")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_tensor = item['image']

        # nan/inf 값을 평균으로 대체
        if not torch.isfinite(image_tensor).all():
            print(f"경고: 데이터 인덱스 {idx}에서 nan/inf 값을 감지하여 평균으로 대체합니다.")
            for c in range(image_tensor.shape[2]):
                channel_data = image_tensor[:, :, c]
                if not torch.isfinite(channel_data).all():
                    mean_val = channel_data[torch.isfinite(channel_data)].mean()
                    image_tensor[:, :, c] = torch.nan_to_num(channel_data, nan=mean_val, posinf=mean_val,
                                                             neginf=mean_val)

        image_tensor = image_tensor.unsqueeze(0)
        yield_value = torch.tensor(item['yield'], dtype=torch.float32)

        return image_tensor, yield_value


# 메인 실행 블록
if __name__ == '__main__':
    DATA_DIR = 'data'
    TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train_data')
    TEST_DATA_DIR = os.path.join(DATA_DIR, 'test_data')
    CSV_PATH = os.path.join(DATA_DIR, 'yield_data.csv')

    print("=== 데이터셋 및 모델 초기화 ===")
    try:
        yield_data = pd.read_csv(CSV_PATH)
        yield_data['yield'] = yield_data['yield'].astype(float)
        yield_map = dict(zip(yield_data['field_id'], yield_data['yield']))
        print(f"  > '{CSV_PATH}' 파일에서 수확량 데이터 로딩 완료.")
    except FileNotFoundError:
        print(f"오류: '{CSV_PATH}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()

    try:
        train_filenames = [d for d in os.listdir(TRAIN_DATA_DIR) if os.path.isdir(os.path.join(TRAIN_DATA_DIR, d))]
        if not train_filenames:
            print(f"오류: '{TRAIN_DATA_DIR}' 폴더에 필지 디렉터리가 없습니다.")
            exit()
    except FileNotFoundError:
        print(f"오류: '{TRAIN_DATA_DIR}' 폴더를 찾을 수 없습니다. 폴더를 생성해주세요.")
        exit()

    try:
        test_filenames = [d for d in os.listdir(TEST_DATA_DIR) if os.path.isdir(os.path.join(TEST_DATA_DIR, d))]
        if not test_filenames:
            print(f"오류: '{TEST_DATA_DIR}' 폴더에 필지 디렉터리가 없습니다.")
            exit()
    except FileNotFoundError:
        print(f"오류: '{TEST_DATA_DIR}' 폴더를 찾을 수 없습니다. 폴더를 생성해주세요.")
        exit()

    BATCH_SIZE = 1

    train_dataset = RiceYieldDataset(TRAIN_DATA_DIR, train_filenames, yield_map)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = RiceYieldDataset(TEST_DATA_DIR, test_filenames, yield_map)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # === GPU 장치 설정 수정 ===
    # 사용할 GPU 번호 지정 (0, 1, 2, ...)
    # RTX 4090이 GPU1이므로 1로 설정합니다.
    device_id = 0
    if torch.cuda.is_available() and torch.cuda.device_count() > device_id:
        device = torch.device(f'cuda:{device_id}')
        print(f"--- GPU {device_id} ({torch.cuda.get_device_name(device_id)})를 사용합니다. ---")
    else:
        device = torch.device('cpu')
        print("--- 경고: 지정된 GPU를 사용할 수 없어 CPU로 학습합니다. ---")
    # ==========================

    model = ViViT().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)

    scaler = GradScaler()
    best_loss = float('inf')
    start_epoch = 0
    checkpoint_path = "checkpoint.pth"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"  > 체크포인트 '{checkpoint_path}'를 불러와 학습을 재개합니다.")
        print(f"  > Epoch {start_epoch}부터 시작합니다.")

    print("=== 학습 시작 ===")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        print(f"--- Epoch {epoch + 1}/{EPOCHS} 시작 ---")
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)  # set_to_none=True는 메모리 효율을 높입니다.

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)

            if torch.isnan(loss):
                print(f"  경고: 배치 [{i + 1}]에서 nan 손실이 발생했습니다. 이 배치는 건너뜁니다.")
                continue

            scaler.scale(loss).backward()

            # 경사 클리핑 (옵션, GradScaler와 함께 사용 시)
            # scaler.unscale_()를 먼저 호출하여 기울기를 원래 스케일로 되돌립니다.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"  배치 [{i + 1}/{len(train_loader)}], 배치 손실: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0

        scheduler.step(avg_train_loss)

        print(f'Epoch [{epoch + 1}/{EPOCHS}] 완료, 평균 훈련 손실: {avg_train_loss:.4f}')
        print(f'현재 학습률: {optimizer.param_groups[0]["lr"]:.6f}')

        if avg_train_loss < best_loss:
            print(f'평균 손실 {best_loss:.4f} -> {avg_train_loss:.4f}로 개선. 최적 모델을 저장합니다.')
            best_loss = avg_train_loss
            torch.save(model.state_dict(), 'best_model.pth')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
        }, checkpoint_path)

    print("=== 학습 완료 ===")
    print(f"최적 모델 가중치가 'best_model.pth'에 저장되었습니다.")

    print("=== 예측 시작 ===")
    model.eval()
    with torch.no_grad():
        test_predictions_per_field = []

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                predicted_yield = model(inputs)
            test_predictions_per_field.append(predicted_yield.item())

        test_predictions_per_field = np.array(test_predictions_per_field)

        final_predicted_yield = np.mean(test_predictions_per_field)

        actual_yield = test_dataset.data[0]['yield']

        print(f"  > 테스트 필지 코드: {test_filenames[0]}")
        print(f"  > 실제 수확량: {actual_yield:.2f} kg")
        print(f"  > 예측 수확량 (패치 평균): {final_predicted_yield:.2f} kg")
        print(f"  > 예측 오차: {abs(actual_yield - final_predicted_yield):.2f} kg")

        test_labels = np.array([actual_yield] * len(test_predictions_per_field))
        rmse = np.sqrt(np.mean((test_predictions_per_field - test_labels) ** 2))
        mape = np.mean(np.abs((test_labels - test_predictions_per_field) / (test_labels + 1e-8))) * 100

        print(f"\n  > 테스트 세트 RMSE: {rmse:.2f} kg")
        print(f"  > 테스트 세트 MAPE: {mape:.2f} %")

    print("=== 예측 완료 ===")