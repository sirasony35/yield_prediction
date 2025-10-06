import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import numpy as np
import os
import random
import rasterio
import pandas as pd
from torchvision.transforms import functional as F
from collections import defaultdict
from tqdm import tqdm


# --- 디버깅 (1): 재현성 확보를 위한 시드 고정 ---
def set_seed(seed):
    """모든 무작위성을 제어하여 실행 결과를 재현 가능하게 만듭니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(42)  # 언제나 동일한 결과를 위해 시드 고정

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


# --- 안정화 (1): 수정된 ViViT 모델 클래스 ---
class ViViT(nn.Module):
    def __init__(self, timesteps=TIMESTEPS, height=HEIGHT, width=WIDTH, channels=CHANNELS, patch_size=PATCH_SIZE,
                 num_encoder_layers=NUM_ENCODER_LAYERS, num_heads=NUM_HEADS, embed_dim=EMBED_DIM, mlp_dim=MLP_DIM):
        super().__init__()
        patch_t, patch_h, patch_w = patch_size
        num_patches = (timesteps // patch_t) * (height // patch_h) * (width // patch_w)

        self.tubelet_embedding = nn.Conv3d(in_channels=channels, out_channels=embed_dim, kernel_size=patch_size,
                                           stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            batch_first=True,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc_head = nn.Linear(embed_dim, 1)

        for name, layer in self.named_modules():
            if isinstance(layer, nn.LayerNorm):
                layer.eps = 1e-4

    def forward(self, x):
        x = rearrange(x, 'b t h w c -> b c t h w')
        x = self.tubelet_embedding(x)
        x = rearrange(x, 'b d t h w -> b (t h w) d')
        x += self.pos_embedding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc_head(x)
        return x


# --- 데이터 전처리: RiceYieldDataset 클래스 ---
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


# --- 안정화 (2): 가중치 초기화 함수 ---
def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Conv3d)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)


# --- 메인 실행 블록 ---
if __name__ == '__main__':
    # --- 1. 기본 설정 ---
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

    # --- 2. 하이퍼파라미터 및 성능 최적화 설정 ---
    EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.05
    NUM_WORKERS = 8

    # --- 3. 데이터 로더 생성 ---
    train_dataset = RiceYieldDataset(TRAIN_DATA_DIR, train_filenames, yield_map, is_train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_dataset = RiceYieldDataset(TEST_DATA_DIR, test_filenames, yield_map, is_train=False)
    test_dataset.set_stats(train_dataset.means, train_dataset.stds)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # --- 4. 모델 및 학습 구성 요소 초기화 ---
    print("\n--- GPU 확인 ---")
    if torch.cuda.is_available():
        device_id = 0 # 사용할 GPU 번호 (0번 GPU 사용)
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        print(f"✅ CUDA 사용 가능. 총 {torch.cuda.device_count()}개의 GPU 발견.")
        print(f"   -> 현재 장치: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("⚠️ 경고: CUDA를 사용할 수 없습니다. CPU로 학습을 진행합니다.")
    print("-----------------\n")

    model = ViViT().to(device)
    model.apply(weights_init)
    print("--- 모델 가중치 초기화 완료 ---")

    best_model_path = 'best_model.pth'
    checkpoint_path = 'checkpoint.pth'

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # --- 체크포인트 (1): 학습 상태 불러오기 ---
    if os.path.exists(checkpoint_path):
        print(f"--- '{checkpoint_path}' 파일에서 체크포인트를 불러옵니다. ---")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f"  -> 학습을 재개합니다. 시작 에포크: {start_epoch}")
    else:
        print("--- 체크포인트 파일이 없습니다. 처음부터 학습을 시작합니다. ---")
        start_epoch = 0
        best_loss = float('inf')

    # --- 5. 학습 루프 ---
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
                problem_batch_path = 'problem_batch.pt'
                if not os.path.exists(problem_batch_path):
                    print(f"  -> 문제가 발생한 배치를 '{problem_batch_path}' 파일로 저장합니다.")
                    torch.save(inputs.cpu(), problem_batch_path)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        scheduler.step(avg_loss)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] 완료, 평균 훈련 손실: {avg_loss:.4f}, 현재 학습률: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> 새로운 최적 모델 저장됨 (손실: {best_loss:.4f})")

        # --- 체크포인트 (2): 학습 상태 저장 ---
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
        }, checkpoint_path)

    print("=== 학습 완료 ===")

    # --- 6. 평가 루프 ---
    print("\n=== 예측 및 평가 시작 ===")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    predictions = defaultdict(list)
    actuals = {}
    with torch.no_grad():
        for inputs, labels, field_names in tqdm(test_loader, desc="평가 중"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            for idx, field_name in enumerate(field_names):
                predictions[field_name].append(outputs[idx].item())
                if field_name not in actuals:
                    actuals[field_name] = labels[idx].item()

    print("\n--- 필지별 예측 결과 ---")
    all_predictions, all_actuals = [], []
    for field_name, preds in predictions.items():
        avg_prediction = np.mean(preds)
        actual_yield = actuals[field_name]
        all_predictions.append(avg_prediction)
        all_actuals.append(actual_yield)
        print(
            f"  > 필지: {field_name}, 실제: {actual_yield:.2f} kg, 예측: {avg_prediction:.2f} kg, 오차: {abs(actual_yield - avg_prediction):.2f} kg")

    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    rmse = np.sqrt(np.mean((all_actuals - all_predictions) ** 2))
    mape = np.mean(np.abs((all_actuals - all_predictions) / (all_actuals + 1e-8))) * 100

    print("\n--- 전체 테스트 세트 최종 평가 ---")
    print(f"  > RMSE: {rmse:.2f} kg")
    print(f"  > MAPE: {mape:.2f} %")
    print("=== 평가 완료 ===")