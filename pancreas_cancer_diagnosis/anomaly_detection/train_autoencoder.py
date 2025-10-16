
import os
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# --- 프로젝트의 기존 모듈 임포트 ---
# 경로 설정을 위해 프로젝트 루트를 sys.path에 추가
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from pancreas_cancer_diagnosis.data.datamodule import SegmentationDataModule
from pancreas_cancer_diagnosis.segmentation.models.unet import UNet3D

# --- 가중치 손실 함수 정의 ---
class WeightedMSELoss(nn.Module):
    """췌장 영역에 가중치를 부여하는 MSE 손실 함수"""
    def __init__(self, pancreas_weight: float = 10.0):
        super().__init__()
        self.pancreas_weight = pancreas_weight
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, recon, original, pancreas_mask):
        # 픽셀별 MSE 계산
        loss = self.mse_loss(recon, original)

        # 가중치 맵 생성 (췌장 영역은 높은 가중치, 배경은 1)
        weight_map = torch.ones_like(pancreas_mask)
        weight_map[pancreas_mask > 0] = self.pancreas_weight

        # 가중치 적용
        weighted_loss = loss * weight_map

        # 전체 손실의 평균 계산
        return weighted_loss.mean()

# --- 메트릭 시각화 함수 ---
def plot_metrics(metrics, output_dir):
    """학습 메트릭을 그래프로 시각화"""
    epochs = range(1, len(metrics['train_loss']) + 1)

    # 2x3 subplot 생성
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Autoencoder Training Metrics', fontsize=16, fontweight='bold')

    # 1. Train/Val Loss
    ax = axes[0, 0]
    if len(metrics['train_loss']) > 0:
        ax.plot(epochs, metrics['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=8)
    if len(metrics['val_loss']) > 0:
        ax.plot(epochs, metrics['val_loss'], 'r-o', label='Val Loss', linewidth=2, markersize=8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. AUC
    ax = axes[0, 1]
    if len(metrics['auc']) > 0:
        ax.plot(epochs[:len(metrics['auc'])], metrics['auc'], 'g-o', linewidth=2, markersize=8)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title('AUC Score', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No AUC data', ha='center', va='center', fontsize=12)
        ax.set_title('AUC Score', fontsize=14, fontweight='bold')

    # 3. Sensitivity (Recall)
    ax = axes[0, 2]
    if len(metrics['sensitivity']) > 0:
        ax.plot(epochs[:len(metrics['sensitivity'])], metrics['sensitivity'], 'm-o', linewidth=2, markersize=8)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Sensitivity', fontsize=12)
        ax.set_title('Sensitivity (Recall)', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Sensitivity data', ha='center', va='center', fontsize=12)
        ax.set_title('Sensitivity (Recall)', fontsize=14, fontweight='bold')

    # 4. Specificity
    ax = axes[1, 0]
    if len(metrics['specificity']) > 0:
        ax.plot(epochs[:len(metrics['specificity'])], metrics['specificity'], 'c-o', linewidth=2, markersize=8)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Specificity', fontsize=12)
        ax.set_title('Specificity', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Specificity data', ha='center', va='center', fontsize=12)
        ax.set_title('Specificity', fontsize=14, fontweight='bold')

    # 5. Accuracy
    ax = axes[1, 1]
    if len(metrics['accuracy']) > 0:
        ax.plot(epochs[:len(metrics['accuracy'])], metrics['accuracy'], 'orange', marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Accuracy data', ha='center', va='center', fontsize=12)
        ax.set_title('Accuracy', fontsize=14, fontweight='bold')

    # 6. All metrics together
    ax = axes[1, 2]
    if len(metrics['auc']) > 0:
        ax.plot(epochs[:len(metrics['auc'])], metrics['auc'], 'g-o', label='AUC', linewidth=2, markersize=6)
    if len(metrics['sensitivity']) > 0:
        ax.plot(epochs[:len(metrics['sensitivity'])], metrics['sensitivity'], 'm-o', label='Sensitivity', linewidth=2, markersize=6)
    if len(metrics['specificity']) > 0:
        ax.plot(epochs[:len(metrics['specificity'])], metrics['specificity'], 'c-o', label='Specificity', linewidth=2, markersize=6)
    if len(metrics['accuracy']) > 0:
        ax.plot(epochs[:len(metrics['accuracy'])], metrics['accuracy'], 'orange', marker='o', label='Accuracy', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('All Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 저장
    save_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ 메트릭 그래프 저장: {save_path}")

    # 최종 메트릭 요약 출력
    print("\n" + "="*60)
    print("📊 최종 메트릭 요약")
    print("="*60)
    if len(metrics['auc']) > 0:
        print(f"최종 AUC:         {metrics['auc'][-1]:.4f}")
    if len(metrics['sensitivity']) > 0:
        print(f"최종 Sensitivity: {metrics['sensitivity'][-1]:.4f}")
    if len(metrics['specificity']) > 0:
        print(f"최종 Specificity: {metrics['specificity'][-1]:.4f}")
    if len(metrics['accuracy']) > 0:
        print(f"최종 Accuracy:    {metrics['accuracy'][-1]:.4f}")
    print("="*60 + "\n")

# --- 시각화 콜백 정의 ---
class VisualizationCallback(Callback):
    """검증 단계에서 원본/복원/오류 이미지를 저장하는 콜백"""
    def __init__(self, output_dir: str, freq: int = 5):
        super().__init__()
        self.output_dir = output_dir
        self.freq = freq
        os.makedirs(self.output_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        """검증 에폭이 끝날 때마다 호출"""
        if (trainer.current_epoch + 1) % self.freq != 0:
            return

        # 검증 데이터로더에서 한 배치 가져오기
        val_dataloader = trainer.datamodule.val_dataloader()
        if not val_dataloader:
            return
        
        try:
            batch = next(iter(val_dataloader))
        except StopIteration:
            return # 데이터로더가 비어있을 경우

        images, labels = batch['image'].to(pl_module.device), batch['label'].to(pl_module.device)

        # Ensure 5D tensors: [B, C, D, H, W]
        if images.ndim == 4:  # [B, D, H, W]
            images = images.unsqueeze(1)  # [B, 1, D, H, W]
        if labels.ndim == 4:  # [B, D, H, W]
            labels = labels.unsqueeze(1)  # [B, 1, D, H, W]

        # 한 개의 샘플 선택 (첫 번째 샘플)
        image = images[0].unsqueeze(0)
        label = labels[0].unsqueeze(0)

        # 모델 추론
        pl_module.eval()
        with torch.no_grad():
            recon = pl_module(image)
        pl_module.train()

        # CPU로 데이터 이동 및 numpy 변환
        image_np = image.cpu().numpy().squeeze()
        recon_np = recon.cpu().numpy().squeeze()
        label_np = label.cpu().numpy().squeeze()

        # 중앙 슬라이스 선택
        mid_slice_idx = image_np.shape[0] // 2
        img_slice = image_np[mid_slice_idx, :, :]
        recon_slice = recon_np[mid_slice_idx, :, :]
        label_slice = label_np[mid_slice_idx, :, :]
        error_map = np.abs(img_slice - recon_slice)

        # 시각화
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'Epoch {trainer.current_epoch + 1}', fontsize=16)

        axes[0].imshow(img_slice, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(recon_slice, cmap='gray')
        axes[1].set_title('Reconstructed Image')
        axes[1].axis('off')

        im = axes[2].imshow(error_map, cmap='hot')
        axes[2].set_title('Reconstruction Error')
        axes[2].axis('off')
        fig.colorbar(im, ax=axes[2])

        axes[3].imshow(label_slice, cmap='gray')
        axes[3].set_title('Pancreas Label')
        axes[3].axis('off')

        # 이미지 파일로 저장
        save_path = os.path.join(self.output_dir, f'epoch_{trainer.current_epoch + 1}.png')
        plt.savefig(save_path)
        plt.close(fig)
        print(f"\n✅ Visualization saved to {save_path}")

# --- PyTorch Lightning 모델 정의 ---
class LitAutoencoder(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, pancreas_weight=10.0):
        super().__init__()
        self.save_hyperparameters()

        # 기존 U-Net 모델을 오토인코더로 사용
        self.model = UNet3D(in_channels=1, num_classes=1) # 출력 채널을 1로 변경
        self.loss_fn = WeightedMSELoss(pancreas_weight=pancreas_weight)

        # 메트릭 저장을 위한 리스트
        self.validation_step_outputs = []
        self.epoch_metrics = {
            'train_loss': [],
            'val_loss': [],
            'auc': [],
            'sensitivity': [],
            'specificity': [],
            'accuracy': []
        }

    def forward(self, x):
        # U-Net의 출력이 복원된 이미지가 되도록 함
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']

        # Ensure 5D tensors: [B, C, D, H, W]
        if images.ndim == 4:  # [B, D, H, W]
            images = images.unsqueeze(1)  # [B, 1, D, H, W]
        if labels.ndim == 4:  # [B, D, H, W]
            labels = labels.unsqueeze(1)  # [B, 1, D, H, W]

        recons = self(images)
        loss = self.loss_fn(recons, images, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label']

        # Ensure 5D tensors: [B, C, D, H, W]
        if images.ndim == 4:  # [B, D, H, W]
            images = images.unsqueeze(1)  # [B, 1, D, H, W]
        if labels.ndim == 4:  # [B, D, H, W]
            labels = labels.unsqueeze(1)  # [B, 1, D, H, W]

        recons = self(images)
        loss = self.loss_fn(recons, images, labels)

        # Reconstruction error 계산 (anomaly score로 사용)
        recon_error = torch.mean((recons - images) ** 2, dim=[1, 2, 3, 4])  # [B]

        # 췌장 영역이 있으면 anomaly (class 1), 없으면 normal (class 0)
        # 실제로는 tumor 영역 기준이어야 하지만, 여기서는 췌장 존재 여부로 간단히 처리
        has_pancreas = (torch.sum(labels, dim=[1, 2, 3, 4]) > 0).float()

        self.validation_step_outputs.append({
            'recon_error': recon_error.cpu(),
            'labels': has_pancreas.cpu(),
            'loss': loss
        })

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        """검증 에폭 종료 시 메트릭 계산"""
        if len(self.validation_step_outputs) == 0:
            return

        # 모든 validation outputs 수집
        all_errors = torch.cat([x['recon_error'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()

        # 메트릭 계산 (최소 2개 이상의 샘플이 필요)
        if len(all_errors) >= 2 and len(torch.unique(all_labels)) >= 2:
            try:
                # AUC 계산
                auc = roc_auc_score(all_labels.numpy(), all_errors.numpy())

                # Threshold를 median으로 설정하여 binary prediction 생성
                threshold = torch.median(all_errors).item()
                predictions = (all_errors > threshold).float()

                # Confusion matrix 계산
                tn, fp, fn, tp = confusion_matrix(all_labels.numpy(), predictions.numpy()).ravel()

                # 메트릭 계산
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                # 메트릭 저장
                self.epoch_metrics['auc'].append(auc)
                self.epoch_metrics['sensitivity'].append(sensitivity)
                self.epoch_metrics['specificity'].append(specificity)
                self.epoch_metrics['accuracy'].append(accuracy)

                # 로깅
                self.log('auc', auc, prog_bar=True)
                self.log('sensitivity', sensitivity, prog_bar=True)
                self.log('specificity', specificity, prog_bar=True)
                self.log('accuracy', accuracy, prog_bar=True)

                print(f"\n📊 Epoch {self.current_epoch} Metrics:")
                print(f"   AUC: {auc:.4f}")
                print(f"   Sensitivity: {sensitivity:.4f}")
                print(f"   Specificity: {specificity:.4f}")
                print(f"   Accuracy: {accuracy:.4f}")

            except Exception as e:
                print(f"⚠️  메트릭 계산 실패: {e}")

        # 에폭 손실 저장
        train_loss_val = self.trainer.callback_metrics.get('train_loss_epoch', 0)
        train_loss_val = train_loss_val.item() if hasattr(train_loss_val, 'item') else train_loss_val
        self.epoch_metrics['train_loss'].append(train_loss_val)
        self.epoch_metrics['val_loss'].append(avg_loss.item())

        # outputs 초기화
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# --- 메인 실행 함수 ---
def main(args):
    # 데이터 모듈 준비 (dataset 내부에서 64x64x64로 리사이즈)
    dm = SegmentationDataModule(
        data_root=args.data_root,
        data_list_file=args.data_list_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=1, # 레이블의 클래스 수 (여기서는 췌장 하나만 필요)
        train_transform=None,  # dataset에서 자동 리사이즈
        val_transform=None
    )

    # 모델 초기화
    model = LitAutoencoder(
        learning_rate=args.learning_rate,
        pancreas_weight=args.pancreas_weight
    )

    # 콜백 준비
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, 'checkpoints'),
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    vis_callback = VisualizationCallback(
        output_dir=os.path.join(args.output_dir, 'visualizations')
    )

    # 트레이너 준비
    # 가속기 자동 선택 (CUDA만 사용, MPS는 max_pool3d 미지원)
    if torch.cuda.is_available():
        accelerator = 'gpu'
    else:
        # MPS는 3D 연산 미지원, CPU 사용
        accelerator = 'cpu'

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[checkpoint_callback, vis_callback],
        default_root_dir=args.output_dir
    )

    # 학습 시작
    print("--- 이상 탐지 오토인코더 학습 시작 ---")
    trainer.fit(model, dm)
    print("--- 학습 완료 ---")

    # 메트릭 그래프 생성
    print("\n📈 메트릭 그래프 생성 중...")
    plot_metrics(model.epoch_metrics, args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly Detection Autoencoder Training')

    # 데이터 관련 인자
    parser.add_argument('--data_root', type=str, default='data', help='NIfTI 파일이 있는 루트 디렉토리')
    parser.add_argument('--data_list_file', type=str, default='data/manifests/pancreas_ct_manifest.json', help='전체 데이터 목록 JSON')
    parser.add_argument('--batch_size', type=int, default=1, help='배치 크기')
    parser.add_argument('--num_workers', type=int, default=4, help='데이터 로더 워커 수')

    # 학습 관련 인자
    parser.add_argument('--epochs', type=int, default=50, help='총 학습 에폭')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='학습률')
    parser.add_argument('--pancreas_weight', type=float, default=10.0, help='췌장 영역 손실 가중치')

    # 출력 관련 인자
    parser.add_argument('--output_dir', type=str, default='outputs/anomaly_detection', help='학습 결과물 저장 디렉토리')

    args = parser.parse_args()

    main(args)
