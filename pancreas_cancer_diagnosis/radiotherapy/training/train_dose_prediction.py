"""
Dose Prediction 모델 학습 스크립트
==================================

3D 선량 분포 예측
"""

import os
import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger

from pancreas_cancer_diagnosis.radiotherapy.models import DosePredictionModel
from pancreas_cancer_diagnosis.radiotherapy.data import DosePredictionDataModule


def main(args):
    """메인 학습 함수"""

    # 시드 고정 (재현성)
    pl.seed_everything(args.seed)

    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 모듈 초기화
    print("\n" + "="*60)
    print("1. 데이터 로딩 중...")
    print("="*60)

    data_module = DosePredictionDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        spatial_size=tuple(args.spatial_size),
        num_oars=args.num_oars
    )

    # 2. 모델 초기화
    print("\n" + "="*60)
    print("2. 모델 초기화 중...")
    print("="*60)

    # in_channels = CT(1) + tumor(1) + OARs(num_oars)
    in_channels = 2 + args.num_oars

    model = DosePredictionModel(
        in_channels=in_channels,
        base_channels=args.base_channels,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

    print(f"✅ 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   입력 채널: {in_channels} (CT + tumor + {args.num_oars} OARs)")

    # 3. Callbacks 설정
    callbacks = [
        # 체크포인트 저장 (val_loss 기준)
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="dose-{epoch:02d}-{val/total_loss:.4f}",
            monitor="val/total_loss",
            mode="min",
            save_top_k=3,
            save_last=True
        ),
        # Early stopping
        EarlyStopping(
            monitor="val/total_loss",
            patience=args.patience,
            mode="min",
            verbose=True
        ),
        # Learning rate 모니터
        LearningRateMonitor(logging_interval="epoch")
    ]

    # 4. Logger 설정
    logger = TensorBoardLogger(
        save_dir=output_dir / "logs",
        name="dose_prediction"
    )

    # 5. Trainer 초기화
    print("\n" + "="*60)
    print("3. Trainer 초기화 중...")
    print("="*60)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else "auto",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=args.val_check_interval,
        gradient_clip_val=args.gradient_clip_val,
        precision=args.precision,
        deterministic=True
    )

    # 6. 학습 시작
    print("\n" + "="*60)
    print("4. 학습 시작!")
    print("="*60)
    print(f"출력 디렉토리: {output_dir}")
    print(f"배치 크기: {args.batch_size}")
    print(f"최대 에폭: {args.max_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Spatial size: {args.spatial_size}")
    print("="*60 + "\n")

    trainer.fit(model, data_module)

    # 7. 테스트 (옵션)
    if args.test:
        print("\n" + "="*60)
        print("5. 테스트 시작!")
        print("="*60)
        trainer.test(model, data_module, ckpt_path="best")

    print("\n" + "="*60)
    print("✅ 학습 완료!")
    print("="*60)
    print(f"체크포인트 저장 위치: {output_dir / 'checkpoints'}")
    print(f"로그 저장 위치: {output_dir / 'logs'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dose prediction model training")

    # 데이터 관련
    parser.add_argument("--data_root", type=str, default="data/radiotherapy",
                        help="데이터 루트 디렉토리")
    parser.add_argument("--output_dir", type=str, default="outputs/dose_prediction",
                        help="출력 디렉토리")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="배치 크기 (dose prediction은 메모리 많이 사용)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker 수")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[128, 128, 128],
                        help="이미지 크기 (D H W)")
    parser.add_argument("--num_oars", type=int, default=2,
                        help="OAR 개수 (duodenum, stomach)")

    # 모델 관련
    parser.add_argument("--base_channels", type=int, default=32,
                        help="Base 채널 수")

    # 학습 관련
    parser.add_argument("--max_epochs", type=int, default=150,
                        help="최대 에폭 수")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience")
    parser.add_argument("--val_check_interval", type=float, default=1.0,
                        help="Validation 체크 간격 (epoch 단위)")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0,
                        help="Gradient clipping")

    # 하드웨어 관련
    parser.add_argument("--gpus", type=int, default=1,
                        help="GPU 개수")
    parser.add_argument("--precision", type=str, default="32",
                        choices=["16", "32", "bf16"],
                        help="학습 precision")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # 기타
    parser.add_argument("--test", action="store_true",
                        help="학습 후 테스트 실행")

    args = parser.parse_args()
    main(args)
