"""
Multi-Task Learning 모델 학습 스크립트
======================================

Survival + Toxicity + Response 동시 예측
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

from pancreas_cancer_diagnosis.radiotherapy.models import MultiTaskRadiotherapyModel
from pancreas_cancer_diagnosis.radiotherapy.data import MultiTaskDataModule


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

    data_module = MultiTaskDataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        spatial_size=tuple(args.spatial_size)
    )

    # 2. 모델 초기화
    print("\n" + "="*60)
    print("2. 모델 초기화 중...")
    print("="*60)

    task_weights = {
        'survival': args.weight_survival,
        'toxicity': args.weight_toxicity,
        'response': args.weight_response
    }

    model = MultiTaskRadiotherapyModel(
        in_channels=1,
        base_channels=args.base_channels,
        num_blocks=4,
        clinical_features=10,
        num_toxicity_classes=4,
        task_weights=task_weights,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

    print(f"✅ 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Callbacks 설정
    callbacks = [
        # 체크포인트 저장 (val_loss 기준)
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="multitask-{epoch:02d}-{val/total_loss:.4f}",
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
        name="multitask"
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
    print(f"Task weights: {task_weights}")
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
    parser = argparse.ArgumentParser(description="Multi-task radiotherapy model training")

    # 데이터 관련
    parser.add_argument("--data_root", type=str, default="data/radiotherapy",
                        help="데이터 루트 디렉토리")
    parser.add_argument("--output_dir", type=str, default="outputs/multitask",
                        help="출력 디렉토리")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="배치 크기")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker 수")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[96, 96, 96],
                        help="이미지 크기 (D H W)")

    # 모델 관련
    parser.add_argument("--base_channels", type=int, default=32,
                        help="Base 채널 수")
    parser.add_argument("--weight_survival", type=float, default=1.0,
                        help="Survival loss weight")
    parser.add_argument("--weight_toxicity", type=float, default=1.0,
                        help="Toxicity loss weight")
    parser.add_argument("--weight_response", type=float, default=1.0,
                        help="Response loss weight")

    # 학습 관련
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="최대 에폭 수")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--patience", type=int, default=20,
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
