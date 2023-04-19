import argparse
import logging
import os
import sys
from pdb import set_trace

# 理論上只要把label studio output export到data/raw_data，並改名ls_data.json就可以跑main了
# 不同實驗應該名稱改experiment_name應該就可以


def argument_handler(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--label_studio_output",
        default="./data/raw_data/ls_data.json",
        type=str,
        help="The exported file from label studio (JSON format only).",
    )
    parser.add_argument(
        "--doccano_ext_dir",
        default="./data/raw_data/doccano_ext.jsonl",
        type=str,
        help="The converted file from label-studio \
            to decanno (JSONL format only).",
    )
    parser.add_argument(
        "--splits_ration",
        default="0.7 0.2 0.1",
        type=str,
        help="The ration representing 'Training Data/Validation Data\
            /Testing Data' (Use 'space' to split each ratio).",
    )
    parser.add_argument(
        "--seed",
        default="123",
        type=str,
        help="Set seed in pipeline.",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["gpu", "cpu"],
        help="gpu or cpu for training section.",
    )
    parser.add_argument(
        "--experiment_name",
        default="V1",
        type=str,
        help="Specific the experiment name.",
    )
    parser.add_argument(
        "--max_seq_length",
        default="512",
        type=str,
        help="The length of max sequence in UIE model.",
    )
    parser.add_argument(
        "--batch_size",
        default="16",
        type=str,
        help="The batch size in training and evaluation.",
    )
    parser.add_argument(
        "--epochs",
        default="10",
        type=str,
        help="Number of epochs in funetune.",
    )
    parser.add_argument(
        "--learning_rate",
        default="1e-5",
        type=str,
        help="The learning rate in training.",
    )

    return parser


def main():
    # convert label-studio to doccano
    simple_logger.debug("===file location=" + os.getcwd())
    simple_logger.info("Converting label-studio file to doccano format...")
    os.system(
        f"python3 ./PaddleNLP/labelstudio2doccano.py \
            --labelstudio_file {args.label_studio_output} \
            --doccano_file {args.doccano_ext_dir}"
    )
    set_trace()
    # split data into train/valid/test
    simple_logger.info("Start spliting data...")
    os.system(
        f"python3 ./PaddleNLP/doccano.py \
            --doccano_file {args.doccano_ext_dir} \
            --task_type ext \
            --save_dir {experiment_path}\
            --seed {args.seed} \
            --splits {args.splits_ration} \
            --schema_lang ch \
            --negative_ratio 5 "
    )

    # start finetune
    simple_logger.info("Start finetune...")
    os.system(
        f"python3 ./PaddleNLP/finetune.py  \
            --device {args.device} \
            --logging_steps 10 \
            --save_steps 100 \
            --eval_steps 100 \
            --seed {args.seed} \
            --model_name_or_path uie-base  \
            --output_dir {experiment_path}/checkpoint/model_best\
            --train_path {experiment_path}/train.txt \
            --dev_path {experiment_path}/dev.txt  \
            --max_seq_length {args.max_seq_length}  \
            --per_device_eval_batch_size {args.batch_size}\
            --per_device_train_batch_size  {args.batch_size} \
            --num_train_epochs {args.epochs} \
            --learning_rate {args.learning_rate} \
            --label_names 'start_positions' 'end_positions' \
            --do_train \
            --do_eval \
            --do_export \
            --export_model_dir {experiment_path}/checkpoint/model_best \
            --overwrite_output_dir \
            --disable_tqdm True \
            --metric_for_best_model eval_f1 \
            --load_best_model_at_end  True \
            --save_total_limit 1"
    )


def eval():
    os.system(
        f"python3 ./PaddleNLP/evaluate.py \
            --model_path {experiment_path}/checkpoint/model_best \
            --test_path {experiment_path}/test.txt \
            --batch_size {args.batch_size} \
            --max_seq_len {args.max_seq_length}"
    )


if __name__ == "__main__":
    # log
    simple_logger = logging.getLogger("Experiment log")
    simple_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    simple_logger.addHandler(handler)

    # get argument
    parser = argument_handler(argparse.ArgumentParser())
    args = parser.parse_args()
    experiment_path = "./data/experiment/" + args.experiment_name

    # finetune
    simple_logger.info("Start finetune...")
    main()
    simple_logger.info("End finetune...")

    # evaluation on testing data
    simple_logger.info("Start evaluation...")
    eval()
    simple_logger.info("End evaluation...")
