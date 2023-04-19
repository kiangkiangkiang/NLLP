import argparse

from paddlenlp import Taskflow

# "./data_v3/checkpoint/model_best"
"""
def inference(text: list):
    my_ie = Taskflow(
        "information_extraction", schema=args.inference_schema,
         task_path=args., precision="fp32"
    )
        parser.add_argument(
        "--inference_schema",
        default=["薪資費用", "醫療費用", "精神慰撫金額"],
        type=str,
        nargs="+",
        help="The schame in inference section (Use 'space' to split each ratio).",
    )
    pass
"""
# inference

# TODO read txt file which contain 'text' that would be inference
# simple_logger.debug(f"inference_schema={args.inference_schema}")
# inference()

def main():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="uie-base",
        type=str,
        help="Specific model or a best model path. Ex. ./checkpoint/model_best"
    )
    parser.add_argument(
        "--schema",
        default=["薪資費用", "醫療費用", "精神慰撫金額"],
        
    )
    main()