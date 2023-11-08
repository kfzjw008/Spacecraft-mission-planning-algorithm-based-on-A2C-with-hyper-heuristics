# 检查并删除现有模型文件的函数
import os


def delete_model_files(*file_paths):
    for file_path in file_paths:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted existing model file: {file_path}")