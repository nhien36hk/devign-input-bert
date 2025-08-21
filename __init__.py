import sys
import os

# Tự động thêm project root vào sys.path khi import
def setup_kaggle_path():
    """Setup path cho Kaggle environment"""
    # Kiểm tra xem có đang chạy trên Kaggle không
    if os.path.exists('/kaggle/input'):
        kaggle_path = '/kaggle/input/embedding-kaggle/KaggleTrain'
        if kaggle_path not in sys.path:
            sys.path.insert(0, kaggle_path)
            print(f"✅ Added Kaggle path: {kaggle_path}")

# Tự động chạy khi import module này
setup_kaggle_path()
