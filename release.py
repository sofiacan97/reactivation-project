import os
import zipfile
import shutil

def create_clean_zip(zip_name='Reactivation_Project_Release.zip'):

    include_paths = [
        'src/', 
        'runner.py', 
        'predict.py', 
        'requirements.txt',
        'outputs/model/reactivation_bundle.joblib' 
    ]
    
    exclude_extensions = ('.pyc', '.pyo', '.pyd', '.log', '.xlsx', '.csv')
    exclude_folders = ['__pycache__', '.git', '.ipynb_checkpoints']

    print(f"Creating ZIP: {zip_name}...")

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for path in include_paths:
            if os.path.isfile(path):
                if not path.endswith(exclude_extensions):
                    zipf.write(path)
                    print(f"  Aggiunto file: {path}")
            
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    dirs[:] = [d for d in dirs if d not in exclude_folders]
                    
                    for file in files:
                        if not file.endswith(exclude_extensions):
                            file_path = os.path.join(root, file)

                            arcname = os.path.relpath(file_path, os.getcwd())
                            zipf.write(file_path, arcname)
                print(f"Creating folder: {path}")

    print("-" * 30)
    print(f"Done: {os.path.abspath(zip_name)}")

if __name__ == "__main__":
    create_clean_zip()
